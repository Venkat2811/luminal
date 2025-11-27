# Kernel Generation: From Graphs to GPU Code

## Overview

Luminal's most complex feature is **automatic GPU kernel generation**. This document explains how Luminal transforms computation graphs into optimized Metal/CUDA kernels.

**Key Source Files:**
- `src/search.rs:54-326` - Graph translation to kernel IR
- `src/search.rs:393-608` - Code generation from IR
- `src/search.rs:1199-1660` - Kernel splitting and organization

## The Code Generation Pipeline

```
┌─────────────────────────────────────────────────────┐
│            Computation Graph                         │
│  [Add] → [Mul] → [SumReduce] → [Output]            │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│        1. Translate to GraphTerm IR                  │
│  Convert ops to loop-based representation           │
│  Source: src/search.rs:54-326                       │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│          2. Split into Kernels                       │
│  Determine kernel boundaries                        │
│  Source: src/search.rs:1199-1660                    │
└─────────────────────────────────────────────────────┐
                    ↓
┌─────────────────────────────────────────────────────┐
│          3. Generate Loop Structure                  │
│  Create grid/threadblock/thread loops               │
│  Source: src/search.rs:618-1158                     │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│       4. Generate Metal/CUDA Source                  │
│  Emit actual kernel code                            │
│  Source: src/search.rs:493-606                      │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│         5. Compile and Execute                       │
│  Metal/CUDA compiler → Binary kernel                │
└─────────────────────────────────────────────────────┘
```

## Kernel Intermediate Representation (GraphTerm)

### GraphTerm Enum

**Location:** `src/search.rs:328-357`

```rust
pub enum GraphTerm {
    /// Global memory buffer
    GMEM { label: Option<String> },

    /// Loop input (read from tensor)
    LoopIn {
        range: Expression,     // How many iterations
        stride: Expression,    // Memory stride per iteration
        marker: String,        // Debug label
    },

    /// Loop output (write to tensor)
    LoopOut {
        range: Expression,
        stride: Expression,
        marker: String,
    },

    // Arithmetic operations
    Add, Mul, Mod, Max, LessThan,

    // Unary operations
    Exp, Sin, Recip, Sqrt, Neg,

    // Shared memory operations
    SMEM,       // Shared memory buffer
    SMEMLoad,   // Load from global to shared memory
    SMEMRead,   // Read from shared memory
}
```

### Translation Example: Add Operation

**Input Graph:**
```
[Load A (shape: 3)]  [Load B (shape: 3)]
        │                     │
        └─────────┬───────────┘
                  ↓
               [Add]
                  ↓
            [Output (3)]
```

**Translated to GraphTerm:**
```
GMEM(label="A")          GMEM(label="B")
    │                        │
    ↓                        ↓
LoopIn(range=3, stride=z)   LoopIn(range=3, stride=z)
    │                        │
    └───────────┬────────────┘
                ↓
              Add
                ↓
    LoopOut(range=3, stride=z)
                ↓
        GMEM(label="Output")
```

**What this means:**
- `z` is the loop index variable
- `stride=z` means: "access memory at index z"
- Range=3 means: "loop from 0 to 3"

**Reference:** `src/search.rs:69-119` - Add/Mul/Binary translation

## Loop Hierarchy

GPU kernels execute with a 3-level hierarchy:

```
┌──────────────────────────────────────────────────────┐
│              Grid (Blocks)                            │
│  ┌────────────────────────────────────────────────┐  │
│  │         Threadblock (Threads)                  │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │          Thread (Loop)                   │  │  │
│  │  │  for i in 0..n {                        │  │  │
│  │  │      // Process elements                │  │  │
│  │  │  }                                       │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**Luminal's convention** (`src/search.rs:13-15`):
```rust
const GRID_DIMS: usize = 2;          // First 2 loop levels = grid
const THREADBLOCK_DIMS: usize = 2;   // Next 2 loop levels = threadblock
const MAX_THREADBLOCK_SIZE: usize = 1024;
```

**Example - 4D loop:**

```
LoopIn(range=128, ...)  → Grid dimension 0 (blockIdx.x)
  LoopIn(range=256, ...)  → Grid dimension 1 (blockIdx.y)
    LoopIn(range=32, ...)  → Threadblock dimension 0 (threadIdx.x)
      LoopIn(range=16, ...)  → Threadblock dimension 1 (threadIdx.y)
        LoopIn(range=4, ...)  → Actual thread-level loop (for i...)
```

## Generating Metal/CUDA Code

### The make_kernel Function

**Location:** `src/search.rs:618-1158`

This is the core code generator. It walks the GraphTerm graph and emits kernel source code.

**Simplified overview:**

```rust
fn make_kernel(
    kernel_graph: &StableGraph<(GraphTerm, usize), ()>,
    include_nodes: HashSet<NodeIndex>,
    node_to_var: &mut HashMap<NodeIndex, (usize, bool)>,
    prev_max_var: &mut usize,
    loop_levels: &mut Vec<Expression>,
    arch: &mut GPUArch,
) -> Option<Vec<String>> {
    let mut kernel_lines = vec![];

    for node in toposort_subset(kernel_graph, &include_nodes) {
        match term {
            GraphTerm::LoopIn { range, stride, .. } => {
                // Generate loop or use grid/threadblock index
            }
            GraphTerm::Add | GraphTerm::Mul | ... => {
                // Generate arithmetic operation
            }
            GraphTerm::LoopOut { range, stride, .. } => {
                // Generate output indexing
            }
            // ...
        }
    }

    Some(kernel_lines)
}
```

### Example: Simple Element-wise Kernel

**Graph:**
```rust
let a = cx.tensor(1024);
let b = cx.tensor(1024);
let c = (a + b) * 2.0;
```

**GraphTerm IR:**
```
GMEM("a") → LoopIn(range=1024, stride=z) ──┐
                                            ↓
GMEM("b") → LoopIn(range=1024, stride=z) → Add → Mul ← Constant(2.0)
                                                    ↓
                                         LoopOut(range=1024, stride=z)
                                                    ↓
                                                GMEM("output")
```

**Generated Metal Kernel:**

```metal
#include <metal_stdlib>
using namespace metal;

kernel void kernel0(
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    device float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    device float* output [[buffer(2)]]
) {
    // Loop level 0: Grid dimension
    int loop_a = blockIdx.x;  // range=1024, so grid size = 1024/blockDim

    // Loop level 1: Threadblock dimension
    int loop_b = threadIdx.x;

    // Calculate global index
    int idx = loop_a * blockDim.x + loop_b;
    if (idx >= 1024) return;

    // Pointer arithmetic for inputs
    device float* ptr_a = a + idx;
    device float* ptr_b = b + idx;

    // Load values
    float val_a = *ptr_a;
    float val_b = *ptr_b;

    // Compute: (a + b) * 2.0
    float tmp1 = val_a + val_b;
    float tmp2 = tmp1 * 2.0;

    // Store result
    device float* ptr_out = output + idx;
    *ptr_out = tmp2;
}
```

**Reference:** `src/search.rs:493-587` - Metal kernel template generation

### Handling Reductions

Reductions (SumReduce, MaxReduce) are more complex.

**Example: Sum reduction along dimension**

```rust
// Sum over dimension 1
let a = cx.tensor((128, 256));  // Shape: (128, 256)
let sum = a.sum_reduce(1);      // Result: (128,)
```

**GraphTerm IR:**
```
GMEM("acc") ← Accumulator initialized to 0
    ↓
LoopIn(range=128, ...)  ← Iterate over first dimension
    ↓
  LoopIn(range=256, ..., stride=Acc('a'))  ← Reduce over second dimension
      ↓
    Add ← Reduction operation
      ↓
  LoopOut(range=256, ..., stride=Acc('a'))
    ↓
LoopOut(range=128, ...)
    ↓
GMEM("output")
```

**Key innovation - Accumulators:**

**Location:** `src/search.rs:186-311`

```rust
// Create accumulator for reduction
let acc_label = format!("acc_{}", accumulators.len());
accumulators.push((acc_label.clone(), start_val));

let acc = new_graph.add_node(GraphTerm::GMEM {
    label: Some(acc_label),
});

// Accumulator has special stride: Acc('a')
// This means: "this is a reduction accumulator"
```

**Generated kernel** (simplified):

```metal
kernel void sum_reduce(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Outer dimension
    if (i >= 128) return;

    // Thread-local accumulator
    float acc = 0.0;

    // Reduce over inner dimension
    for (int j = 0; j < 256; j++) {
        acc += input[i * 256 + j];
    }

    // Write result
    output[i] = acc;
}
```

**Reference:** `src/search.rs:707-802` - Accumulator generation

## Kernel Splitting

Not all operations can go in one kernel. Luminal **automatically splits** the graph into multiple kernels.

**Location:** `src/search.rs:1199-1660` (split_kernels function)

### Why Split?

1. **Cross-kernel dependencies**: Some operations need synchronization
2. **Memory barriers**: Need to ensure writes finish before reads
3. **Shared memory**: Limited to threadblock scope
4. **Kernel complexity**: Keep kernels manageable

### Split Example

```
Original graph:
    [Input] → [MatMul] → [Softmax] → [MatMul] → [Output]

Split into 3 kernels:
    Kernel 0: [Input] → [MatMul] → [Temp1]
    Kernel 1: [Temp1] → [Softmax] → [Temp2]
    Kernel 2: [Temp2] → [MatMul] → [Output]

Why? Each MatMul needs different grid/threadblock configuration
```

### Split Criteria

**Location:** `src/search.rs:1286-1299`

```rust
// Split when:
// 1. LoopOut → LoopIn transition at grid/threadblock level
if matches!(term, GraphTerm::LoopIn { .. })
    && matches!(src_term, GraphTerm::LoopOut { .. })
    && curr_level.len() < GRID_DIMS + THREADBLOCK_DIMS
{
    // Create new kernel!
    let max_kernel = *src_kernel.iter().max().unwrap();
    n_kernels = n_kernels.max(max_kernel + 2);
    *src_kernel = vec![max_kernel + 1];
}
```

**What this means:**
- If a LoopOut feeds into a LoopIn
- And this happens at the grid/threadblock level (not thread level)
- → Need kernel boundary (can't change grid size within a kernel)

### Kernel Metadata

Each kernel has metadata for execution:

**Location:** `src/search.rs:44-52`

```rust
pub struct Kernel {
    pub code: String,                    // Source code
    pub grid: (Expression, Expression, Expression),         // Grid dimensions
    pub threadblock: (Expression, Expression, Expression),  // Threadblock dimensions
    pub smem: Expression,                // Shared memory size
    pub outputs: Vec<Expression>,        // Output buffer sizes
}
```

**Example:**
```rust
Kernel {
    code: "kernel void kernel0(...) { ... }",
    grid: (128, 1, 1),           // 128 blocks, 1D grid
    threadblock: (256, 1, 1),    // 256 threads per block
    smem: 0,                     // No shared memory
    outputs: vec![128*256*4],    // One output buffer, size in bytes
}
```

## Shared Memory Optimization

For operations that benefit from shared memory (e.g., matrix multiply):

**Location:** `src/search.rs:1038-1087`

### What is Shared Memory?

```
GPU Memory Hierarchy:
┌────────────────────────────────────┐
│     Global Memory (GMEM)           │  Slowest, largest
│     ~10-100 GB, ~500 GB/s          │
└────────────────────────────────────┘
              ↑ ↓
┌────────────────────────────────────┐
│   Shared Memory (SMEM/Threadgroup) │  Fast, small
│   ~64 KB per block, ~1 TB/s        │
└────────────────────────────────────┘
              ↑ ↓
┌────────────────────────────────────┐
│        Registers                   │  Fastest, tiny
│   ~64 KB per thread, ~10 TB/s      │
└────────────────────────────────────┘
```

### Shared Memory Pattern

```
Without shared memory:
    Thread 0: Load A[0] from global memory
    Thread 1: Load A[0] from global memory  ← Redundant!
    Thread 2: Load A[0] from global memory  ← Redundant!

With shared memory:
    Thread 0: Load A[0] to shared memory
    __syncthreads()  ← Barrier
    All threads: Read A[0] from shared memory  ← Fast!
```

### Generated Code

```metal
kernel void matmul_shared(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    threadgroup float* shared_mem [[threadgroup(0)]]  ← Shared memory
) {
    // Partition shared memory
    threadgroup float* shared_A = shared_mem;
    threadgroup float* shared_B = shared_mem + TILE_SIZE*TILE_SIZE;

    // Cooperatively load tile into shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared_A[threadIdx.x] = A[...];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute using shared memory
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += shared_A[...] * shared_B[...];
    }

    C[...] = sum;
}
```

**Reference:** `src/search.rs:1578-1612` - Shared memory buffer setup

## Dynamic Dimensions

Luminal supports **symbolic dimensions** that are resolved at runtime.

**Location:** `src/shape/symbolic.rs`

### Example

```rust
// Define symbolic dimension
let batch_size = cx.dyn_dim('b');

// Create tensor with symbolic shape
let input = cx.tensor(('b', 512));

// Set dimension at runtime
cx.set_dyn_dim('b', 32);

// Generated kernel uses runtime value:
kernel void process(
    device float* input [[buffer(0)]],
    device int& b [[buffer(1)]],  ← Passed as parameter
    ...
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= b * 512) return;
    // ...
}
```

**Benefits:**
- Write generic code
- Compile once
- Run with different batch sizes

**Reference:** `src/search.rs:353-362` - Dynamic dimension inputs

## Putting It All Together

### Full Compilation Example

```rust
use luminal::prelude::*;
use luminal_metal::prelude::*;

fn main() {
    let mut cx = Graph::new();

    // Build computation
    let a = cx.tensor((128, 256)).set(vec![...]);
    let b = cx.tensor((256, 512)).set(vec![...]);
    let mut c = (a.matmul(b)).relu().retrieve();

    // Compile
    cx.compile(MetalCompiler::<f32>::default(), &mut c);

    // What happened:
    // 1. MatMul expanded to reshape + mul + reduce
    // 2. Translated to GraphTerm IR
    // 3. Split into kernels:
    //    - Kernel 0: MatMul (uses shared memory)
    //    - Kernel 1: ReLU (fused with matmul if possible)
    // 4. Generated Metal source code
    // 5. Metal compiler compiled to GPU binary
    // 6. Kernels ready to execute

    cx.execute();
    // Executes compiled kernels on GPU
}
```

### Generated Kernel Structure (Metal)

```metal
#include <metal_stdlib>
using namespace metal;

kernel void kernel0(
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    device float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    threadgroup float* sm [[threadgroup(0)]]
) {
    // Shared memory setup
    threadgroup float* shared_A = sm;
    threadgroup float* shared_B = sm + 4096;

    // Matrix multiply with tiling
    int row = blockIdx.x * 16 + threadIdx.x;
    int col = blockIdx.y * 16 + threadIdx.y;

    float sum = 0.0;
    for (int tile = 0; tile < 16; tile++) {
        // Load tile to shared memory
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared_A[threadIdx.x * 16 + threadIdx.y] = a[row * 256 + tile*16 + threadIdx.y];
        shared_B[threadIdx.x * 16 + threadIdx.y] = b[(tile*16 + threadIdx.x) * 512 + col];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute tile
        for (int k = 0; k < 16; k++) {
            sum += shared_A[threadIdx.x * 16 + k] * shared_B[k * 16 + threadIdx.y];
        }
    }

    // Write result
    if (row < 128 && col < 512) {
        output[row * 512 + col] = sum;
    }
}

kernel void kernel1(
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 128 * 512) return;

    // ReLU: max(0, x)
    float val = input[idx];
    output[idx] = fmax(0.0, val);
}
```

## Summary

Kernel generation in Luminal:

1. **Graph → GraphTerm IR**
   - Translate ops to loop-based representation
   - Express memory access patterns

2. **Kernel Splitting**
   - Determine kernel boundaries
   - Handle dependencies

3. **Loop Structure**
   - Map loops to grid/threadblock/thread hierarchy
   - Optimize parallelism

4. **Code Generation**
   - Emit Metal/CUDA source code
   - Handle shared memory, accumulators, etc.

5. **Compilation & Execution**
   - Metal/CUDA compiler generates binary
   - Execute on GPU

**The result:** Automatically generated, optimized GPU kernels from high-level operations!

## Conclusion

This documentation suite has covered:
- [Overview](./01-overview.md) - Luminal as a compiler
- [Graph Architecture](./02-graph-architecture.md) - Computation graphs
- [Compilation Pipeline](./03-compilation-pipeline.md) - Transformation stages
- [Performance](./04-performance.md) - Speed optimizations
- **Kernel Generation** (this document) - GPU code generation

Together, these techniques enable Luminal to achieve **"speed of light"** performance while remaining simple and maintainable.
