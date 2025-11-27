# Compilation Pipeline: Transforming Graphs

## Overview

Compilation in Luminal is a **multi-stage transformation process** that converts high-level operations into optimized, device-specific kernels. This document explains each compilation stage.

**Key Source Files:**
- `src/compiler_utils.rs:175-315` - Compiler trait and infrastructure
- `src/generic_compiler.rs` - Platform-agnostic optimizations
- `crates/luminal_metal/src/lib.rs` - Metal-specific compilation
- `crates/luminal_cuda/src/lib.rs` - CUDA-specific compilation

## The Compiler Trait

All compilation passes implement the `Compiler` trait:

**Location:** `src/compiler_utils.rs:175-179`

```rust
pub trait Compiler {
    type Output;

    /// Run a compilation pass on the graph
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, ids: T) -> Self::Output;
}
```

**Key insight:** Compilers **mutate the graph in-place**, transforming operations while preserving semantics.

## Composing Compilers

Compilers can be composed using tuples:

```rust
// src/compiler_utils.rs:246-315
impl<M1: Compiler, M2: Compiler> Compiler for (M1, M2) {
    type Output = (M1::Output, M2::Output);

    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) -> Self::Output {
        (
            self.0.compile(graph, &mut remap),  // Run first compiler
            self.1.compile(graph, &mut remap),  // Run second compiler
        )
    }
}
```

**Example:**

```rust
type MyCompiler = (
    GenericCompiler,      // Run generic optimizations
    MetalCompiler<f16>,   // Compile to Metal with f16 precision
);

cx.compile(MyCompiler::default(), &mut outputs);
```

## Full Compilation Stack

### Typical Metal Compilation Pipeline

```rust
// crates/luminal_metal/src/lib.rs:32-46
pub type MetalCompiler<T> = (
    MetalCompilerPreBuffer<T>,  // All passes before buffer management
    BufferCompilers,             // Buffer sharing passes
);

pub type MetalCompilerPreBuffer<T> = (
    prim::PrimitiveCompiler<T>,           // 1. Convert primitives to Metal
    SpecialOpsCompiler<T>,                // 2. Replace with specialized ops
    other::CopyCompiler<T>,               // 3. Handle copy operations
    elementwise_fusion::ElementwiseFusionCompiler<T>,  // 4. Fuse element-wise ops
);

pub type BufferCompilers = (
    command_buffer::CommandBufferCompiler,   // 5. Share command buffers
    storage_buffer::StorageBufferCompiler,   // 6. Share storage buffers
);
```

**Execution order:** Left to right in the tuple.

### Visual Flow

```
User Code
    ↓
┌────────────────────────────────────────────────────┐
│          High-Level Graph Construction             │
│  MatMul, Conv2d, LayerNorm, etc.                  │
└────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────┐
│         Generic Compilation (Platform-Agnostic)    │
│                                                     │
│  1. RemoveUnusedNodes                             │
│     • Delete ops with no consumers                │
│                                                     │
│  2. ArithmeticElimination                         │
│     • x + 0 → x                                   │
│     • x * 1 → x                                   │
│     • recip(recip(x)) → x                         │
│     • exp2(log2(x)) → x                           │
│                                                     │
│  3. Common Subexpression Elimination (CSE)        │
│     • Detect duplicate subgraphs                  │
│     • Merge identical operations                  │
└────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────┐
│         Device-Specific Compilation                │
│                                                     │
│  1. PrimitiveCompiler                             │
│     • Primitive ops → Metal/CUDA kernels          │
│                                                     │
│  2. SpecialOpsCompiler                            │
│     • Replace ops with optimized variants         │
│     • Example: Subtraction, Gather, etc.          │
│                                                     │
│  3. CopyCompiler                                  │
│     • Insert explicit copy operations             │
│                                                     │
│  4. ElementwiseFusionCompiler                     │
│     • Fuse compatible element-wise operations     │
│     • Example: Add + Mul + Exp2 → single kernel   │
└────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────┐
│            Buffer Management                        │
│                                                     │
│  1. CommandBufferCompiler                          │
│     • Share Metal command buffers                  │
│     • Reduce API overhead                          │
│                                                     │
│  2. StorageBufferCompiler                          │
│     • Share intermediate storage buffers           │
│     • Reduce memory allocations                    │
└────────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────────┐
│              Search-Based Optimization              │
│  (Optional, for advanced cases)                    │
│                                                     │
│  • Uses e-graphs to search transformation space   │
│  • Can discover Flash Attention automatically      │
└────────────────────────────────────────────────────┘
    ↓
    Optimized Executable Graph
```

## Stage 1: Generic Optimizations

**Location:** `src/generic_compiler.rs`

These optimizations work on all platforms.

### 1.1 Common Subexpression Elimination (CSE)

**Location:** `src/generic_compiler.rs:24-101`

Finds and merges duplicate computations:

```rust
pub struct CSE;  // Common Subexpression Elimination

impl Compiler for CSE {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut ids: T) {
        // Find nodes with identical inputs and operations
        // Merge them into a single node
    }
}
```

**Example:**

```
Before CSE:
    [Input A]  [Input B]
       │  │       │  │
       │  └───┬───┘  │
       │      ↓      │
       │   [Add #1]  │
       │             │
       └──────┬──────┘
              ↓
           [Add #2]  ← Same inputs, same op!

After CSE:
    [Input A]  [Input B]
       │          │
       └────┬─────┘
            ↓
          [Add]  ← Single shared node
            │
            ├──→ Consumer 1
            └──→ Consumer 2
```

**How it works:**
1. Group nodes by their source nodes
2. If two nodes have identical sources and operations, merge them
3. Transfer outgoing edges from duplicate to original
4. Update references in the `ids` remap

### 1.2 Remove Unused Nodes

**Location:** `src/generic_compiler.rs:160-177`

Deletes operations that produce results nobody uses:

```rust
pub struct RemoveUnusedNodes;

impl Compiler for RemoveUnusedNodes {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _: T) {
        // Reverse topological sort
        for node in toposort(&graph.graph, None).unwrap().into_iter().rev() {
            if graph.edges_directed(node, Direction::Outgoing).count() == 0
                && !graph.no_delete.contains(&node)
            {
                graph.remove_node(node);  // No consumers, delete!
            }
        }
    }
}
```

**Example:**

```
Before:
    [A] → [B] → [C] → [Output]
          │
          └─→ [D] → [nowhere]  ← No consumers, delete!

After:
    [A] → [B] → [C] → [Output]
```

### 1.3 Arithmetic Elimination

**Location:** `src/generic_compiler.rs:236-488`

Simplifies arithmetic patterns:

```rust
pub struct ArithmeticElimination;
```

**Patterns eliminated:**

```
x + 0 → x
x * 1 → x
0 + x → x
1 * x → x
recip(recip(x)) → x
exp2(log2(x)) → x
log2(exp2(x)) → x
```

**Implementation uses pattern matching:**

```rust
// Pattern: x + 0
let zero = constant(0.);
let inp = node();
let add = binary::<Add>(zero.clone(), inp.clone());

let mut searcher = add.search(graph);
while searcher.next_match() {
    let (inp, zero, add) = (
        searcher.get(&inp),
        searcher.get(&zero),
        searcher.get(&add)
    );

    // Replace add with direct input
    move_outgoing_edge(add, inp, &mut graph.graph);
    remap(add, inp, &mut ids, graph);
    graph.graph.remove_node(add);
}
```

**Reference:** `src/generic_compiler.rs:247-311` - `x + 0` elimination

## Stage 2: Device-Specific Compilation

### 2.1 Primitive Compiler

Converts primitive operations to device-specific kernels.

**Metal Example:** `crates/luminal_metal/src/prim.rs`

```rust
pub struct PrimitiveCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for PrimitiveCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        // Find all Add operations
        let mut searcher = op::<Add>().search(graph);
        while searcher.next_match() {
            let add_node = searcher.get(&pattern);

            // Replace with MetalAdd kernel
            let metal_add = graph.add_op(MetalAdd::<T>::new());
            move_outgoing_edge(add_node, metal_add, graph);
            remap(add_node, metal_add, &mut remap, graph);
            graph.remove_node(add_node);
        }

        // ... repeat for all primitive operations
    }
}
```

**Transformation:**

```
Before:
    [Input A] → [Add (CPU)] → [Output]
                  ↑
    [Input B] ────┘

After:
    [Input A] → [MetalAdd] → [Output]
                  ↑
    [Input B] ────┘
```

### 2.2 Special Ops Compiler

Replaces operations with specialized, optimized variants.

**Location:** `crates/luminal_metal/src/lib.rs:48-59`

```rust
pub type SpecialOpsCompiler<T> = (
    binary::MetalSubtractionCompiler<T>,  // a - b
    binary::MetalEqualCompiler<T>,         // a == b
    other::ARangeCompiler<T>,              // range(n)
    binary::MetalGatherCompiler<T>,        // gather operation
    unary::MetalExpCompiler<T>,            // exp(x)
    unary::MetalCosCompiler<T>,            // cos(x)
    unary::MeanReduceCompiler<T>,          // mean along dim
    unary::StdNormCompiler<T>,             // std normalization
    matmul::MetalMatMulCompiler<T>,        // matrix multiply
);
```

**Example - Matrix Multiplication Specialization:**

```
Before (composed of primitives):
    [A] → [Reshape] → [Mul] → [SumReduce] → [Reshape] → [Output]
            ↑           ↑
    [B] ────┴───────────┘

After (specialized kernel):
    [A] → [MetalMatMul] → [Output]
            ↑
    [B] ────┘
```

**Why?** Specialized kernels can:
- Use hardware-specific instructions (e.g., Tensor Cores)
- Optimize memory access patterns
- Reduce kernel launch overhead

### 2.3 Elementwise Fusion Compiler

Fuses multiple element-wise operations into a single kernel.

**Location:** `crates/luminal_metal/src/elementwise_fusion.rs`

```rust
pub struct ElementwiseFusionCompiler<T>(PhantomData<T>);
```

**Example fusion:**

```
Before:
    [Input] → [Mul by 2] → [Add 1] → [Exp2] → [Output]

After:
    [Input] → [FusedKernel: x * 2 + 1, then exp2] → [Output]

Generated kernel code:
    kernel void fused_kernel(device float* input, device float* output) {
        uint idx = thread_position_in_grid.x;
        float x = input[idx];
        x = x * 2.0;
        x = x + 1.0;
        x = exp2(x);
        output[idx] = x;
    }
```

**Benefits:**
1. **Reduced memory bandwidth**: Intermediate results stay in registers
2. **Fewer kernel launches**: One kernel instead of three
3. **Better instruction-level parallelism**: GPU can optimize fused operations

**How it works:**
1. Find chains of element-wise operations
2. Generate single kernel with inline operations
3. Replace the chain with the fused kernel

## Stage 3: Buffer Management

### 3.1 Command Buffer Sharing

**Location:** `crates/luminal_metal/src/command_buffer.rs`

Metal kernels need command buffers for execution. Instead of creating one per kernel:

```rust
Before:
    Kernel A: create command buffer → execute → destroy
    Kernel B: create command buffer → execute → destroy
    Kernel C: create command buffer → execute → destroy

After:
    Create shared command buffer
    Kernel A: use shared buffer → execute
    Kernel B: use shared buffer → execute
    Kernel C: use shared buffer → execute
    Destroy shared command buffer
```

**Benefits:**
- Reduced API overhead
- Better batching of GPU work

### 3.2 Storage Buffer Sharing

**Location:** `crates/luminal_metal/src/storage_buffer.rs`

Intermediate buffers can be reused if their lifetimes don't overlap:

```
Timeline:
    ┌─────┐
    │Buf A│ (used by op 1)
    └─────┘
           ┌─────┐
           │Buf B│ (used by op 2)  ← Can reuse Buf A's memory!
           └─────┘
                  ┌─────┐
                  │Buf C│ (used by op 3)  ← Can reuse Buf A's memory!
                  └─────┘
```

**Implementation:**
1. Analyze buffer lifetimes
2. Create interference graph (which buffers overlap in time)
3. Color the graph to find reusable buffers
4. Share buffers with non-overlapping lifetimes

## Stage 4: Search-Based Optimization (Advanced)

**Location:** `src/search.rs`

For complex optimizations, Luminal can use **equality saturation** with e-graphs to search for optimal transformations.

### E-Graph Concept

An e-graph represents multiple equivalent expressions simultaneously:

```
E-class #1: {
    x * 2,
    2 * x,
    x + x,
    x << 1  (if integer)
}

All these expressions are semantically equivalent!
```

### Search Process

```
┌────────────────────────────────────────────┐
│ 1. Initial Graph                           │
│    [A] → [Mul 2] → [Add 1] → [Output]    │
└────────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────────┐
│ 2. Convert to E-Graph                      │
│    Represents all equivalent forms         │
└────────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────────┐
│ 3. Apply Rewrite Rules                     │
│    • Associativity: (a + b) + c = a + (b+c)│
│    • Distributivity: a*(b+c) = a*b + a*c   │
│    • Strength reduction: x*2 = x+x         │
│    • Custom patterns: matmul chains        │
└────────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────────┐
│ 4. Extract Optimal Solution                │
│    Choose cheapest equivalent expression   │
└────────────────────────────────────────────┘
```

### Discovering Flash Attention

Flash Attention is a complex optimization for the attention mechanism:

```
Naive Attention:
    Q @ K.T → Softmax → @ V
    ↑ Materializes large QK matrix

Flash Attention:
    Computes attention in blocks, never materializing QK
    Much lower memory usage, better cache locality
```

**Luminal can discover this automatically** by searching through equivalent implementations and finding the one with lowest memory cost.

**Reference:** README mentions this capability

## Compilation Timing

You can measure compilation time with the `Timed` wrapper:

**Location:** `src/compiler_utils.rs:211-244`

```rust
use luminal::prelude::Timed;

cx.compile(
    Timed(GenericCompiler::default()),
    &mut outputs
);

// Output:
// Starting GenericCompiler
// Finished GenericCompiler in 45ms
```

## Looped Compilation

Some optimizations need to run until convergence:

**Location:** `src/compiler_utils.rs:187-209`

```rust
use luminal::prelude::Looped;

cx.compile(
    Looped(ArithmeticElimination::default()),
    &mut outputs
);

// Runs ArithmeticElimination repeatedly until graph stops changing
```

**Why?** Some patterns only emerge after other optimizations:

```
Initial:  a + (b + 0)
Pass 1:   a + b        (eliminate b + 0)
Pass 2:   a + b        (no change, done!)
```

## Practical Example: Full Compilation

```rust
use luminal::prelude::*;
use luminal_metal::prelude::*;

fn main() {
    let mut cx = Graph::new();

    // Build model
    let input = cx.tensor((128, 512));
    let weights = cx.tensor((512, 256));
    let mut output = input.matmul(weights).relu().retrieve();

    // Compile
    type FullCompiler = (
        GenericCompiler,           // Generic optimizations
        MetalCompiler<f16>,        // Metal-specific compilation
    );

    cx.compile(FullCompiler::default(), &mut output);

    // What happened during compilation:
    // 1. GenericCompiler:
    //    - Removed dead code
    //    - Eliminated x + 0, x * 1 patterns
    //    - Merged duplicate computations
    //
    // 2. MetalCompiler:
    //    - Converted Add → MetalAdd, Mul → MetalMul
    //    - Replaced MatMul chain with MetalMatMul
    //    - Fused MatMul + ReLU into single kernel
    //    - Shared command and storage buffers

    // Execute
    cx.execute();
}
```

## Summary

The compilation pipeline transforms graphs through multiple stages:

1. **Generic Optimizations**
   - CSE, dead code elimination, arithmetic simplification
   - Platform-independent

2. **Device Compilation**
   - Convert primitives to device kernels
   - Apply specialized operations
   - Fuse element-wise operations

3. **Buffer Management**
   - Share command buffers
   - Reuse storage buffers

4. **Search (Optional)**
   - E-graph equality saturation
   - Discover complex optimizations

Each stage preserves semantics while improving performance!

**Next:** [Performance Optimizations](./04-performance.md)
