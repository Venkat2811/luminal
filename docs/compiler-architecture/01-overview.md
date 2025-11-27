# Overview: Luminal as a Compiler

## What is Luminal?

Luminal is a **deep learning library that uses search-based compilation** to achieve high performance. Unlike traditional eager-execution frameworks (PyTorch, TensorFlow 1.x), Luminal is a **compiler-first** framework that treats neural networks as computation graphs to be optimized and compiled.

**Key Source Files:**
- `src/graph.rs:14-38` - Core `Graph` structure
- `src/op.rs:79-90` - `Operator` trait definition
- `src/compiler_utils.rs:175-179` - `Compiler` trait

## Core Philosophy

### 1. Ahead-of-Time Compilation

```rust
// When you write this in Luminal:
let mut cx = Graph::new();
let a = cx.tensor((3, 1)).set([[1.0], [2.0], [3.0]]);
let b = cx.tensor((1, 4)).set([[1.0, 2.0, 3.0, 4.0]]);
let mut c = a.matmul(b).retrieve();

// NO COMPUTATION HAPPENS HERE!
// Instead, a computation graph is built:
//
//     [Load A]    [Load B]
//         ↓           ↓
//         └─────┬─────┘
//               ↓
//           [MatMul]
//               ↓
//           [Store C]
```

**Reference:** `src/graph.rs:115-130` - Graph tensor creation

### 2. Lazy Execution

Operations don't execute immediately. They're recorded in a graph, then:

```
User Code → Build Graph → Compile → Execute
```

**Only when `cx.execute()` is called** does computation happen:

```rust
cx.compile(<(GenericCompiler, CPUCompiler)>::default(), &mut c);
cx.execute();  // ← Computation happens here
```

**Reference:** `src/graph.rs:132-138` - Compile method
**Reference:** `src/graph.rs:190-224` - Execute method

## Architecture Overview

### The Compilation Stack

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  User writes neural network code using high-level ops       │
│  (matmul, conv2d, layer_norm, attention, etc.)             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   High-Level Operations                      │
│  Complex ops built from primitive ops                       │
│  Source: src/hl_ops/                                        │
│  - matmul, reduction, binary, unary, movement ops           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  12 Primitive Operations                     │
│  Source: src/op.rs:197-410                                  │
│                                                              │
│  Unary:  Log2, Exp2, Sin, Sqrt, Recip                      │
│  Binary: Add, Mul, Mod, LessThan                           │
│  Reduce: SumReduce, MaxReduce                              │
│  Other:  Contiguous                                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   Computation Graph (DAG)                    │
│  Source: src/graph.rs:14-38                                 │
│                                                              │
│  - Nodes: Operations (Box<dyn Operator>)                   │
│  - Edges: Data dependencies with shapes                     │
│  - Storage: petgraph::StableGraph                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│               Generic Compilation Passes                     │
│  Source: src/generic_compiler.rs                            │
│                                                              │
│  1. Common Subexpression Elimination (CSE)                  │
│  2. Dead Code Elimination (RemoveUnusedNodes)              │
│  3. Arithmetic Simplification                               │
│     - x + 0 → x                                            │
│     - x * 1 → x                                            │
│     - recip(recip(x)) → x                                  │
│     - exp2(log2(x)) → x                                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│            Device-Specific Compilation                       │
│  Sources:                                                    │
│  - crates/luminal_cpu/src/lib.rs                           │
│  - crates/luminal_metal/src/lib.rs                         │
│  - crates/luminal_cuda/src/lib.rs                          │
│                                                              │
│  Converts primitive ops to device-specific kernels          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   Kernel Fusion                              │
│  Source: crates/luminal_metal/src/elementwise_fusion.rs    │
│                                                              │
│  Combines multiple operations into single GPU kernels       │
│  Example: Add + Mul + Exp2 → single fused kernel           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Search-Based Optimization                       │
│  Source: src/search.rs                                      │
│                                                              │
│  Uses e-graphs to search for optimal implementations        │
│  Can automatically discover Flash Attention and similar     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   Code Generation                            │
│  Source: src/search.rs:393-608 (codegen function)          │
│                                                              │
│  Generates Metal/CUDA kernel source code                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                      Execution                               │
│  GPU kernels execute on Metal/CUDA devices                  │
└─────────────────────────────────────────────────────────────┘
```

## The 12 Primitive Operations

Luminal uses a **RISC-style architecture** with only 12 primitive operations:

### Unary Operations (A → A)
```rust
// src/op.rs:217-289
Log2      // Logarithm base 2
Exp2      // Exponential base 2
Sin       // Sine function
Sqrt      // Square root
Recip     // Reciprocal (1/x)
```

### Binary Operations (A × A → A)
```rust
// src/op.rs:294-356
Add       // Element-wise addition
Mul       // Element-wise multiplication
Mod       // Modulo operation
LessThan  // Element-wise comparison
```

### Reduction Operations (A → B)
```rust
// src/op.rs:360-409
SumReduce(dim)  // Sum along dimension
MaxReduce(dim)  // Max along dimension
```

### Memory Operations
```rust
// src/op.rs:199-214
Contiguous      // Ensure contiguous memory layout
```

### Why Only 12 Operations?

1. **Simplicity**: Easy to understand, test, and maintain
2. **Composability**: Complex operations built from these primitives
3. **Optimization**: Easier to optimize a small set comprehensively
4. **Compilation**: Simpler to generate efficient GPU kernels
5. **Search**: Smaller search space for optimization

Example - Softmax built from primitives:
```rust
// softmax(x) = exp(x) / sum(exp(x))
fn softmax(x: GraphTensor) -> GraphTensor {
    let max_val = x.max_reduce();          // MaxReduce
    let shifted = x - max_val;             // Add (with negation)
    let exp_x = shifted.exp2();            // Exp2 (after log2 conversion)
    let sum_exp = exp_x.sum_reduce();      // SumReduce
    exp_x / sum_exp                        // Mul (with recip)
}
```

## Key Data Structures

### 1. Graph
**Location:** `src/graph.rs:14-38`

```rust
pub struct Graph {
    /// Storage for tensor data
    pub tensors: FxHashMap<(NodeIndex, u8), Tensor>,

    /// Dynamic dimension mappings
    pub dyn_map: FxHashMap<char, usize>,

    /// The computation graph (DAG)
    pub graph: StorageGraph,  // StableGraph<Box<dyn Operator>, Dependency>

    /// Tensors that won't be deleted during execution
    pub no_delete: FxHashSet<NodeIndex>,

    /// Cached execution order
    linearized_graph: Option<Vec<(NodeIndex, Vec<(NodeIndex, u8, ShapeTracker)>)>>,
}
```

### 2. Operator Trait
**Location:** `src/op.rs:79-90`

```rust
pub trait Operator: Debug + as_any::AsAny {
    /// Process input tensors and produce output tensors
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor>;

    /// Custom functionality hook
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        None
    }
}
```

### 3. Dependency
**Location:** `src/graph.rs:39-72`

```rust
pub enum Dependency {
    /// Data dependency (tensor transfer)
    Data {
        input_order: u8,
        output_order: u8,
        shape: ShapeTracker,
    },

    /// Explicit ordering dependency (no data transfer)
    Schedule,
}
```

### 4. ShapeTracker
**Location:** `src/shape/tracker.rs`

A powerful system for tracking tensor views without copying data:
- Handles reshapes, permutes, slices, expands
- Computes indexing expressions for lazy evaluation
- Enables zero-copy tensor operations

## Compilation Flow Example

Let's trace a simple matrix multiplication:

```rust
let mut cx = Graph::new();
let a = cx.tensor((2, 3));
let b = cx.tensor((3, 4));
let c = a.matmul(b);
```

**Step 1: Graph Construction**
```
[Load A (2,3)]    [Load B (3,4)]
       ↓                 ↓
       └────────┬────────┘
                ↓
           [MatMul]
                ↓
         [Result (2,4)]
```

**Step 2: High-Level → Primitive Expansion**

MatMul expands to:
```
[Load A]  [Load B]
   ↓         ↓
   └───┬─────┘
       ↓
   [Reshape]
       ↓
     [Mul]
       ↓
  [SumReduce]
       ↓
   [Reshape]
```

**Step 3: Generic Optimization**
- Common subexpression elimination
- Dead code removal
- Arithmetic simplification

**Step 4: Device Compilation**
- Replace primitive ops with Metal/CUDA kernels
- Fuse compatible operations

**Step 5: Execution**
- Topological sort of graph
- Execute each kernel in order
- Manage memory lifetimes

**Reference:** `src/graph.rs:190-224` - Execute implementation

## Performance Characteristics

### Why This Design is Fast

1. **No Python Overhead**: Pure Rust, compiled to native code
2. **Lazy Evaluation**: Only compute what's needed
3. **Kernel Fusion**: Reduce memory bandwidth by fusing ops
4. **View-Based Operations**: Avoid copies via ShapeTracker
5. **Search-Based Optimization**: Discover optimal implementations
6. **Static Compilation**: All graph structure known at compile time

### Memory Management

Luminal tracks tensor consumers to delete tensors as soon as possible:

```rust
// src/graph.rs:151-164
// Count consumers per tensor output
self.consumers_map = Some(
    self.graph.node_indices()
        .flat_map(|i| {
            // Count how many times each output is used
        })
        .collect()
);

// During execution, delete tensors when consumer count hits 0
// src/graph.rs:218-221
for (id, ind, _) in src_ids {
    *consumers.get_mut(&(*id, *ind)).unwrap() -= 1;
}
```

## Next Steps

- [Graph Architecture](./02-graph-architecture.md) - Detailed look at the computation graph
- [Compilation Pipeline](./03-compilation-pipeline.md) - How compilation works
- [Performance Optimizations](./04-performance.md) - Speed optimization techniques
- [Kernel Generation](./05-kernel-generation.md) - GPU code generation
