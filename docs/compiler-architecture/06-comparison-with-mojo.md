# Luminal vs Modular Mojo: Compiler Architecture Comparison

## Overview

Both Luminal and Mojo are **compiler-based approaches** to high-performance ML, but they take fundamentally different paths to achieve performance. This document compares their architectures, design philosophies, and trade-offs.

## Quick Comparison Table

| Aspect | Luminal | Mojo |
|--------|---------|------|
| **Language** | Rust library | New language (Python superset) |
| **Compilation** | Search-based (e-graphs) | MLIR-based multi-level |
| **Primitive Ops** | 12 RISC-style ops | MLIR dialects (100+ ops) |
| **Target Users** | Systems programmers | Python ML practitioners |
| **IR** | Custom GraphTerm + e-graphs | MLIR (industry standard) |
| **Code Generation** | Direct Metal/CUDA | LLVM → Native/GPU |
| **Optimization** | Automatic search | Hand-tuned + auto-vectorization |
| **Python Compat** | None (pure Rust) | Full Python superset |
| **Maturity** | Experimental | Production-ready |
| **License** | Open source (MIT/Apache) | Proprietary (free to use) |

## Architectural Differences

### 1. Language Design Philosophy

**Luminal:**
```
┌─────────────────────────────────────┐
│    Application Layer (Rust)         │
│  Pure Rust, no Python involvement   │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│   Luminal Library (Rust)            │
│  Graph construction + compilation   │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│   Metal/CUDA Kernels                │
└─────────────────────────────────────┘
```

**Mojo:**
```
┌─────────────────────────────────────┐
│    Application Layer (Mojo)         │
│  Python-like syntax + new features  │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│   Mojo Compiler                     │
│  Type inference, ownership, etc.    │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│   MLIR (Multi-Level IR)             │
│  High-level → Low-level transforms  │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│   LLVM → Native Code                │
└─────────────────────────────────────┘
```

**Key Difference:**
- **Luminal**: Library embedded in existing language (Rust)
- **Mojo**: Entirely new language with custom compiler

### 2. Intermediate Representation

**Luminal's IR Stack:**

```
User Code (Rust)
    ↓
Computation Graph (petgraph DAG)
    ↓
GraphTerm IR (loop-based)
    ↓
E-Graph (equality saturation)
    ↓
Metal/CUDA Source Code
```

**Example - Luminal IR:**
```rust
// GraphTerm representation
GMEM("input")
  → LoopIn(range=1024, stride=z)
    → Add
      → LoopOut(range=1024, stride=z)
        → GMEM("output")
```

**Mojo's IR Stack:**

```
Mojo Source Code
    ↓
Mojo IR (high-level)
    ↓
MLIR Dialects:
  - linalg (linear algebra)
  - tensor (tensor operations)
  - arith (arithmetic)
  - scf (control flow)
  - gpu (GPU ops)
    ↓
LLVM IR
    ↓
Machine Code
```

**Example - MLIR representation:**
```mlir
// MLIR linalg dialect
%result = linalg.matmul
  ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
  outs(%C : tensor<128x512xf32>)
  -> tensor<128x512xf32>

// Lower to loop form
scf.for %i = %c0 to %c128 step %c1 {
  scf.for %j = %c0 to %c512 step %c1 {
    // ...
  }
}
```

**Key Difference:**
- **Luminal**: Custom minimal IR, search-based optimization
- **Mojo**: MLIR (mature, industry-standard, multi-level)

### 3. Optimization Strategy

**Luminal: Search-Based Optimization**

```
┌──────────────────────────────────────┐
│  Define Rewrite Rules                │
│  • Associativity                     │
│  • Commutativity                     │
│  • Fusion patterns                   │
│  • Domain-specific (Flash Attention) │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│  E-Graph Construction                │
│  Apply all rules exhaustively        │
│  Build equivalence classes           │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│  Extraction                          │
│  Choose minimal-cost implementation  │
│  cost = f(memory, compute, launches) │
└──────────────────────────────────────┘
```

**Advantage:** Can discover novel optimizations automatically (e.g., Flash Attention)

**Disadvantage:** Expensive compilation time, no guarantees on finding global optimum

**Mojo: MLIR Pattern Matching + Passes**

```
┌──────────────────────────────────────┐
│  High-Level MLIR                     │
│  (linalg, tensor ops)                │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│  Transformation Passes               │
│  • Tiling                            │
│  • Fusion                            │
│  • Vectorization                     │
│  • Loop unrolling                    │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│  Low-Level MLIR                      │
│  (scf, arith, memref)                │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│  LLVM Optimizations                  │
│  • Auto-vectorization                │
│  • Instruction selection             │
└──────────────────────────────────────┘
```

**Advantage:** Mature, proven optimization pipeline. Fast compilation.

**Disadvantage:** Relies on hand-written passes, may miss novel optimizations

### 4. Code Example Comparison

**Luminal (Rust):**

```rust
use luminal::prelude::*;
use luminal_metal::prelude::*;

fn main() {
    let mut cx = Graph::new();

    // Define computation
    let a = cx.tensor((128, 256));
    let b = cx.tensor((256, 512));
    let mut c = a.matmul(b).relu().retrieve();

    // Compile for Metal
    cx.compile(MetalCompiler::<f16>::default(), &mut c);

    // Execute
    cx.execute();
}
```

**What happens:**
1. Builds computation graph
2. Compiles to Metal kernels
3. Optimizes with search
4. Generates GPU code
5. Executes

**Mojo:**

```python
from tensor import Tensor
from algorithm import vectorize

fn matmul_relu[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    # Matrix multiply
    let c = a @ b

    # ReLU (vectorized)
    @parameter
    fn relu_vectorized[width: Int](i: Int):
        c.simd_store[width](i, max(c.simd_load[width](i), 0))

    vectorize[relu_vectorized, simd_width](c.num_elements())
    return c

fn main():
    let a = Tensor[DType.float16](128, 256)
    let b = Tensor[DType.float16](256, 512)
    let result = matmul_relu(a, b)
```

**What happens:**
1. Type checking and inference
2. MLIR lowering
3. Optimization passes (tiling, fusion, vectorization)
4. LLVM code generation
5. Executes

**Key Difference:**
- **Luminal**: Graph construction, then compile everything
- **Mojo**: Progressive lowering through MLIR levels

## Primitive Operations

### Luminal: 12 RISC-Style Primitives

```
Unary:  Log2, Exp2, Sin, Sqrt, Recip
Binary: Add, Mul, Mod, LessThan
Reduce: SumReduce, MaxReduce
Other:  Contiguous
```

**Philosophy:** Minimal, composable primitives. Complex ops built from simple ones.

**Example - Softmax:**
```rust
fn softmax(x: GraphTensor) -> GraphTensor {
    let max_val = x.max_reduce();     // MaxReduce
    let shifted = x - max_val;        // Add + Mul
    let exp_x = shifted.exp2();       // Exp2
    let sum_exp = exp_x.sum_reduce(); // SumReduce
    exp_x / sum_exp                   // Mul + Recip
}
```

### Mojo: MLIR Dialects (100+ Operations)

```
Linalg dialect:
  - linalg.matmul
  - linalg.conv_2d
  - linalg.pooling
  - linalg.softmax
  - ...

Tensor dialect:
  - tensor.reshape
  - tensor.extract
  - tensor.insert
  - ...

GPU dialect:
  - gpu.launch
  - gpu.barrier
  - gpu.thread_id
  - ...
```

**Philosophy:** Rich, domain-specific operations. Direct mapping to hardware.

**Example - Softmax (built-in):**
```python
fn softmax(x: Tensor[dtype]) -> Tensor[dtype]:
    return linalg.softmax(x, axis=-1)
```

**Tradeoff:**
- **Luminal**: Small surface area, more optimization opportunities
- **Mojo**: Larger surface area, clearer semantics, easier to optimize specific patterns

## Performance Characteristics

### Luminal

**Strengths:**
- ✅ Zero Python overhead (pure Rust)
- ✅ Can discover novel optimizations (Flash Attention)
- ✅ Aggressive kernel fusion
- ✅ Zero-copy tensor views
- ✅ Minimal abstraction layers

**Weaknesses:**
- ❌ Longer compilation times (search is expensive)
- ❌ Less mature than MLIR
- ❌ Smaller operator library
- ❌ Limited platform support (Metal, CUDA only)

**Benchmarks (from README):**
- Llama 3 8B (Q8): 15-25 tokens/sec on M-series Mac

### Mojo

**Strengths:**
- ✅ Python compatibility (easy adoption)
- ✅ Mature MLIR infrastructure
- ✅ Fast compilation (proven passes)
- ✅ Broad platform support (CPU, GPU, TPU, etc.)
- ✅ Auto-vectorization
- ✅ Compile-time metaprogramming

**Weaknesses:**
- ❌ New language (learning curve)
- ❌ Proprietary (though free to use)
- ❌ Relies on hand-tuned optimizations
- ❌ Still maturing (beta stage)

**Benchmarks (from Modular):**
- Matrix multiply: 35,000x faster than Python
- Llama models: Competitive with llama.cpp

## When to Use Each?

### Use Luminal When:

1. **You're comfortable with Rust** and prefer systems programming
2. **You want automatic optimization discovery** without hand-tuning
3. **You need minimal dependencies** (no Python runtime)
4. **You're targeting Metal/CUDA** specifically
5. **You want to experiment** with cutting-edge compiler techniques
6. **You value simplicity** (12 ops vs 100+)

**Example Use Case:**
```rust
// Building a custom inference engine in Rust
// Want to deploy without Python runtime
// Need maximum performance on Apple Silicon
use luminal_metal::prelude::*;

fn deploy_model() {
    let mut cx = Graph::new();
    // Build model...
    cx.compile(MetalCompiler::<f16>::default(), &mut output);
    // Zero Python, pure Rust binary
}
```

### Use Mojo When:

1. **You're a Python ML practitioner** wanting better performance
2. **You need production-ready tools** with corporate support
3. **You want gradual migration** from Python
4. **You need broad platform support** (CPUs, GPUs, TPUs, edge devices)
5. **You value ecosystem compatibility** (NumPy, etc.)
6. **You need stable, proven optimizations**

**Example Use Case:**
```python
# Migrating existing PyTorch/TensorFlow code
# Need to maintain Python compatibility
# Want significant speedup with minimal rewrites
from max import engine

def inference_pipeline(model_path: String):
    # Load PyTorch model
    # Run with Mojo runtime
    # Get 10-100x speedup
```

## Compilation Approach Comparison

### Luminal: "Search Everything"

```
Problem: What's the best way to compute this?

Approach:
1. Generate ALL equivalent implementations
2. Score each by cost (memory + compute)
3. Choose the best

Analogy: Exhaustive search with pruning

Example:
  x * 2 could be:
    - x + x
    - x << 1 (if power of 2)
    - (x * 4) / 2
    - ...

  Try all, pick cheapest!
```

**Benefit:** Can find unexpected optimizations

**Cost:** Expensive compilation (but only once)

### Mojo: "Progressive Lowering"

```
Problem: How do we optimize this matmul?

Approach:
1. Recognize matmul pattern
2. Apply tiling pass (hand-written)
3. Apply fusion pass (hand-written)
4. Vectorize (LLVM auto-vectorization)

Analogy: Recipe book of known optimizations

Example:
  matmul(A, B):
    1. Tile into blocks (64x64)
    2. Fuse with next op if possible
    3. Vectorize with SIMD
    4. Generate efficient code
```

**Benefit:** Fast, predictable compilation

**Cost:** Limited to known patterns

## Target Audience

### Luminal Target User

```
Profile:
- Rust developer
- Understands compilers
- Building custom ML infrastructure
- Values control and minimalism
- Willing to experiment

Motivation:
"I want maximum performance without Python,
 and I'm willing to work at a lower level
 to get it."
```

### Mojo Target User

```
Profile:
- Python ML engineer/researcher
- Knows PyTorch/TensorFlow
- Wants better performance
- Needs Python compatibility
- Values productivity

Motivation:
"I want my existing Python code to run faster
 without completely rewriting everything."
```

## Future Directions

### Luminal Roadmap (from README)

- Expand search space for Tensor Cores
- CUDA parity with Metal
- ROCm backend
- Distributed training (data/pipeline/tensor parallel)
- Beat PyTorch 2.0 on inference **and** training

**Focus:** More platforms, better search, distributed computing

### Mojo Direction (from Modular)

- Full Python compatibility
- MAX platform integration
- Enterprise features
- Broader hardware support
- Package ecosystem

**Focus:** Ecosystem, stability, production readiness

## Technical Deep Dive: Kernel Generation

### Luminal's Approach

```rust
// User writes:
let c = (a + b) * 2.0;

// Luminal generates:
kernel void fused_add_mul(
    device float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    device float* c [[buffer(2)]]
) {
    int idx = get_global_id(0);
    float tmp = a[idx] + b[idx];  // Fused!
    c[idx] = tmp * 2.0;           // Single kernel
}
```

**How:** Direct code generation from GraphTerm IR

**Control:** Full control over every instruction

### Mojo's Approach

```python
# User writes:
fn add_mul(a: Tensor[f32], b: Tensor[f32]) -> Tensor[f32]:
    let c = (a + b) * 2.0
    return c

# Mojo lowers to MLIR:
%0 = arith.addf %a, %b : tensor<1024xf32>
%1 = arith.mulf %0, %cst : tensor<1024xf32>

# MLIR fusion pass combines them
# LLVM generates vectorized code
```

**How:** Multi-stage lowering through MLIR

**Control:** Rely on MLIR/LLVM optimization passes

## Philosophical Difference

**Luminal Philosophy:**
> "The compiler should discover optimizations automatically
>  through exhaustive search, not rely on human insight."

**Mojo Philosophy:**
> "Build on proven compiler infrastructure (MLIR/LLVM)
>  and make it accessible to Python programmers."

## Summary

| Aspect | Luminal | Mojo |
|--------|---------|------|
| **Best For** | Custom Rust ML infrastructure | Python ML with better performance |
| **Optimization** | Automatic discovery via search | Proven MLIR passes |
| **Learning Curve** | Steep (Rust + compilers) | Gentle (Python-like) |
| **Compilation Time** | Slower (search is expensive) | Faster (direct passes) |
| **Runtime Performance** | Excellent (zero overhead) | Excellent (LLVM optimizations) |
| **Ecosystem** | Small, growing | Large (Python compatible) |
| **Innovation** | Cutting-edge (e-graphs, search) | Mature (MLIR, proven) |

## Conclusion

**Luminal and Mojo solve different problems:**

- **Luminal**: For systems programmers who want compiler-based ML without Python, willing to explore novel compilation techniques
- **Mojo**: For ML practitioners who want Python-like ergonomics with C-level performance

Both are excellent examples of **compiler-first** approaches to ML, but they target different audiences and make different tradeoffs.

**Not competing, complementary:**
- Luminal pushes boundaries of automatic optimization
- Mojo brings proven compiler tech to Python users

The ML compiler landscape benefits from both approaches! 🚀
