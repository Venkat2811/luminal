# The ML Compiler Landscape: A Comprehensive Survey

## Introduction

The machine learning compiler ecosystem has exploded in recent years. This document provides a comprehensive survey of major open-source ML compiler projects, their architectures, and how they compare to Luminal.

**Survey Methodology:** Analysis based on official documentation, GitHub repositories, research papers, and web searches conducted in November 2025.

## The ML Compiler Ecosystem Map

```
                    ML COMPILER LANDSCAPE

┌──────────────────────────────────────────────────────┐
│           Production Frameworks                       │
│  - Apache TVM (universal deployment)                 │
│  - XLA/OpenXLA (TensorFlow, JAX backend)            │
│  - PyTorch 2.0 (torch.compile)                       │
│  - ONNX Runtime (model deployment)                   │
└──────────────────────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────┐
│         GPU Programming & Kernels                     │
│  - OpenAI Triton (Python → GPU kernels)             │
│  - Halide (image processing DSL)                     │
│  - Mojo/MAX (hand-optimized kernels)                │
└──────────────────────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────┐
│        Infrastructure & Runtime                       │
│  - MLIR (multi-level IR infrastructure)              │
│  - IREE (MLIR-based runtime)                        │
│  - Glow (Meta's accelerator compiler)               │
└──────────────────────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────┐
│         Specialized & Research                        │
│  - Luminal (search-based, Rust)                     │
│  - tinygrad (minimalist, George Hotz)               │
│  - MLC-LLM (LLM deployment)                          │
│  - Enzyme (automatic differentiation)               │
│  - Codon (Python → native)                           │
└──────────────────────────────────────────────────────┘
```

## Detailed Project Analysis

### 1. Apache TVM - Universal ML Compiler

**Website:** [tvm.apache.org](https://tvm.apache.org/)
**GitHub:** [apache/tvm](https://github.com/apache/tvm)
**Paper:** [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (OSDI 2018)](https://arxiv.org/abs/1802.04799)

#### Overview
Apache TVM is an open deep learning compiler stack for CPUs, GPUs, and specialized accelerators. It aims to close the gap between productivity-focused deep learning frameworks and performance- or efficiency-oriented hardware backends.

#### Architecture

```
┌─────────────────────────────────────┐
│   Frontend (PyTorch, TF, ONNX)      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Relax (Graph-level IR)            │
│   - High-level computation graph    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   TensorIR (Tensor-level IR)        │
│   - Loop-based computation          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   AutoTVM / Ansor                   │
│   - Auto-tuning optimization        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Code Generation                   │
│   - CUDA, OpenCL, Metal, etc.       │
└─────────────────────────────────────┘
```

#### Key Features
- **Auto-tuning**: Machine learning-based search for optimal implementations
- **Cross-platform**: CPU, GPU, TPU, VTA, and more
- **Python-first**: Easy to use from Python
- **Modular**: Pluggable backends for new hardware

#### Performance
- Often matches or exceeds hand-tuned libraries
- Auto-tuning can take hours but delivers good results

#### Comparison with Luminal

| Aspect | TVM | Luminal |
|--------|-----|---------|
| **Approach** | ML-based auto-tuning | Search-based (e-graphs) |
| **Language** | Python + C++ | Pure Rust |
| **IR** | Relax + TensorIR | GraphTerm + e-graphs |
| **Optimization** | AutoTVM/Ansor search | Equality saturation |
| **Maturity** | Production (Apache project) | Experimental |
| **Hardware** | Very broad | Metal, CUDA |

**When to use TVM over Luminal:**
- Need broad hardware support (TPUs, VTA, etc.)
- Want mature, production-tested system
- Have time for auto-tuning
- Prefer Python ecosystem

---

### 2. XLA / OpenXLA - Accelerated Linear Algebra

**Website:** [openxla.org](https://openxla.org/xla)
**Architecture Docs:** [XLA Architecture](https://openxla.org/xla/architecture)

#### Overview
XLA (Accelerated Linear Algebra) is a compiler for linear algebra that optimizes TensorFlow and JAX computations. Now part of OpenXLA, it's a collaborative open-source ML compiler ecosystem.

#### Architecture

```
┌─────────────────────────────────────┐
│   ML Framework (TF, JAX, PyTorch)   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   StableHLO (Stable High-Level Ops) │
│   - Portability layer               │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   HLO (High-Level Operations)       │
│   - Target-independent IR           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Target-Independent Optimization   │
│   - Fusion, CSE, DCE                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Backend (GPU, CPU, TPU)           │
│   - Device-specific optimization    │
└─────────────────────────────────────┘
```

#### Key Features
- **Fusion**: Aggressive operation fusion (XLA's killer feature)
- **JIT Compilation**: Just-in-time compilation for dynamic shapes
- **StableHLO**: Version-stable operation set
- **TPU Support**: First-class support for Google TPUs

#### Performance
- JAX + XLA: State-of-art performance for ML research
- TensorFlow + XLA: 1.5-3x speedup on many models

#### Comparison with Luminal

| Aspect | XLA | Luminal |
|--------|-----|---------|
| **Fusion** | Pattern-based | Search-based discovery |
| **IR** | HLO (MLIR-based) | Custom GraphTerm |
| **Backend** | CPU, GPU, TPU | Metal, CUDA |
| **Ecosystem** | TF, JAX, PyTorch | Standalone Rust |
| **Innovation** | Proven fusion passes | Automatic discovery |

**When to use XLA over Luminal:**
- Using JAX or TensorFlow
- Need TPU support
- Want proven, stable compilation
- Require dynamic shape support

---

### 3. OpenAI Triton - GPU Programming Made Easy

**Website:** [Introduction to Triton (OpenAI)](https://openai.com/index/triton/)
**GitHub:** [triton-lang/triton](https://github.com/triton-lang/triton)

#### Overview
Triton is a Python-like language for writing highly efficient GPU kernels. It enables researchers without CUDA expertise to write performance-competitive code, often matching expert-written kernels.

#### Architecture

```
┌─────────────────────────────────────┐
│   Python-like Triton Code           │
│   @triton.jit                       │
│   def kernel(...):                  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Triton IR                         │
│   - Block-level operations          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Automatic Optimization            │
│   - Shared memory management        │
│   - Memory coalescing               │
│   - Synchronization                 │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   PTX / CUDA (NVIDIA)               │
│   or                                │
│   ROCm (AMD)                        │
└─────────────────────────────────────┘
```

#### Key Features
- **Python-like syntax**: No need to learn CUDA
- **Automatic optimization**: Handles memory, barriers, etc.
- **Block-level programming**: Think in tiles, not threads
- **Performance**: Matches expert CUDA code
- **Hardware support**: NVIDIA (Ampere, Hopper, Blackwell), AMD

#### Example
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

#### Integration
Triton is the **default backend for PyTorch 2.0's torch.compile** on GPUs!

#### Comparison with Luminal

| Aspect | Triton | Luminal |
|--------|--------|---------|
| **Level** | Kernel programming | Full framework |
| **User writes** | GPU kernels | High-level ops |
| **Optimization** | Automatic (memory, sync) | Search-based (fusion, tiling) |
| **Output** | PTX/LLVM IR | Metal/CUDA kernels |
| **Use case** | Custom kernels | End-to-end models |

**Synergy:** Luminal could potentially **target Triton** as a backend instead of generating Metal/CUDA directly!

**When to use Triton:**
- Writing custom GPU kernels
- Using PyTorch 2.0 (it's built-in!)
- Need fine control over GPU code
- Want Python-like syntax

---

### 4. PyTorch 2.0 - torch.compile

**Docs:** [torch.compile](https://docs.pytorch.org/docs/stable/torch.compiler.html)
**Tutorial:** [Introduction to torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
**Paper:** [PyTorch 2 paper @ ASPLOS 2024](https://pytorch.org/blog/pytorch-pytorch-2-paper-tutorial/)

#### Overview
PyTorch 2.0 introduced `torch.compile`, a JIT compiler that makes PyTorch code run faster while maintaining the flexibility of eager execution.

#### Architecture

```
┌─────────────────────────────────────┐
│   User PyTorch Code                 │
│   model = MyModel()                 │
│   compiled_model = torch.compile(model) │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   TorchDynamo                       │
│   - Captures Python bytecode        │
│   - Extracts PyTorch operations     │
│   - Creates FX graph                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Compiler Backends                 │
│   - TorchInductor (default)         │
│   - nvFuser, TorchScript, etc.      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   TorchInductor                     │
│   - Generates Triton (GPU)          │
│   - Generates C++ (CPU)             │
└─────────────────────────────────────┘
```

#### TorchDynamo
- **Python-level JIT**: Captures graphs dynamically
- **Frame Evaluation API**: Modifies bytecode at runtime
- **Graph breaks**: Falls back to eager mode when needed

#### TorchInductor
- **Default backend**: Translates FX graphs to optimized code
- **GPU**: Generates Triton kernels
- **CPU**: Generates C++ with vectorization

#### Performance
- **2.27x geometric mean speedup** on inference (A100 GPU, 180+ models)
- **1.41x speedup** on training

#### Comparison with Luminal

| Aspect | PyTorch 2.0 | Luminal |
|--------|-------------|---------|
| **Execution** | Dynamic (JIT) | Static (AOT) |
| **Language** | Python | Rust |
| **Graph capture** | Runtime bytecode | Build-time construction |
| **Backend** | Triton/C++ | Metal/CUDA direct |
| **Ecosystem** | Massive (PyTorch) | Standalone |

**When to use PyTorch 2.0:**
- Already using PyTorch
- Want easy wins (just add `torch.compile`)
- Need dynamic control flow
- Prefer gradual migration

---

### 5. MLIR & IREE - Infrastructure and Runtime

**MLIR:** [mlir.llvm.org](https://mlir.llvm.org/)
**IREE:** [iree.dev](https://iree.dev/)
**GitHub:** [iree-org/iree](https://github.com/iree-org/iree)

#### MLIR (Multi-Level Intermediate Representation)
MLIR is a compiler infrastructure project providing a modular, extensible IR framework for building domain-specific compilers.

#### IREE (Intermediate Representation Execution Environment)
IREE is an MLIR-based end-to-end compiler and runtime that targets datacenter to mobile/edge deployments.

#### Architecture

```
┌─────────────────────────────────────┐
│   ML Framework Input                │
│   (TensorFlow, PyTorch, JAX)        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   MLIR Dialects                     │
│   - linalg (linear algebra)         │
│   - tensor, arith, scf              │
│   - gpu, vulkan                     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Progressive Lowering              │
│   High-level → Mid-level → Low-level│
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   IREE Compiler                     │
│   - Device-specific optimization    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Deployment Artifacts              │
│   - Vulkan, CUDA, CPU, Metal        │
└─────────────────────────────────────┘
```

#### Key Features (IREE)
- **MLIR-native**: Built entirely on MLIR infrastructure
- **Retargetable**: CPU, GPU, accelerators
- **Async execution**: Pipelined execution model
- **Mobile-optimized**: Low overhead for edge devices
- **LF AI Foundation**: Joined as sandbox project (May 2024)

#### Comparison with Luminal

| Aspect | IREE | Luminal |
|--------|------|---------|
| **Foundation** | MLIR (mature) | Custom (minimal) |
| **IR Levels** | Many (progressive lowering) | Two (graph + kernel) |
| **Targets** | Very broad | Metal, CUDA |
| **Approach** | Industry-standard passes | Search-based |
| **Overhead** | Low (designed for mobile) | Minimal (Rust) |

**When to use IREE:**
- Need mobile/edge deployment
- Want MLIR ecosystem benefits
- Require proven infrastructure
- Need broad hardware support

---

### 6. Halide - Image Processing DSL

**Website:** [halide-lang.org](https://halide-lang.org/)
**Paper:** [Halide (PLDI 2013)](https://people.csail.mit.edu/jrk/halide-pldi13.pdf)

#### Overview
Halide is a DSL for image processing that **separates algorithm from schedule**. This separation enables automatic optimization of parallel pipelines.

#### Key Innovation: Algorithm/Schedule Separation

```python
# Algorithm (WHAT to compute)
def blur(input):
    clamped = clamp(input)
    blurred_x = convolve_x(clamped, [1, 4, 6, 4, 1])
    blurred_y = convolve_y(blurred_x, [1, 4, 6, 4, 1])
    return blurred_y

# Schedule (HOW to compute)
blurred_x.vectorize(x, 8).parallel(y)
blurred_y.tile(x, y, 64, 64).parallel(y)
```

#### Performance
- **5x faster than hand-tuned C** (in some cases)
- **Matches expert CUDA code**
- Used in production: Google Pixel, Adobe Photoshop

#### Comparison with Luminal

| Aspect | Halide | Luminal |
|--------|--------|---------|
| **Domain** | Image processing | General ML |
| **Key idea** | Algorithm/schedule separation | Search-based optimization |
| **User control** | Explicit schedules | Automatic |
| **Strength** | Tiling, locality | Fusion, search |

**Inspiration:** Luminal's separation of graph construction and compilation is similar to Halide's philosophy!

**When to use Halide:**
- Image/array processing pipelines
- Need explicit control over scheduling
- Want production-proven DSL

---

### 7. tinygrad - Minimalist Deep Learning

**Website:** [tinygrad.org](https://tinygrad.org/)
**GitHub:** [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)
**Creator:** George Hotz (@geohot)

#### Overview
tinygrad is a **minimalist deep learning framework** that compiles custom kernels for every operation. Pure Python, <10K lines of code.

#### Philosophy
> "Between PyTorch and micrograd. You like PyTorch? You like micrograd? You love tinygrad!"

#### Architecture

```
User Code (pure Python)
    ↓
LazyBuffer (all ops are lazy)
    ↓
Lowering to GPU ops
    ↓
Kernel fusion & optimization
    ↓
Generate GPU code (OpenCL, CUDA, Metal)
    ↓
Execute
```

#### Key Features
- **All lazy**: Every operation builds a graph
- **Custom kernels**: Compiles specific kernel per operation
- **Shape specialization**: Extreme shape-specific optimization
- **Pure Python**: No C++ code
- **Minimal**: <10K lines total

#### Performance Claims
- Competitive with PyTorch for inference
- Focus on simplicity over absolute performance

#### Comparison with Luminal

| Aspect | tinygrad | Luminal |
|--------|----------|---------|
| **Language** | Pure Python | Pure Rust |
| **Size** | <10K lines | ~10K lines |
| **Philosophy** | Minimalism | Compiler research |
| **Optimization** | Kernel fusion | Search-based |
| **Maturity** | Experimental | Experimental |

**Similarity:** Both value **small, understandable codebases**!

**When to use tinygrad:**
- You want to understand everything
- Pure Python is important
- Learning deep learning internals
- Following George Hotz

---

### 8. MLC-LLM - Machine Learning Compilation for LLMs

**Website:** [llm.mlc.ai](https://llm.mlc.ai/)
**GitHub:** [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)

#### Overview
MLC-LLM is a universal LLM deployment engine focused on enabling everyone to run large language models natively on diverse platforms.

#### Key Features
- **Universal deployment**: AMD GPUs, NVIDIA GPUs, Apple Metal, Web (WebGPU), iOS, Android
- **Based on Apache TVM**: Leverages TVM's compilation stack
- **OpenAI-compatible API**: Drop-in replacement
- **Model library**: Pre-compiled models ready to use

#### Architecture

```
LLM Model (Llama, Mistral, etc.)
    ↓
Compile with TVM
    ↓
Optimize for target hardware
    ↓
Generate model library
    ↓
Deploy via MLCEngine (unified runtime)
```

#### Comparison with Luminal

| Aspect | MLC-LLM | Luminal |
|--------|---------|---------|
| **Focus** | LLM deployment | General ML |
| **Based on** | Apache TVM | Custom |
| **Use case** | Run pre-trained LLMs | Train/inference |
| **Target** | Broad (web, mobile) | Metal, CUDA |

**When to use MLC-LLM:**
- Deploying LLMs to diverse platforms
- Need web/mobile LLM inference
- Want OpenAI-compatible API

---

### 9. Enzyme - Automatic Differentiation for LLVM

**Website:** [enzyme.mit.edu](https://enzyme.mit.edu/)
**GitHub:** [EnzymeAD/Enzyme](https://github.com/EnzymeAD/Enzyme)

#### Overview
Enzyme is a plugin that performs automatic differentiation (AD) of LLVM and MLIR code. It works **after optimization**, generating faster derivatives than traditional AD tools.

#### Key Innovation
Traditional AD tools differentiate **before optimization**:
```
Source → AD → Optimize
```

Enzyme differentiates **after optimization**:
```
Source → Optimize → AD (Enzyme)
```

This produces much faster gradients!

#### Multi-Language Support
By working at LLVM level, Enzyme supports:
- C, C++, Swift, Julia, Rust, Fortran, **Python**, etc.

#### Integration with ML
- PyTorch integration available
- TensorFlow integration available
- Enables differentiating foreign code

#### Comparison with Luminal

| Aspect | Enzyme | Luminal |
|--------|--------|---------|
| **Focus** | Automatic differentiation | Full ML framework |
| **Level** | LLVM/MLIR IR | Computation graph |
| **Innovation** | Post-optimization AD | Search-based compilation |
| **Use case** | Differentiate any code | End-to-end ML |

**Synergy:** Luminal could potentially use Enzyme for automatic differentiation!

---

### 10. Codon - High-Performance Python Compiler

**Website:** [MIT News](https://news.mit.edu/2023/codon-python-based-compiler-achieve-orders-magnitude-speedups-0314)
**GitHub:** [exaloop/codon](https://github.com/exaloop/codon)
**Paper:** [Codon: A Compiler for High-Performance Pythonic Applications and DSLs (CC 2023)](https://dl.acm.org/doi/abs/10.1145/3578360.3580275)

#### Overview
Codon is a **Python compiler** that achieves C/C++ performance through ahead-of-time compilation with type inference.

#### Performance
- **10-100x speedup** over vanilla Python
- **On par with C/C++** in many cases
- **Zero runtime overhead**

#### Origins
Started as **Seq**, a DSL for genomics and bioinformatics. Now general-purpose Python compiler.

#### Key Features
- **Type inference**: No manual type annotations needed
- **Native code**: Compiles to machine code
- **NumPy support**: Built-in NumPy compatibility
- **DSL support**: Create domain-specific languages in Python

#### Applications
- Bioinformatics
- Deep learning
- Quantitative finance

#### Comparison with Luminal

| Aspect | Codon | Luminal |
|--------|-------|---------|
| **Goal** | Make Python fast | ML compilation |
| **Approach** | Type inference + compilation | Graph compilation |
| **Language** | Python | Rust |
| **Target domain** | General Python | ML/DL |

**Different goals:** Codon accelerates Python, Luminal is an ML framework

---

### 11. Glow - Meta's ML Compiler

**Website:** [ai.meta.com/tools/glow](https://ai.meta.com/tools/glow/)
**GitHub:** [pytorch/glow](https://github.com/pytorch/glow)
**Paper:** [Glow: Graph Lowering Compiler Techniques for Neural Networks](https://arxiv.org/pdf/1805.00907)

#### Overview
Glow is a machine learning compiler from Meta (Facebook) designed to optimize neural networks for hardware accelerators.

#### Two-Phase IR
1. **High-level IR**: Domain-specific optimizations
2. **Low-level IR**: Memory-related optimizations (instruction scheduling, static allocation)

#### Industry Support
- Cadence, Esperanto, Intel, Marvell, Qualcomm committed support

#### Status
- Originally for PyTorch integration
- Now less actively maintained
- Some ideas incorporated into PyTorch 2.0

#### Comparison with Luminal

| Aspect | Glow | Luminal |
|--------|------|---------|
| **Status** | Less active | Active development |
| **Integration** | PyTorch | Standalone |
| **IR** | Two-phase | Multi-level |

---

### 12. ONNX Runtime - Cross-Platform Inference

**Website:** [onnxruntime.ai](https://onnxruntime.ai/)
**Performance Docs:** [ONNX Runtime Performance](https://onnxruntime.ai/docs/performance/)

#### Overview
ONNX Runtime is a cross-platform, high-performance ML inferencing and training accelerator.

#### Key Features
- **Model format**: Uses ONNX (Open Neural Network Exchange)
- **Graph optimizations**: Constant folding, node fusion, layout optimization
- **Execution providers**: CPU, CUDA, TensorRT, DirectML, CoreML, etc.
- **Transformer optimization**: Special optimizations for transformers

#### Optimization Levels
- **O1**: Basic optimizations
- **O2**: Extended optimizations + transformer fusions

#### Comparison with Luminal

| Aspect | ONNX Runtime | Luminal |
|--------|--------------|---------|
| **Input** | ONNX models | Native API |
| **Focus** | Inference | Training + Inference |
| **Optimization** | Pattern-based | Search-based |
| **Deployment** | Production-ready | Experimental |

---

## Comparison Matrix: All Projects

| Project | Language | Approach | IR | Auto-tune | Production | Unique Feature |
|---------|----------|----------|-----|-----------|------------|----------------|
| **Luminal** | Rust | Search (e-graphs) | GraphTerm | ✓ (search) | ✗ | Automatic discovery |
| **TVM** | Python/C++ | ML auto-tuning | Relax+TensorIR | ✓ (AutoTVM) | ✓ | Universal hardware |
| **XLA** | C++ | Fusion passes | HLO/StableHLO | ✗ | ✓ | JAX/TF backend |
| **Triton** | Python | Block-level | Triton IR | ✗ | ✓ | Easy GPU kernels |
| **PyTorch 2.0** | Python | Dynamic JIT | FX Graph | ✗ | ✓ | Drop-in speedup |
| **IREE** | C++ | MLIR-based | MLIR dialects | ✗ | ✓ | Mobile-optimized |
| **Halide** | C++ | Schedule DSL | Halide IR | ✓ (auto-schedule) | ✓ | Algorithm/schedule |
| **tinygrad** | Python | Lazy + fusion | LazyBuffer | ✗ | ✗ | Minimalist |
| **MLC-LLM** | Python | TVM-based | TVM IR | ✓ | ✓ | LLM deployment |
| **Enzyme** | C++ | LLVM plugin | LLVM/MLIR | ✗ | ✓ | Post-opt AD |
| **Codon** | Python | Type inference | LLVM IR | ✗ | ✓ | Python → native |
| **Mojo** | Mojo | Hand-optimized | MLIR | ✗ | ✓ | Python-like perf |

## Key Trends in ML Compilers

### 1. MLIR Convergence
Many projects converging on MLIR as IR:
- IREE (fully MLIR-based)
- XLA (migrating to MLIR)
- Mojo (built on MLIR)
- PyTorch (exploring MLIR)

### 2. Search-Based Optimization
Growing interest in automated search:
- TVM (AutoTVM, Ansor)
- Halide (auto-scheduler)
- **Luminal (e-graphs)**

### 3. Python for GPU Kernels
Making GPU programming accessible:
- Triton (Python-like syntax)
- Mojo (Python superset)
- tinygrad (pure Python)

### 4. Specialization for LLMs
LLM-specific optimizations:
- MLC-LLM (dedicated LLM compiler)
- ONNX Runtime (transformer optimizations)
- PyTorch 2.0 (attention fusion)

### 5. Post-Optimization Techniques
Optimizing after traditional passes:
- Enzyme (post-opt AD)
- Luminal (search after expansion)

## Where Luminal Fits

### Luminal's Unique Position

```
         Maturity
            ↑
Production  │  TVM  XLA  PyTorch 2.0  ONNX  Mojo
            │  IREE Halide
            │
Research    │  Luminal  tinygrad
            │
            └──────────────────────────→
              Manual        Automatic
              Optimization  Discovery
```

**Luminal occupies a unique niche:**
- **Research-oriented** like tinygrad
- **Automatic optimization** via search
- **Systems language** (Rust) unlike Python competitors
- **Focus on compiler innovation** (e-graphs, search)

### Luminal's Advantages

1. **Automatic Discovery**: Only Luminal uses e-graphs for optimization discovery
2. **Minimal Codebase**: ~10K lines, easy to understand and modify
3. **Zero Python**: Pure Rust, no interpreter overhead
4. **Research-Friendly**: Perfect for exploring new compiler techniques

### Areas for Luminal Growth

Based on ecosystem analysis:

1. **MLIR Integration**: Consider MLIR as IR (industry standard)
2. **Triton Backend**: Generate Triton instead of raw Metal/CUDA
3. **LLM Focus**: Specialized LLM optimizations (like MLC-LLM)
4. **Auto-Differentiation**: Integration with Enzyme
5. **Broader Hardware**: TPU, WebGPU support

## Recommendations for Developers

### Choose Luminal if:
- ✓ Researching compiler techniques
- ✓ Want pure Rust, no Python
- ✓ Interested in e-graphs and search
- ✓ Building custom ML infrastructure
- ✓ Value small, understandable codebase

### Choose TVM if:
- ✓ Need broad hardware support
- ✓ Want production-ready system
- ✓ Have time for auto-tuning
- ✓ Deploying to exotic accelerators

### Choose PyTorch 2.0 if:
- ✓ Already using PyTorch
- ✓ Want easy speedups (torch.compile)
- ✓ Need dynamic control flow
- ✓ Prefer gradual migration

### Choose Triton if:
- ✓ Writing custom GPU kernels
- ✓ Want Python-like syntax
- ✓ Need expert-level performance
- ✓ Using PyTorch 2.0

### Choose XLA/JAX if:
- ✓ Doing ML research
- ✓ Need TPU support
- ✓ Want functional transformations
- ✓ Value composability

### Choose tinygrad if:
- ✓ Want to understand everything
- ✓ Learning internals
- ✓ Value minimalism
- ✓ Pure Python is important

## Conclusion

The ML compiler landscape is **rich and diverse**, with projects tackling different aspects:

- **Infrastructure**: MLIR, IREE
- **Production**: TVM, XLA, PyTorch 2.0, ONNX Runtime
- **GPU Programming**: Triton, Mojo
- **Research**: Luminal, tinygrad
- **Specialized**: MLC-LLM, Halide, Enzyme, Codon

**Luminal's contribution:** Exploring **search-based optimization with e-graphs** in a **minimal, pure-Rust** framework. This is a unique and valuable position in the ecosystem!

## Sources

- [Apache TVM](https://tvm.apache.org/)
- [OpenXLA](https://openxla.org/xla)
- [OpenAI Triton](https://openai.com/index/triton/)
- [PyTorch 2.0 torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)
- [IREE](https://iree.dev/)
- [Halide](https://halide-lang.org/)
- [tinygrad](https://tinygrad.org/)
- [MLC-LLM](https://llm.mlc.ai/)
- [Enzyme AD](https://enzyme.mit.edu/)
- [Codon](https://github.com/exaloop/codon)
- [Glow](https://ai.meta.com/tools/glow/)
- [ONNX Runtime](https://onnxruntime.ai/)
