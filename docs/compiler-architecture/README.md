# Luminal Compiler Architecture Documentation

## Table of Contents

1. [Overview](./01-overview.md) - High-level architecture and design philosophy
2. [Graph Architecture](./02-graph-architecture.md) - Computation graph design and data flow
3. [Compilation Pipeline](./03-compilation-pipeline.md) - Multi-stage compilation process
4. [Performance Optimizations](./04-performance.md) - How Luminal achieves "speed of light"
5. [Kernel Generation](./05-kernel-generation.md) - GPU code generation and execution
6. [Shape System](./06-shape-system.md) - ShapeTracker and lazy view system

## Quick Summary

Luminal is a **compiler-based deep learning library** that achieves high performance through:

- **Search-based compilation** instead of hand-written heuristics
- **12 RISC-style primitive operations** that compose into complex operations
- **Aggressive kernel fusion** combining multiple ops into single GPU kernels
- **Lazy evaluation** with ahead-of-time compilation
- **Direct Metal/CUDA interaction** with zero Python overhead

```
┌─────────────────────────────────────────────────────────────┐
│                    User Code (Rust)                         │
│  let mut cx = Graph::new();                                 │
│  let a = cx.tensor((3, 1));                                 │
│  let c = a.matmul(b);                                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Computation Graph (DAG)                        │
│  [Load] → [Reshape] → [MatMul] → [Reduce] → [Store]       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│            Compilation Pipeline                             │
│  1. Graph Construction                                      │
│  2. Generic Optimizations (CSE, Dead Code Elimination)     │
│  3. Device-Specific Compilation (Metal/CUDA)               │
│  4. Kernel Fusion & Search-based Optimization              │
│  5. Code Generation                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│          GPU Kernels (Metal/CUDA)                           │
│  Highly optimized, fused operations running on GPU         │
└─────────────────────────────────────────────────────────────┘
```

## Key Innovation: Search-Based Compilation

Instead of using hand-written heuristics, Luminal **searches** through possible graph transformations using e-graphs (equality graphs) to automatically discover optimal implementations:

```
Traditional Approach:         Luminal's Approach:
┌──────────────────┐         ┌──────────────────┐
│ Hand-written     │         │ Define search    │
│ optimization     │         │ space of all     │
│ rules            │         │ valid transforms │
└──────────────────┘         └──────────────────┘
         ↓                            ↓
┌──────────────────┐         ┌──────────────────┐
│ Apply rules      │         │ Automatically    │
│ in fixed order   │         │ search for best  │
└──────────────────┘         │ implementation   │
         ↓                   └──────────────────┘
┌──────────────────┐                  ↓
│ Hope it's good   │         ┌──────────────────┐
└──────────────────┘         │ Discover complex │
                             │ optimizations    │
                             │ (e.g., Flash     │
                             │  Attention)      │
                             └──────────────────┘
```

This allows Luminal to automatically derive complex optimizations like Flash Attention that would otherwise require manual implementation.

## Documentation Goals

This documentation aims to provide a comprehensive understanding of:

1. **How Luminal works as a compiler** - the compilation stages and transformations
2. **Why it's fast** - the specific techniques that enable high performance
3. **The architecture design** - key data structures and algorithms
4. **Code generation** - how GPU kernels are produced and executed

Each document includes ASCII diagrams and references to specific source code locations.
