# Graph Architecture: Computation Graphs and Data Flow

## Overview

At the heart of Luminal is a **Directed Acyclic Graph (DAG)** representation of computation. Every tensor operation creates nodes and edges in this graph, which is then compiled and executed.

**Key Source File:** `src/graph.rs`

## The Graph Structure

### Core Components

```rust
// src/graph.rs:14-38
pub struct Graph {
    // Tensor data storage: maps (node_id, output_index) → actual data
    pub tensors: FxHashMap<(NodeIndex, u8), Tensor>,

    // Dynamic dimension resolution: 'a' → 128, 'b' → 256, etc.
    pub dyn_map: FxHashMap<char, usize>,

    // The actual computation graph
    pub graph: StorageGraph,

    // Nodes marked to not be deleted during execution
    pub no_delete: FxHashSet<NodeIndex>,

    // Nodes that need to be retrieved after execution
    pub to_retrieve: FxHashMap<NodeIndex, (u8, ShapeTracker)>,

    // Cached topologically sorted execution order
    linearized_graph: Option<Vec<(NodeIndex, Vec<(NodeIndex, u8, ShapeTracker)>)>>,

    // Reference count for automatic memory management
    consumers_map: Option<FxHashMap<(NodeIndex, u8), usize>>,
}
```

### Graph Type Definition

```rust
// src/graph.rs:14
pub type StorageGraph = StableGraph<Box<dyn Operator>, Dependency>;
//                       ^           ^                   ^
//                       |           |                   |
//                       |           |                   Edge weight (dependency type)
//                       |           Node weight (operation)
//                       petgraph's stable directed graph
```

## Visual Example

Let's build a simple computation graph step by step:

### Example 1: Element-wise Operations

```rust
let mut cx = Graph::new();
let a = cx.tensor(3).set([1.0, 2.0, 3.0]);
let b = cx.tensor(3).set([4.0, 5.0, 6.0]);
let c = (a + b) * 2.0;
```

**Graph Structure:**

```
Node 0: Function("Tensor Load")          Node 1: Function("Tensor Load")
        Data: [1.0, 2.0, 3.0]                    Data: [4.0, 5.0, 6.0]
        Shape: (3,)                              Shape: (3,)
              │                                         │
              └────────────┬────────────────────────────┘
                           │
                           ↓
                      Node 2: Add
                      Shape: (3,)
                           │
                           ↓
              Node 3: Constant(2.0)
                      Shape: (1,)
                           │
              ┌────────────┘
              │
              ↓
         Node 4: Mul
         Shape: (3,)
              │
              ↓
         [Output]
```

### Example 2: Matrix Multiplication

```rust
let a = cx.tensor((2, 3));  // 2x3 matrix
let b = cx.tensor((3, 4));  // 3x4 matrix
let c = a.matmul(b);        // 2x4 result
```

**Expanded Graph** (MatMul is a high-level op composed of primitives):

```
[Load A (2,3)]              [Load B (3,4)]
      │                           │
      ├─[Permute(0,1)]           ├─[Permute(1,0)]
      │                           │  Shape: (4,3)
      │                           │
      └───[Reshape(2,3,1)]       └──[Reshape(1,3,4)]
                  │                       │
                  └──────────┬────────────┘
                             │
                             ↓
                         [Mul]
                    Shape: (2,3,4)
                             │
                             ↓
                      [SumReduce(dim=1)]
                       Shape: (2,4)
                             │
                             ↓
                         [Output]
```

**Note:** High-level operations like `matmul` decompose into primitive operations during graph construction.

**Reference:** `src/hl_ops/matmul.rs` - MatMul implementation

## Dependencies: Data Flow

### Two Types of Dependencies

**Location:** `src/graph.rs:39-72`

```rust
pub enum Dependency {
    /// Data dependency: transfers a tensor
    Data {
        input_order: u8,      // Which input slot (0, 1, 2, ...)
        output_order: u8,     // Which output from source node
        shape: ShapeTracker,  // Shape/view information
    },

    /// Schedule dependency: ordering without data transfer
    Schedule,
}
```

### Data Dependencies

These represent actual tensor data flowing between operations:

```
        [Node A]
            │  output_order=0, shape=(3,4)
            │
            ↓
        [Node B]
        input_order=0
```

Multiple inputs example:

```
    [Node A]                [Node B]
        │ output=0              │ output=0
        │ shape=(3,4)           │ shape=(3,4)
        │                       │
        └──────┬───────────────┘
               │
               ↓
           [Add Node]
       input_0    input_1
```

**Code Reference:**
```rust
// src/graph.rs:553-561
pub fn input(mut self, id: NodeIndex, from_output: u8, shape: ShapeTracker) -> Self {
    self.graph_ref.graph.add_edge(
        id,
        self.new_op_id,
        Dependency::Data {
            input_order: self.num_srcs,
            output_order: from_output,
            shape,
        },
    );
    // ...
}
```

### Schedule Dependencies

These enforce execution order without transferring data:

```
    [Kernel A]
        │ (Schedule dependency)
        ↓
    [Kernel B]

    Meaning: B must execute after A, but no data flows from A to B
```

**Use Cases:**
- Enforcing kernel execution order
- Synchronization points
- Memory barriers

**Reference:** `src/graph.rs:350-352`

## Graph Operations

### Adding Operations to the Graph

**Location:** `src/graph.rs:320-365`

```rust
// Add an operation to the graph
let node_id = cx.add_op(Add)
    .input(a_id, 0, a_shape)
    .input(b_id, 0, b_shape)
    .finish();
```

**What happens internally:**

```
Step 1: Create node
┌──────────────────┐
│  graph.add_node  │
│  (Box::new(Add)) │
└──────────────────┘
        │
        ↓ Returns NodeIndex

Step 2: Add input edges
        [Node A] ──Data(order=0, shape)──┐
                                          │
        [Node B] ──Data(order=1, shape)──┤
                                          ↓
                                    [New Add Node]

Step 3: Return node ID
        Returns NodeIndex for use in subsequent operations
```

### Topological Sorting

Before execution, the graph must be sorted to ensure dependencies are satisfied:

**Location:** `src/graph.rs:140-165`

```rust
pub(crate) fn toposort(&mut self) {
    self.linearized_graph = Some(
        petgraph::algo::toposort(&self.graph, None)
            .unwrap()
            .into_iter()
            .map(|node| (node, self.get_sources(node)))
            .collect(),
    );
}
```

**Example:**

```
Original graph:
    A ──→ C
    B ──→ C ──→ D

Topological order: [A, B, C, D]  (one valid ordering)
```

### Getting Sources and Destinations

```rust
// Get all inputs to a node
// src/graph.rs:470-477
pub fn get_sources(&self, node_id: NodeIndex) -> Vec<(NodeIndex, u8, ShapeTracker)> {
    self.graph
        .edges_directed(node_id, Direction::Incoming)
        .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
        .sorted_by_key(|(_, (i, _, _))| *i)
        .map(|(a, (_, c, b))| (a, c, b))
        .collect()
}

// Get all outputs from a node
// src/graph.rs:480-488
pub fn get_dests(&self, node_id: NodeIndex) -> Vec<(NodeIndex, &Box<dyn Operator>)> {
    self.graph
        .edges_directed(node_id, Direction::Outgoing)
        // ...
}
```

## Memory Management

### Reference Counting

Luminal tracks how many times each tensor output is used:

**Location:** `src/graph.rs:151-164`

```rust
// Build consumer count map
self.consumers_map = Some(
    self.graph
        .node_indices()
        .flat_map(|i| {
            self.graph
                .edges_directed(i, Direction::Outgoing)
                .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                .group_by(|(_, (_, i, _))| *i)
                .into_iter()
                .map(|(ind, g)| ((i, ind), g.count()))
                .collect::<Vec<_>>()
        })
        .collect()
);
```

**Visualization:**

```
        [Node A] ──output 0──┬──→ [Node B]
                             │
                             ├──→ [Node C]
                             │
                             └──→ [Node D]

    consumers_map[(A, 0)] = 3  (used by B, C, and D)
```

### Automatic Cleanup

During execution, tensors are freed when their consumer count hits zero:

**Location:** `src/graph.rs:381-404`

```rust
fn get_source_tensors(
    no_delete: &FxHashSet<NodeIndex>,
    tensors: *mut FxHashMap<(NodeIndex, u8), Tensor>,
    src_ids: &[(NodeIndex, u8, ShapeTracker)],
    consumers: &FxHashMap<(NodeIndex, u8), usize>,
) -> Vec<(InputTensor, ShapeTracker)> {
    for (id, ind, sh) in src_ids {
        let id = &(*id, *ind);
        if consumers[id] == 1 && !no_delete.contains(&id.0) {
            // Last consumer - take ownership (move)
            srcs.push((
                InputTensor::Owned(tensors.remove(id).unwrap()),
                *sh,
            ));
        } else {
            // More consumers remaining - borrow
            srcs.push((
                InputTensor::Borrowed(tensors.get(id).unwrap()),
                *sh,
            ));
        }
    }
}
```

**Example:**

```
Execution order: [A, B, C, D]

After executing B:
    consumers[(A, 0)] = 3
    consumers[(A, 0)] -= 1 = 2  (still used by C and D, keep alive)

After executing C:
    consumers[(A, 0)] -= 1 = 1  (still used by D, keep alive)

After executing D:
    consumers[(A, 0)] -= 1 = 0  (no more consumers, DELETE!)
```

## The no_delete Set

Some tensors must be kept alive even if they have no consumers:

**Location:** `src/graph.rs:29`

```rust
pub no_delete: FxHashSet<NodeIndex>
```

**Use cases:**
1. **Output tensors** that the user wants to retrieve
2. **Intermediate results** needed for debugging
3. **Model parameters** that persist across executions

```rust
// Mark a tensor to keep alive
cx.keep_tensors(&output_tensor);

// During cleanup (src/graph.rs:186-188)
pub fn reset(&mut self) {
    self.tensors.retain(|(n, _), _| self.no_delete.contains(n));
}
```

## Graph Visualization

Luminal can display graphs in the browser using Graphviz:

**Location:** `src/compiler_utils.rs:437-454`

```rust
pub fn display(&self) {
    let (g, e, _) = self.debug_graph(false);
    display_graph(&g, &e, &[]);
}

pub fn display_shapes(&self) {
    let (g, e, _) = self.debug_graph(true);
    display_graph(&g, &e, &[]);
}
```

**Example output URL:**
```
https://dreampuf.github.io/GraphvizOnline/#<encoded_graph>
```

## Shape Tracking Through the Graph

Each edge carries a `ShapeTracker` that describes how to interpret the tensor:

```
    [Node A]
    Output shape: (2, 3, 4)
         │
         │ Edge: Data {
         │   shape: ShapeTracker {
         │     dims: [2, 3, 4],
         │     strides: [12, 4, 1],
         │     mask: None,
         │   }
         │ }
         ↓
    [Node B]
    Interprets input as (2, 3, 4) with specific memory layout
```

**Reshapes and permutes** are represented as ShapeTracker transformations, often avoiding actual data movement:

```
Original:     [1, 2, 3, 4, 5, 6]  shape (2, 3)
Permute(1,0): [1, 4, 2, 5, 3, 6]  shape (3, 2) ← may not copy!
                                  Just change strides
```

See [Shape System Documentation](./06-shape-system.md) for details.

## Graph Compilation

Once the graph is built, it goes through compilation:

```rust
// src/graph.rs:132-138
pub fn compile<T: ToIdsMut, C: Compiler>(&mut self, compiler: C, remap: T) -> C::Output {
    let output = compiler.compile(self, remap);
    self.toposort();  // Re-sort after compilation
    self.reset();     // Clear old tensor data
    output
}
```

**Compilation transforms the graph** by:
1. Replacing high-level ops with primitive ops
2. Fusing compatible operations
3. Inserting device-specific kernels
4. Optimizing the graph structure

**The graph structure changes during compilation!**

```
Before compilation:
    [Load A] → [MatMul] → [ReLU] → [Output]

After compilation:
    [Load A] → [MetalMatMul] → [MetalReLU] → [Output]
                    ↓
            (actually a fused kernel)
```

## Execution

Finally, the graph executes:

**Location:** `src/graph.rs:190-224`

```rust
pub fn execute(&mut self) {
    for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
        // Get source tensors (may take ownership if last consumer)
        let srcs = get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

        // Resolve dynamic dimensions
        for (_, st) in srcs.iter_mut() {
            st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
        }

        // Execute the operation
        let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);

        // Store results
        for (i, tensor) in tensors.into_iter().enumerate() {
            self.tensors.insert((*node, i as u8), tensor);
        }

        // Update consumer counts
        for (id, ind, _) in src_ids {
            *consumers.get_mut(&(*id, *ind)).unwrap() -= 1;
        }
    }
}
```

**Execution flow:**

```
┌──────────────────────────────────────────┐
│ 1. Get next node from linearized_graph  │
└──────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│ 2. Collect source tensors                │
│    - Borrow if more consumers remain     │
│    - Take ownership if last consumer     │
└──────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│ 3. Resolve dynamic dimensions            │
└──────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│ 4. Execute operator.process()            │
└──────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│ 5. Store output tensors                  │
└──────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│ 6. Decrement consumer counts             │
│    (triggers automatic cleanup)          │
└──────────────────────────────────────────┘
```

## Summary

The computation graph is the core abstraction in Luminal:

1. **DAG Structure**: Operations as nodes, data flow as edges
2. **Lazy Construction**: Build graph without executing
3. **Shape Tracking**: Each edge carries shape information
4. **Memory Management**: Automatic cleanup via reference counting
5. **Compilation**: Graph is transformed by compiler passes
6. **Execution**: Topologically sorted, minimal memory footprint

**Next:** [Compilation Pipeline](./03-compilation-pipeline.md)
