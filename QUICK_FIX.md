# Quick Fix Reference

## The Problem
```
ValueError: Features must be of length 4 or less; got length 7.
```

## The Solution

### Before (BROKEN)
```python
def build_vqc(n_qubits, n_layers, edge_embed=False):
    if edge_embed:
        # Tried to pass 7 features to AngleEmbedding expecting 4
        @qml.qnode(dev)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))  # FAILS: inputs has 7 features
            ...
```

### After (FIXED)
```python
def build_vqc_edge(n_qubits, n_layers):
    @qml.qnode(dev)
    def circuit(inputs, weights):
        node_inputs = inputs[:n_qubits]      # Take first 4
        edge_angles = inputs[n_qubits:]      # Take remaining 3
        qml.AngleEmbedding(node_inputs, wires=range(n_qubits))  # WORKS: 4 features
        ...
        qml.CRY(edge_angles[i], wires=[i, i+1])  # Use edge features here
```

## Why It Works

1. **Input**: 7 features (4 node + 3 edge) passed to circuit
2. **Split**: Inside circuit, split into node (4) and edge (3)
3. **Embed**: Only node features (4) go to AngleEmbedding
4. **Entangle**: Edge features (3) control CRY gates
5. **Output**: 4 expectation values

## Verify the Fix

```bash
cd src
python validate_fix.py
```

Should see: `RESULT: ALL CHECKS PASSED`

## Run Experiments

```bash
cd src
python run_experiments.py
```

All 5 model variants will now work without errors.
