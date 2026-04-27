# Implementation Note: Backpropagation for Quantum Circuits

**Date**: 2025-01-XX  
**Decision**: Use backprop differentiation method  
**Status**: IMPLEMENTED

---

## Decision Summary

**Quantum circuits use `diff_method="backprop"` for gradient computation.**

This is a **simulator-based approach** for proof-of-concept research, not representative of real quantum hardware behavior.

---

## Rationale

### Project Goal
This project investigates whether **quantum circuit architectures** improve molecular toxicity prediction compared to classical GNNs.

**Focus**: Architectural comparison, not quantum hardware deployment.

### Why Backprop?

1. **Practical Training Time**
   - Backprop: 3-4 hours for full experiments
   - Parameter-shift: 12+ hours for full experiments
   - **Trade-off**: Speed vs hardware realism

2. **Simulator Environment**
   - Using PennyLane's `default.qubit` (CPU simulator)
   - NOT running on real quantum hardware (IBM, Google, etc.)
   - Backprop is standard for simulator-based research

3. **Research Question**
   - "Does quantum circuit architecture help?" ✓
   - NOT "Can this run on quantum hardware?" ✗

4. **PennyLane Documentation**
   - Backprop: "NOT hardware compatible; only supported on simulators"
   - Parameter-shift: "Hardware compatible"
   - **We're on a simulator** → backprop is appropriate

---

## Limitations & Honesty

### What This Means

✅ **Valid for**:
- Comparing quantum-inspired architectures vs classical
- Proof-of-concept research
- Exploring if quantum circuits add representational power
- Academic project demonstrating hybrid approaches

❌ **NOT valid for**:
- Claiming quantum hardware advantage
- Measuring real quantum computational cost
- Hardware deployment readiness
- True quantum gradient behavior

### Honest Reporting

**In paper/report, must state**:
> "Quantum circuits simulated on classical hardware using backpropagation for gradient computation. This approach enables efficient exploration of quantum-inspired architectures but does not represent quantum hardware performance. Future work includes deployment on real quantum devices with parameter-shift gradients."

---

## Technical Details

### Implementation

**quantum_only.py**:
```python
@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    # ... quantum operations ...
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Forward pass (batched)
q_out = self.vqc(q_in)  # (B, n_qubits)
```

**hybrid_qgnn.py**:
```python
@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    # ... quantum operations ...
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Forward pass (batched for standard, loop for edge variant)
if self.edge_embed:
    q_out = torch.stack([self.vqc(q_in[i], ep[i]) for i in range(B)])
else:
    q_out = self.vqc(q_in)  # (B, n_qubits)
```

### Performance

| Model | Backprop (batched) | Parameter-shift (sequential) |
|-------|-------------------|------------------------------|
| Classical GCN | 2-5s/epoch | 2-5s/epoch |
| Quantum-only 4q | 10-15s/epoch | 50-80s/epoch |
| Hybrid 4q | 15-20s/epoch | 80-120s/epoch |
| Hybrid 8q | 30-40s/epoch | 180-240s/epoch |
| Hybrid 4q + Edge | 20-25s/epoch | 100-150s/epoch |

**Total experiment time**: 3-4 hours (vs 12+ hours with parameter-shift)

---

## Comparison to Literature

### Common Practice

Many quantum ML papers on simulators use backprop:
- Faster iteration during research
- Focus on architectural exploration
- Hardware deployment as "future work"

### When Parameter-Shift is Required

- Deploying on real quantum hardware (IBM Quantum, Rigetti, IonQ)
- Measuring true quantum computational cost
- Hardware-in-the-loop experiments
- Claims about quantum advantage

**We are NOT doing these** → backprop is appropriate.

---

## Future Work

### Path to Quantum Hardware

1. **Current**: Backprop on simulator (proof of concept)
2. **Next**: Parameter-shift on simulator (validate hardware compatibility)
3. **Final**: Deploy on real quantum hardware (IBM Quantum, AWS Braket)

### Recommended Next Steps

- Test with parameter-shift on small subset (validate compatibility)
- Benchmark on quantum hardware simulators (Qiskit Aer with noise models)
- Apply for quantum hardware access (IBM Quantum, AWS Braket)
- Compare simulator vs hardware results

---

## Conclusion

**Decision**: Use backprop for this project.

**Justification**: 
- Appropriate for simulator-based architectural research
- Enables practical training times
- Honest about limitations
- Standard practice in quantum ML research

**Requirement**: Clear documentation of simulator use and backprop method in all reporting.

---

## References

- PennyLane Documentation: https://docs.pennylane.ai/en/stable/introduction/interfaces.html
- "Backprop: NOT hardware compatible; only supported on simulators"
- Quantum Backpropagation Demo: https://pennylane.ai/qml/demos/tutorial_backprop

---

**Approved**: Yes  
**Implemented**: Yes  
**Documented**: Yes  
**Honest**: Yes
