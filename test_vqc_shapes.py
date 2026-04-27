import torch
import pennylane as qml

n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    print(f"inputs shape: {inputs.shape}")
    print(f"inputs: {inputs}")
    
    node_inputs = inputs[:n_qubits]
    edge_angles = inputs[n_qubits:]
    
    print(f"node_inputs shape: {node_inputs.shape}")
    print(f"edge_angles shape: {edge_angles.shape}")
    
    for i in range(n_qubits):
        qml.RY(node_inputs[i], wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 2)}
vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

# Test with batch size 1
test_input = torch.randn(1, 7)
print(f"\nTest input shape: {test_input.shape}")
output = vqc(test_input)
print(f"Output shape: {output.shape}")
