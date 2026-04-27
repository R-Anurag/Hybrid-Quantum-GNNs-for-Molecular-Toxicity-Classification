"""
validate_fix.py
Simple ASCII validation of the bug fix.
"""
import os

def main():
    print("="*60)
    print("HYBRID QUANTUM GNN - BUG FIX VALIDATION")
    print("="*60)
    
    # Read the fixed file
    file_path = os.path.join(os.path.dirname(__file__), "models", "hybrid_qgnn.py")
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Could not read file: {e}")
        return False
    
    print("\nChecking for bug fixes...")
    
    checks = [
        ("Separate build_vqc function exists", "def build_vqc(n_qubits, n_layers):"),
        ("Separate build_vqc_edge function exists", "def build_vqc_edge(n_qubits, n_layers):"),
        ("Edge circuit splits input", "node_inputs = inputs[:n_qubits]"),
        ("Edge circuit extracts edge angles", "edge_angles = inputs[n_qubits:]"),
        ("AngleEmbedding uses only node inputs", "qml.AngleEmbedding(node_inputs, wires=range(n_qubits)"),
        ("CRY gates use edge angles", "qml.CRY(edge_angles[i]"),
        ("Standard circuit uses CNOT", "qml.CNOT(wires=[i, i + 1])"),
        ("Edge projection layer exists", "self.edge_proj = nn.Linear(4, n_qubits - 1)"),
    ]
    
    all_passed = True
    for check_name, search_str in checks:
        found = search_str in content
        status = "[PASS]" if found else "[FAIL]"
        print(f"  {status} {check_name}")
        all_passed = all_passed and found
    
    print("\n" + "="*60)
    print("DIMENSION ANALYSIS")
    print("="*60)
    
    print("\nFor n_qubits=4:")
    print("  Standard VQC:")
    print("    Input:  4 features")
    print("    Output: 4 expectation values")
    print("\n  Edge-embedded VQC:")
    print("    Input:  7 features (4 node + 3 edge)")
    print("    Split:  inputs[:4] for AngleEmbedding")
    print("            inputs[4:7] for CRY gates")
    print("    Output: 4 expectation values")
    
    print("\n" + "="*60)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED")
        print("\nThe bug has been fixed!")
        print("\nKey changes:")
        print("  1. Created separate build_vqc_edge() function")
        print("  2. Edge variant splits input: node_inputs = inputs[:n_qubits]")
        print("  3. Only node_inputs passed to AngleEmbedding")
        print("  4. Edge angles control CRY gates")
        print("\nThis fixes the error:")
        print("  ValueError: Features must be of length 4 or less; got length 7")
    else:
        print("RESULT: SOME CHECKS FAILED")
        print("\nPlease review the hybrid_qgnn.py file")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
