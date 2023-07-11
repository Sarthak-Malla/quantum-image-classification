from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

algorithm_globals.random_seed = 1234

def create_qnn(n_qubits):
    feature_map = ZZFeatureMap(n_qubits)
    ansatz = RealAmplitudes(n_qubits, reps=1)
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    observables = (
        SparsePauliOp("ZIII"),
        SparsePauliOp("IZII"),
        SparsePauliOp("IIZI"),
        SparsePauliOp("IIIZ"),
    )
    
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        observables=observables
    )
    
    return qnn