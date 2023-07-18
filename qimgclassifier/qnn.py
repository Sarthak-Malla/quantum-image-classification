import numpy as np

from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN

from .config import config

algorithm_globals.random_seed = config.seed

def create_qnn(n_qubits, add_observables=False):
    feature_map = ZZFeatureMap(n_qubits)
    ansatz = RealAmplitudes(n_qubits, reps=1)
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    observables = None
    if add_observables:
        """
        Create observables for the circuit such that each qubit is measured in the Z basis.
        """
        obs = []
        I = ["I"] * n_qubits

        if (config.num_observables == n_qubits):
            for i in range(config.num_observables):
                I[i] = "Z"
                obs.append("".join(I))
                I[i] = "I"
        else:
            # # have config.num_observables random permutation of I and Z for each qubit, but the same observables should not repeat
            # while (len(obs) < config.num_observables):
            #     I = ["I"] * n_qubits
            #     for j in range(n_qubits):
            #         if (np.random.rand() < 0.5):
            #             I[j] = "Z"
            #     obs.append("".join(I))
            #     obs = list(set(obs))
                
            #     # avoiding the case where all qubits are measured in the Z basis
            #     if ("I"*n_qubits in obs):
            #         obs.remove("I"*n_qubits)

            # This was chosen from a random run for the above commented code
            obs = ['IZZI', 'ZIZZ', 'IZII', 'IIZI', 'ZIZI', 'ZIIZ', 'ZZZI', 'ZIII', 'ZZZZ', 'ZZIZ']

        observables = tuple(SparsePauliOp(o) for o in obs)
        print("Observables: ", obs)
    
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        observables=observables
    )
    
    return qnn