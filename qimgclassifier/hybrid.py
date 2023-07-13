import torch.nn as nn

from .qnet import QuantumCircuit, HybridFunction
from .config import Config

config = Config()

class Hybrid(nn.Module):
    def __init__(self, backend, shots, shift) -> None:
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(config.n_qubits, backend, shots)
        self.shift = shift
    
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)