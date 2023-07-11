import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function

import qiskit

from device import device

class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)

        self._circuit.measure_all()

        self._backend = backend
        self._shots = shots

    def run(self, thetas):
        circuit_ = [self._circuit.bind_parameters({self.theta: theta}) for theta in thetas]
        job = self._backend.run(qiskit.transpile(circuit_, self._backend), shots=self._shots)

        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        probabilities = counts / self._shots

        expectation = np.sum(states * probabilities)

        return np.array([expectation])

# functions to determine the forward and backward passes in the neural network
class HybridFunction(Function):

    @staticmethod
    def backward(ctx, grad_output):
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])

            gradient = np.array([expectation_right]) - np.array([expectation_left])
            gradients.append(gradient)

        gradients = np.array([gradients]).T
        gradients = torch.from_numpy(gradients).to(device)
        grad_output = grad_output.to(device)

        return gradients * grad_output.float(), None, None

    @staticmethod
    def forward(ctx, inputs, quantum_circuit, shift):
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        expectation_z = []
        for input in inputs:
            expectation_z.append(ctx.quantum_circuit.run(input.tolist()))
        result = torch.tensor(np.array(expectation_z)).to(device)

        ctx.save_for_backward(inputs, result)
        return result
