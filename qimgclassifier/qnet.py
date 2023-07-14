import logging

import numpy as np

from math import log

import torch
import torch.nn as nn
from torch.autograd import Function

import qiskit
from qiskit import QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

from qiskit import IBMQ, Aer
from qiskit.providers import QiskitBackendNotFoundError
from qiskit.providers.ibmq import IBMQAccountError
from qiskit.providers.ibmq.api.exceptions import AuthenticationLicenseError

from .config import config
from .deprecated import get_subsystems_counts, pauli_single
from .gradients import *


class QuantumCircuit:
    """
    Creates the quantum circuit
    """
    def __init__(self, n_qubits, backend, shots) -> None:
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit = self.qubit_circuit(n_qubits, self.theta)

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
    
    def qubit_circuit(self, n_qubits, theta):
        """
        Creates a quantum circuit
        """
        circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = [i for i in range(n_qubits)]

        circuit.h(all_qubits)
        circuit.barrier()
        circuit.ry(theta, all_qubits)

        circuit.measure_all()

        return circuit


# functions to determine the forward and backward passes in the neural network
class HybridFunction(Function):
    """
    Defines the forward and backward pass
    """
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
        gradients = torch.from_numpy(gradients).to(config.device)
        grad_output = grad_output.to(config.device)

        return gradients * grad_output.float(), None, None

    @staticmethod
    def forward(ctx, inputs, quantum_circuit, shift):
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        expectation_z = []
        for input in inputs:
            expectation_z.append(ctx.quantum_circuit.run(input.tolist()))
        result = torch.tensor(np.array(expectation_z)).to(config.device)

        ctx.save_for_backward(inputs, result)
        return result

"""
The code below is taken from the a Github repository:
https://github.com/bjader/quantum-neural-network
"""
class QuantumNetworkCircuit:
    """
    A quantum neural network. Combines state preparation circuit and variational ansatz to produce quantum neural
    network circuit.
    """

    def __init__(self, input_qubits, input_data=None):
        
        # hyperparameters
        self.layers = 1
        self.sweeps_per_layer = 1
        self.param_counter = 0

        self.input_qubits = input_qubits
        self.input_data = input_data
        self.input_circuit_parameters = None

        self.ansatz_circuit = self._create_ansatz_circuit(input_qubits)

        self.ansatz_circuit_parameters = sorted(list(self.ansatz_circuit.parameters),
                                                key=lambda p: int(''.join(filter(str.isdigit, p.name))))

        self.qr = QuantumRegister(self.ansatz_circuit.num_qubits, name='qr')
        self.cr = ClassicalRegister(len(self.qr), name='cr')
        self.qc = qiskit.QuantumCircuit(self.qr, self.cr)

        self.backend = self.get_backend("qasm_simulator")

        if input_data is not None:
            self.construct_network(input_data)

        self.statevectors = []
        self.gradients = []
        self.transpiled = False

    # create a data handling circuit
    def create_input_circuit(self):
        qr = QuantumRegister(len(self.input_data))
        qc = qiskit.QuantumCircuit(qr)

        for index, _ in enumerate(self.input_data):
            param = Parameter("input{}".format(str(index)))
            qc.rx(param, index)

        return qc

    # create a farhi ansatz circuit
    def _create_ansatz_circuit(self, n_data_qubits):
        qr = QuantumRegister(n_data_qubits, name='qr')
        qc = qiskit.QuantumCircuit(qr, name='Shifted circ')

        for layer_no in range(self.layers):
            for sweep in range(0, self.sweeps_per_layer):
                # self.add_rotations(n_data_qubits)
                # self.add_entangling_gates(n_data_qubits)
                for i in range(n_data_qubits - 1):
                    param = Parameter("ansatz{}".format(str(self.param_counter)))
                    qc.rxx(param, qr[-1], qr[i])
                    self.param_counter += 1
                for i in range(n_data_qubits - 1):
                    param = Parameter("ansatz{}".format(str(self.param_counter)))
                    qc.rzx(param, qr[-1], qr[i])
                    self.param_counter += 1

        # applying null activation function meaning no activation function
        return qc

    def construct_network(self, input_data):
        self.input_data = input_data
        input_circuit = self.create_input_circuit()
        self.input_circuit_parameters = sorted(list(input_circuit.parameters),
                                               key=lambda p: int(''.join(filter(str.isdigit, p.name))))

        self.qc.append(input_circuit, self.qr[:input_circuit.num_qubits])
        self.qc = self.qc.compose(self.ansatz_circuit)

        self.qc.measure(self.qr, self.cr)

    def bind_circuit(self, parameter_values):
        """
        Assigns all parameterized gates to values
        :param parameter_values: List of parameter values for circuit. Input parameters should come before ansatz
        parameters.
        """
        if self.input_circuit_parameters is None:
            raise NotImplementedError(
                "No input data was specified before binding. Please call construct_network() first.")
        combined_parameter_list = self.input_circuit_parameters + self.ansatz_circuit_parameters
        if len(parameter_values) != len(combined_parameter_list):
            raise ValueError('Parameter_values must be of length {}'.format(len(combined_parameter_list)))

        binding_dict = {}
        for i, value in enumerate(parameter_values):
            binding_dict[combined_parameter_list[i]] = value

        bound_qc = self.qc.bind_parameters(binding_dict)
        return bound_qc

    def evaluate_circuit(self, parameter_list, shots=100):
        # if self.transpiled is False:
        #     self.qc = transpile(self.qc, optimization_level=0, basis_gates=['cx', 'u1', 'u2', 'u3'])
        #     self.transpiled = True
        circuit = self.bind_circuit(parameter_list)
        job = execute(circuit, backend=self.backend, shots=shots)
        return job.result()

    @staticmethod
    def get_vector_from_results(results, circuit_id=0):
        """
        Calculates the expectation value of individual qubits for a set of observed bitstrings. Assumes counts
        corresponding to classical  register used for final measurement is final element in job counts array in
        order to exclude classical registers used for activation function measurements (if present).
        :param results: Qiskit results object.
        :param circuit_id: For results of multiple circuits, integer labelling which circuit result to use.
        :return: A vector, where the ith element is the expectation value of the ith qubit
        """

        if results.backend_name == 'statevector_simulator':
            state = results.get_statevector(circuit_id)

            n = int(log(len(state), 2))
            vector = [Statevector(state).expectation_value(pauli_single(n, i, 'Z')).real for i in range(n)]
            return vector

        else:
            counts = results.get_counts(circuit_id)
            all_register_counts = get_subsystems_counts(counts)
            output_register_counts = all_register_counts[-1]
            num_measurements = len(next(iter(output_register_counts)))
            vector = np.zeros(num_measurements)

            for counts, frequency in output_register_counts.items():
                for i in range(num_measurements):
                    if counts[i] == '0':
                        vector[i] += frequency
                    elif counts[i] == '1':
                        vector[i] -= frequency
                    else:
                        raise ValueError("Measurement returned unrecognised value")

            return vector / (sum(output_register_counts.values()))
    
    def get_backend(self, backend_name):
        backend = None
        if backend_name not in ['qasm_simulator', 'statevector_simulator']:
            try:
                IBMQ.load_account()
                oxford_provider = IBMQ.get_provider(hub='ibm-q-oxford')
                backend = oxford_provider.get_backend(backend_name)
            except (IBMQAccountError, AuthenticationLicenseError):
                logging.warning("Unable to connect to IBMQ servers.")
                pass
            except QiskitBackendNotFoundError:
                logging.debug('{} is not a valid online backend, trying local simulators.'.format(backend_name))
                pass
        if backend is None:
            backend = Aer.get_backend(backend_name)
        return backend

class QNetFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, qnn: QuantumNetworkCircuit, shots, save_statevectors):
        weight_vector = torch.flatten(weight).tolist()
        batch_size = input.size()[0]

        if (input > np.pi).any() or (input < 0).any():
            logging.info('Input data to quantum neural network is outside range {0,π}. Consider using a bounded             activation function to prevent wrapping round of states within the Bloch sphere.')

        for i in range(batch_size):
            input_vector = torch.flatten(input[i, :]).tolist()

            if i == 0:
                logging.debug("First input vector of batch to QNN: {}".format(input_vector))

            if qnn.input_data is None:
                qnn.construct_network(input_vector)
            ctx.QNN = qnn

            parameter_list = np.concatenate((np.array(input_vector), weight_vector))

            result = qnn.evaluate_circuit(parameter_list, shots=shots)
            vector = torch.tensor(qnn.get_vector_from_results(result)).unsqueeze(0).float()
            if save_statevectors and result.backend_name == 'statevector_simulator':
                state = result.get_statevector(0)
                qnn.statevectors.append(state)

            if i == 0:
                output = vector
            else:
                output = torch.cat((output, vector), 0)

        ctx.shots = shots

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        ctx.save_for_backward(input, weight)
        ctx.device = device
        output = output.to(device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        device = ctx.device
        weight_vector = torch.flatten(weight).tolist()
        batch_size = input.size()[0]

        for i in range(batch_size):
            input_vector = torch.flatten(input[i, :]).tolist()

            gradient = calculate_gradient_list(ctx.QNN, parameter_list=np.concatenate((input_vector, weight_vector)),
                                               method='parameter shift', shots=ctx.shots)

            ctx.QNN.gradients.append(gradient.tolist())

            single_vector_d_out_d_input = torch.tensor(gradient[:len(input_vector)]).double().to(device)
            single_vector_d_out_d_weight = torch.tensor(gradient[len(input_vector):]).double().to(device)

            if i == 0:
                batched_d_out_d_input = single_vector_d_out_d_input.unsqueeze(0)
                batched_d_out_d_weight = single_vector_d_out_d_weight.unsqueeze(0)
            else:
                batched_d_out_d_input = torch.cat((batched_d_out_d_input, single_vector_d_out_d_input.unsqueeze(0)), 0)
                batched_d_out_d_weight = torch.cat((batched_d_out_d_weight, single_vector_d_out_d_weight.unsqueeze(0)),
                                                   0)
        batched_d_loss_d_input = torch.bmm(batched_d_out_d_input, grad_output.unsqueeze(2).double()).squeeze()
        batched_d_loss_d_weight = torch.bmm(batched_d_out_d_weight, grad_output.unsqueeze(2).double()).squeeze()
        return batched_d_loss_d_input.to(device), batched_d_loss_d_weight.to(device), None, None, None