#!/usr/bin/env python
# coding: utf-8

# # MNIST classification using QNN

# In[110]:


# importing necessary libraries
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

from math import log, pi

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torchvision import datasets, transforms
import torchsummary

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, execute, transpile
from qiskit.quantum_info import Pauli, Statevector
from qiskit import transpile

from qiskit import IBMQ, Aer
from qiskit.providers import QiskitBackendNotFoundError
from qiskit.providers.ibmq import IBMQAccountError
from qiskit.providers.ibmq.api.exceptions import AuthenticationLicenseError
from qiskit.circuit import Parameter


# In[111]:


test_accuracy_list = []
training_accuracy_list = []
batches_list = []
parameters_list = []
dhs_list = []


# Deprecated Functions

# In[112]:


def get_subsystems_counts(complete_system_counts, post_select_index=None, post_select_flag=None):
    """
    Extract all subsystems' counts from the single complete system count dictionary.

    If multiple classical registers are used to measure various parts of a quantum system,
    Each of the measurement dictionary's keys would contain spaces as delimiters to separate
    the various parts being measured. For example, you might have three keys
    '11 010', '01 011' and '11 011', among many other, in the count dictionary of the
    5-qubit complete system, and would like to get the two subsystems' counts
    (one 2-qubit, and the other 3-qubit) in order to get the counts for the 2-qubit
    partial measurement '11' or the 3-qubit partial measurement '011'.

    If the post_select_index and post_select_flag parameter are specified, the counts are
    returned subject to that specific post selection, that is, the counts for all subsystems where
    the subsystem at index post_select_index is equal to post_select_flag.


    Args:
        complete_system_counts (dict): The measurement count dictionary of a complete system
            that contains multiple classical registers for measurements s.t. the dictionary's
            keys have space delimiters.
        post_select_index (int): Optional, the index of the subsystem to apply the post selection
            to.
        post_select_flag (str): Optional, the post selection value to apply to the subsystem
            at index post_select_index.

    Returns:
        list: A list of measurement count dictionaries corresponding to
                each of the subsystems measured.
    """
    mixed_measurements = list(complete_system_counts)
    subsystems_counts = [defaultdict(int) for _ in mixed_measurements[0].split()]
    for mixed_measurement in mixed_measurements:
        count = complete_system_counts[mixed_measurement]
        subsystem_measurements = mixed_measurement.split()
        for k, d_l in zip(subsystem_measurements, subsystems_counts):
            if (post_select_index is None
                    or subsystem_measurements[post_select_index] == post_select_flag):
                d_l[k] += count
    return [dict(d) for d in subsystems_counts]

def pauli_single(num_qubits, index, pauli_label):
        """
        DEPRECATED: Generate single qubit pauli at index with pauli_label with length num_qubits.

        Args:
            num_qubits (int): the length of pauli
            index (int): the qubit index to insert the single qubit
            pauli_label (str): pauli

        Returns:
            Pauli: single qubit pauli
        """
        tmp = Pauli(pauli_label)
        ret = Pauli((np.zeros(num_qubits, dtype=bool), np.zeros(num_qubits, dtype=bool)))
        ret.x[index] = tmp.x[0]
        ret.z[index] = tmp.z[0]
        ret.phase = tmp.phase
        return ret


# In[113]:


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
        self.qc = QuantumCircuit(self.qr, self.cr)

        self.backend = self.get_backend("qasm_simulator")

        if input_data is not None:
            self.construct_network(input_data)

        self.statevectors = []
        self.gradients = []
        self.transpiled = False

    # create a data handling circuit
    def create_input_circuit(self):
        qr = QuantumRegister(len(self.input_data))
        qc = QuantumCircuit(qr)

        for index, _ in enumerate(self.input_data):
            param = Parameter("input{}".format(str(index)))
            qc.rx(param, index)

        return qc

    # create a farhi ansatz circuit
    def _create_ansatz_circuit(self, n_data_qubits):
        qr = QuantumRegister(n_data_qubits, name='qr')
        qc = QuantumCircuit(qr, name='Shifted circ')

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

        logging.info("QNN created with {} trainable parameters.".format(len(self.ansatz_circuit_parameters)))

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


# In[114]:


def calculate_gradient_list(qnn, parameter_list, method='parameter shift', shots=100, eps=None):
    parameter_list = np.array(parameter_list, dtype=float)

    if method == 'parameter shift':
        r = 0.5  # for Farhi ansatz, e0 = -1, e1 = +1, a = 1 => r = 0.5 (Using notation in arXiv:1905.13311)

        qc_plus_list, qc_minus_list = get_parameter_shift_circuits(qnn, parameter_list, r)

        expectation_minus, expectation_plus = evaluate_gradient_jobs(qc_minus_list, qc_plus_list, qnn, shots)

        gradient_list = r * (expectation_plus - expectation_minus)

    else:
        raise ValueError("Invalid gradient method")

    gradient_list = gradient_list.reshape([len(parameter_list), -1])
    return gradient_list


def evaluate_gradient_jobs(qc_minus_list, qc_plus_list, qnn, shots):
    qc_minus_list = [transpile(circ, basis_gates=['cx', 'u1', 'u2', 'u3']) for circ in qc_minus_list]
    qc_plus_list = [transpile(circ, basis_gates=['cx', 'u1', 'u2', 'u3']) for circ in qc_plus_list]
    # job = assemble(qc_minus_list + qc_plus_list, backend=qnn.backend, shots=shots)
    # results = qnn.backend.run(job).result()
    results = qnn.backend.run(qc_minus_list + qc_plus_list, shots=shots).result()
    expectation_plus = []
    expectation_minus = []
    num_params = len(qc_plus_list)
    for i in range(num_params):
        expectation_minus.append(qnn.get_vector_from_results(results, i))
        expectation_plus.append(qnn.get_vector_from_results(results, num_params + i))
        logging.debug("Gradient calculated for {} out of {} parameters".format(i, num_params))
    return np.array(expectation_minus), np.array(expectation_plus)


def get_parameter_shift_circuits(qnn, parameter_list, r):
    qc_plus_list, qc_minus_list = [], []
    for i in range(len(parameter_list)):
        shifted_params_plus = np.copy(parameter_list)
        shifted_params_plus[i] = shifted_params_plus[i] + np.pi / (4 * r)
        shifted_params_minus = np.copy(parameter_list)
        shifted_params_minus[i] = shifted_params_minus[i] - np.pi / (4 * r)

        qc_i_plus = qnn.bind_circuit(shifted_params_plus)
        qc_i_minus = qnn.bind_circuit(shifted_params_minus)
        qc_plus_list.append(qc_i_plus)
        qc_minus_list.append(qc_i_minus)

    return qc_plus_list, qc_minus_list


# In[115]:


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


# In[116]:


class QNet(nn.Module):
    """
    Custom PyTorch module implementing neural network layer consisting on a parameterised quantum circuit. Forward and
    backward passes allow this to be directly integrated into a PyTorch network.
    For a "vector" input encoding, inputs should be restricted to the range [0,π) so that there is no wrapping of input
    states round the bloch sphere and extreme value of the input correspond to states with the smallest overlap. If
    inputs are given outside this range during the forward pass, info level logging will occur.
    """

    def __init__(self, n_qubits, shots=100, save_statevectors=False):
        super(QNet, self).__init__()

        self.qnn = QuantumNetworkCircuit(n_qubits)

        self.shots = shots

        num_weights = len(list(self.qnn.ansatz_circuit_parameters))
        self.quantum_weight = nn.Parameter(torch.Tensor(num_weights))

        self.quantum_weight.data.normal_(std=1. / np.sqrt(n_qubits))

        self.save_statevectors = save_statevectors

        logging.debug("Quantum parameters initialised as {}".format(self.quantum_weight.data))

    def forward(self, input_vector):
        return QNetFunction.apply(input_vector, self.quantum_weight, self.qnn, self.shots, self.save_statevectors)


# In[117]:


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, args['width'])
        self.bn1d = nn.BatchNorm1d(args['width'])
        self.test_network = nn.ModuleList()

        if args['quantum']:
            self.test_network.append(QNet(args['width'], args['shots'], save_statevectors=True))
        else:
            for i in range(args['layers']):
                self.test_network.append(nn.Linear(args['width'], args['width'], bias=True))

        self.fc2 = nn.Linear(args['width'], args['classes'])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        if self.args['batchnorm']:
            x = self.bn1d(x)
        x = np.pi * torch.sigmoid(x)
        for f in self.test_network:
            x = f(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# In[118]:


def train(args, model, device, train_loader, optimizer, epoch, test_loader):
    model.train()
    log_start_time = time.time()
    batches_per_epoch = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        model.test_network[0].qnn.statevectors = []
        correct = 0
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args['log_interval'] == 0:
            seen_images = ((batch_idx + 1) * train_loader.batch_size)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tTime: {:.3f}s'.format(
                epoch, seen_images, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(),
                time.time() - log_start_time))

            # Report the training accuracy
            percentage_accuracy = 100. * correct / len(data)
            training_accuracy_list.append(percentage_accuracy)
            print('Training set accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, len(data), percentage_accuracy))

            batches_list.append((epoch - 1) * batches_per_epoch + batch_idx)
            parameters_list.append(list(model.test_network[0].parameters())[0].detach().cpu().numpy().flatten().tolist())

            if args['q_backend'] == 'statevector_simulator':
                statevectors = np.array(model.test_network[0].qnn.statevectors)

                labels = np.array(target)

                class_0_statevectors = statevectors[labels == 0]
                class_1_statevectors = statevectors[labels != 0]

                rho = np.mean([np.outer(vector, np.conj(vector)) for vector in class_0_statevectors], axis=0)
                sigma = np.mean([np.outer(vector, np.conj(vector)) for vector in class_1_statevectors], axis=0)

                dhs = np.trace(np.linalg.matrix_power((rho - sigma), 2))
                dhs_list.append(dhs.real)

            test(model, device, test_loader)

            output = [batches_list, test_accuracy_list, parameters_list, training_accuracy_list, dhs_list]
            if args['quantum']:
                gradients = model.test_network[0].qnn.gradients
                output.append(gradients)

            log_start_time = time.time()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    percentage_accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracy_list.append(percentage_accuracy)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), percentage_accuracy))
    print(test_accuracy_list)
    print(batches_list)


# In[119]:


args = {
    'batch_size': 64,
    'samples_per_class': 500,
    'classes': 10,
    'test_batch_size': 1000,
    'epochs': 10,
    'lr': 0.001,
    'momentum': 0.5,
    'no_cuda': True,
    'seed': 1,
    'log_interval': 10,
    'q_backend': 'qasm_simulator',
    'quantum': True,
    'batchnorm': False,
    'optimizer': 'adam',
    'save_model': True,
    'plot': True,
    'width': 10,
    'shots': 500,
    'layers': 1,
}

# Create the file where results will be saved
use_cuda = torch.cuda.is_available()

torch.manual_seed(1)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}

mnist_trainset = datasets.MNIST('./datasets', train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# hyperparameters
no_training_samples = args['samples_per_class']
num_classes = args['classes']

train_labels = mnist_trainset.targets.numpy()
train_idx = np.concatenate(
    [np.where(train_labels == digit)[0][0:no_training_samples] for digit in range(num_classes)])
mnist_trainset.targets = train_labels[train_idx]
mnist_trainset.data = mnist_trainset.data[train_idx]

train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args['batch_size'], shuffle=True, **kwargs)

no_test_samples = 500

mnist_testset = datasets.MNIST('./datasets', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
test_labels = mnist_testset.targets.numpy()
test_idx = np.concatenate([np.where(test_labels == digit)[0][0:no_test_samples] for digit in range(num_classes)])
mnist_testset.targets = test_labels[test_idx]
mnist_testset.data = mnist_testset.data[test_idx]

test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=args['test_batch_size'], shuffle=True, **kwargs)

model = Net(args).to(device)
print(torchsummary.summary(model, (1, 28, 28)))

if args['optimizer'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
elif args['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])
elif args['optimizer'] == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
else:
    raise ValueError('Optimiser choice not implemented yet')

for epoch in range(1, args['epochs'] + 1):
    train(args, model, device, train_loader, optimizer, epoch, test_loader)
    # test(model, device, test_loader)

if args['save_model']:
    torch.save(model.state_dict(), "mnist_cnn.pt")

if args['plot']:
    ax = plt.subplot(111)
    plt.plot(batches_list, test_accuracy_list, "--o")
    plt.xlabel('Training batches', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.show()


# In[ ]: