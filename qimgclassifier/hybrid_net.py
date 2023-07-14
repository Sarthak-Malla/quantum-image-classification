import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit_machine_learning.connectors import TorchConnector

from .hybrid import Hybrid
from .qnn import create_qnn
from .qnet import QNetFunction, QuantumNetworkCircuit

from .config import config

torch.manual_seed(config.seed)

class HybridNet(nn.Module):
    """
    A hybrid quantum - classical convolutional neural network model.

    Use this class will facilitate custome Hybrid Architectures with a custom Quantum Neural Network forward and backward pass.
    Utilizes the "Hybrid" class that leverages a custom Function class to allow for custom backward passes, essentially allowing to integrate a quantum circuit into a neural network.
    """
    def __init__(self, torch_connector=False):
        super(HybridNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, config.input_size)
        if (torch_connector):
            self.fc2 = nn.Linear(64, config.input_size * config.n_qubits)
            self.hybrid = [TorchConnector(create_qnn(config.n_qubits)).to(config.device) for _ in range(10)]
        else:
            self.hybrid = [Hybrid(qiskit.Aer.get_backend(config.backend), 100, np.pi / 2) for _ in range(10)]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.chunk(x, 10, dim=1)
        x = tuple([hy(x_) for hy, x_ in zip(self.hybrid, x)])
        return torch.cat(x, -1)

class TorchNet(nn.Module):
    """
    A hybrid quantum - classical convolutional neural network model.

    This network uses the TorchConnector class to integrate a quantum neural network into a classical neural network.
    """
    def __init__(self):
        super(TorchNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, config.input_size)
        self.hybrid = TorchConnector(create_qnn(config.n_qubits)).to(config.device)
        self.fc3 = nn.Linear(10, config.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        x = self.fc3(x)
        return x

class HybridCIFARNet(nn.Module):
    # implementing a VGG16 architecture 
    def __init__(self, torch_connector=False):
        super(HybridCIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(25088, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, config.input_size)

        if (torch_connector):
            self.fc16 = nn.Linear(64, config.input_size * config.n_qubits)
            self.hybrid = [TorchConnector(create_qnn(config.n_qubits)).to(config.device) for _ in range(10)]
        else:
            self.hybrid = [Hybrid(qiskit.Aer.get_backend(config.backend), 100, np.pi / 2) for _ in range(10)]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)

        x = torch.chunk(x, 10, dim=1)
        x = tuple([hy(x_) for hy, x_ in zip(self.hybrid, x)])

        return torch.cat(x, -1)
    
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

    def forward(self, input_vector):
        return QNetFunction.apply(input_vector, self.quantum_weight, self.qnn, self.shots, self.save_statevectors)

class QuantumNet(nn.Module):
    """
    Network Architecture using the QNet module
    """
    def __init__(self):
        super(QuantumNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, config.input_size)
        self.bn1d = nn.BatchNorm1d(config.input_size)
        self.test_network = nn.ModuleList()

        self.test_network.append(QNet(config.input_size, config.shots, save_statevectors=True))
        self.fc3 = nn.Linear(config.input_size, config.num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if config.batch_norm:
            x = self.bn1d(x)
        x = np.pi * torch.sigmoid(x)
        for f in self.test_network:
            x = f(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output