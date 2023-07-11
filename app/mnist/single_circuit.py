import numpy as np
import matplotlib.pyplot as plt

import torch
# from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

import qiskit
from qiskit.visualization import *

from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

algorithm_globals.random_seed = 1234

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Concentrating on the first 100 samples
n_samples = 400
batch_size = 4

X_train = datasets.MNIST(root=dir_path+'/data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

X_train.data = X_train.data[:n_samples]
X_train.targets = X_train.targets[:n_samples]

# filter for 4 classes
filtered = torch.logical_or(X_train.targets == 0, X_train.targets == 1)
filtered = torch.logical_or(filtered, X_train.targets == 2)
filtered = torch.logical_or(filtered, X_train.targets == 3)
X_train.data = X_train.data[filtered]
X_train.targets = X_train.targets[filtered]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=8)

n_samples = 150

X_test = datasets.MNIST(root=dir_path+'/data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

X_test.data = X_test.data[:n_samples]
X_test.targets = X_test.targets[:n_samples]

# filter for 4 classes
filtered = torch.logical_or(X_test.targets == 0, X_test.targets == 1)
filtered = torch.logical_or(filtered, X_test.targets == 2)
filtered = torch.logical_or(filtered, X_test.targets == 3)
X_test.data = X_test.data[filtered]
X_test.targets = X_test.targets[filtered]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True, num_workers=8)

# define and create a QNN
def create_qnn(n_qubits):
    feature_map = ZZFeatureMap(n_qubits)
    ansatz = RealAmplitudes(n_qubits, reps=1)
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    observables = (
        SparsePauliOp("ZZZZ"),
        SparsePauliOp("IIZZ"),
        SparsePauliOp("IZIZ"),
        SparsePauliOp("ZZII"),
    )
    
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        observables=observables
    )
    
    return qnn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 4)
        # self.hybrid = Hybrid(qiskit.Aer.get_backend('qasm_simulator'), 100, np.pi / 2)
        # self.hybrid = [TorchConnector(create_qnn()) for i in range(10)]
        self.hybrid = TorchConnector(create_qnn(4)).to(device)
        self.fc3 = nn.Linear(4, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        # x = x.view(-1, 256)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.chunk(x, 10, dim=1)
        x = self.hybrid(x)
        # x = tuple([hy(x_) for hy, x_ in zip(self.hybrid, x)])
        x = self.fc3(x)
        # return torch.cat(x, -1)
        return x

model = Net().to(device)

# load model
# model.load_state_dict(torch.load(dir_path+'/models/mnist/mnist_single_circuit_4.pth'))

optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss_func = nn.NLLLoss()
loss_func = nn.CrossEntropyLoss()

epochs = 20
print("Training on", device)
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)

        # Calculating loss
        loss = loss_func(output, target)

        # Backward pass
        loss.backward()

        # Optimize the weights
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.shape[0]

        print('Test accuracy: {:.0f}%'.format(100. * correct / total))

    # save model for every epoch
    torch.save(model.state_dict(), dir_path+'/models/mnist/v1_4_qubit_4_obs/mnist_single_circuit_{}_v1_4_qubit_without_fc_run2.pth'.format(epoch+1))

"""
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.shape[0]

    print('Test accuracy: {:.0f}%'.format(100. * correct / total))
"""

