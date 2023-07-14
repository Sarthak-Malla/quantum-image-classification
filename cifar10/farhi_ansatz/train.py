import torch
import torch.nn as nn
import torch.optim as optim

import time

from qimgclassifier.config import config
config.dataset = "cifar10"
config.model_name = "farhi_ansatz"

from qimgclassifier.hybrid_net import QuantumCIFARNet
from qimgclassifier.data_load import get_train_loader, get_test_loader, load_data

config.input_size = 10 # this will change the number of qubits in the quantum circuit

total_start_time = time.time()

X_train, X_test = load_data(config.dataset)
model = QuantumCIFARNet().to(config.device)

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
loss_func = nn.CrossEntropyLoss()

epochs = config.num_epochs
losses = []
train_accuracies = []
test_accuracies = []
print("Training on", config.device)
for epoch in range(epochs):
    epoch_start_time = time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(get_train_loader(X_train)):
        data, target = data.to(config.device), target.to(config.device)
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
                    epoch+1, batch_idx * len(data), len(get_train_loader(X_train).dataset),
                    100. * batch_idx / len(get_train_loader(X_train)), loss.item()))
    
    losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in get_train_loader(X_train):
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        print('Train Accuracy: {:.2f}%'.format(train_accuracy))
        
        correct = 0
        total = 0
        for data, target in get_test_loader(X_test):
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        print('Test Accuracy: {:.2f}%'.format(test_accuracy))
    
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print("Epoch time: {:.2f} seconds".format(time.time() - epoch_start_time))

    # save model for every epoch
    torch.save(model.state_dict(), config.model_path + "/model_" + str(epoch + 1) + ".pth")

torch.save(model.state_dict(), config.model_path + "/model_final.pth")
with open("losses.txt", "w") as f:
    f.write("\n".join(losses))
with open("train_accuracies.txt", "w") as f:
    f.write("\n".join(train_accuracies))
with open("test_accuracies.txt", "w") as f:
    f.write("\n".join(test_accuracies))

print("Total time: {:.2f} seconds".format(time.time() - total_start_time))