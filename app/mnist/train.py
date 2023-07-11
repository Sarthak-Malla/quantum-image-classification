import torch
import torch.optim as optim
import torch.nn as nn

from device import device
from qnet import Net

from data_load import get_train_loader, get_test_loader, dir_path

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
    for batch_idx, (data, target) in enumerate(get_train_loader()):
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
                    epoch+1, batch_idx * len(data), len(get_train_loader().dataset),
                    100. * batch_idx / len(get_train_loader()), loss.item()))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in get_test_loader():
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.shape[0]

        print('Test accuracy: {:.0f}%'.format(100. * correct / total))

    # save model for every epoch
    torch.save(model.state_dict(), dir_path+'/models/mnist/v1_4_qubit_4_obs/mnist_single_circuit_{}_v1_4_qubit_without_fc_run2.pth'.format(epoch+1))