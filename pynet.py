import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import os
import time

torch.set_num_threads(1)

train_set = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=1)
test_set = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
testloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=1)

class ProjNet(nn.Module):
    def __init__(self, in_channels=3):
        super(ProjNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 5, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.full = nn.Linear(576, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 576)
        x = self.full(x)
        return x


if __name__== "__main__":
    net = ProjNet(in_channels=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
