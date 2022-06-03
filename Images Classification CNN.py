# import packages (Makes life easier)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Image classification modelled by a fully implemented CNN

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameter
epochs = 50
learning_rate = 0.001
batch_size = 4

# The images in the dataset are PILImage images of range(0, 1)
# We load the data and then, transform the images to tensors of normalised range(-1, 1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Image classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# We create a customised model
class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 199)
        self.fc2 = nn.Linear(199, 99)
        self.fc3 = nn.Linear(99, 10)
        
    
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
# Define the model, loss function and optimizer
model = CNNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images.to(device)
        labels.to(device)
        
        # Forward pass
        output = model(images)
        loss = criterion(output, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # We'll print some information as the model is training
        if (i+1) % 2500 == 0:
            print(f'Epoch: {epoch+1}/{epochs}, Step: {i+1}/{n_total_steps} Loss: {loss:.4f}')
            
print('Finished training')

# Model accuracy check
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0.0 for i in range(10)]
    n_class_samples = [0.0 for i in range(10)]
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted==labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label==pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    
    acc = 100.00 * (n_correct/n_samples)
    print(f"Model's accuracy: {acc}%")
    
    for i in range(10):
        acc = 100.00 * n_class_correct[i]/n_class_samples[i]
        print(f"Accuracy of {classes[i]}: {acc}%")
        
# ifunanyaScript