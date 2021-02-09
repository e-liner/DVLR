
# Based on https://github.com/yhuag/neural-network-lab/blob/master/Feedforward%20Neural%20Network.ipynb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
import csv
import numpy
import random

def getRandom(min, max):
    return random.randint(min, max)


# Initialize variables
input_size = 784        # Image size = 28x28 = 784
hidden_size = 500       # Hidden nodes
num_classes = 10        # Output classes - 0-9
num_epochs = 10         # Number of times we train on the dataset
batch_size = 100          # Size of input data for a batch
correct_learning_rate = 0.075           # Speed of convergence
cor_lr_change = correct_learning_rate * 0   # Rate change
incorrect_learning_rate = 0.025          # Speed of convergence
incor_lr_change = incorrect_learning_rate * 0 # Rate change

# Download MNIST dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

# Load datasets into batches
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

test_epoch_acc_output = []
epoch_acc_output = []
cor_lr_output = []
incor_lr_output = []

# Feedforward Neural Network Model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create model
model = FFNN(input_size, hidden_size, num_classes)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=correct_learning_rate)

start_time = time.time()
cor_min_rate = 19995
cor_max_rate = 20005
incor_min_rate = 67
incor_max_rate = 72

# Training
for epoch in range(0, num_epochs):
    train_correct = 0.
    train_total = 0.
    dev_correct = 0.
    dev_total = 0.
    correct_count = 0
    incorrect_count = 0
    correct_rand_ratio = getRandom(cor_min_rate, cor_max_rate)
    incorrect_rand_ratio = getRandom(incor_min_rate, incor_max_rate)
    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Checking for our extra reinforcement
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        current_correct = (predicted == labels).sum()
        current_incorrect = batch_size - current_correct
        train_correct += (predicted == labels).sum()

        ex_correct_or_incorrect = 0
        if current_correct >= current_incorrect:
            correct_count += current_correct
            incorrect_count += current_incorrect
            ex_correct_or_incorrect = 1
        else:
            correct_count += current_correct
            incorrect_count += current_incorrect
            ex_correct_or_incorrect = 0

        # Update our learning rate based on our correct and incorrect responses
        if correct_count >= correct_rand_ratio:
            correct_learning_rate = correct_learning_rate - cor_lr_change
            correct_count = 0
            correct_rand_ratio = getRandom(cor_min_rate, cor_max_rate)
        elif incorrect_count >= incorrect_rand_ratio:
            incorrect_learning_rate = incorrect_learning_rate + incor_lr_change
            incorrect_count = 0
            incorrect_rand_ratio = getRandom(incor_min_rate, incor_max_rate)

        for param_group in optimizer.param_groups:
            if ex_correct_or_incorrect:
                param_group['lr'] = correct_learning_rate
            else:
                param_group['lr'] = incorrect_learning_rate

        optimizer.step()

    # Grab test accuracy
    test_correct = 0.
    test_total = 0.
    for test_images, test_labels in test_loader:
        test_images = Variable(test_images.view(-1, 28*28))
        test_outputs = model(test_images)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_total += test_labels.size(0)
        test_correct += (test_predicted == test_labels).sum()

        test_correct = test_correct.item()
            
    test_acc = (100. * test_correct / test_total)
    test_epoch_acc_output.append(test_acc)
    cor_lr_output.append(correct_learning_rate)
    incor_lr_output.append(incorrect_learning_rate)
        
    e_acc = (100. * train_correct.item() / train_total)
    print("Accuracy of network for this batch: %.4f %%" % (e_acc))
    epoch_acc_output.append(e_acc)
    
end_time = time.time()
print("Training acc per epoch: ", epoch_acc_output)
print("Testing acc per epoch: ", test_epoch_acc_output)
print("Training time is", (end_time - start_time))
print("Final Correct Learning rate is", correct_learning_rate)
print("Final Incorrect Learning rate is", incorrect_learning_rate)

test_epoch_print = numpy.asarray(test_epoch_acc_output)
numpy.savetxt("test_out.csv", test_epoch_print, delimiter=",")
epoch_print = numpy.asarray(epoch_acc_output)
numpy.savetxt("train_out.csv", epoch_print, delimiter=",")
cor_lr_print = numpy.asarray(cor_lr_output)
numpy.savetxt("cor_lr_out.csv", cor_lr_print, delimiter=",")
incor_lr_print = numpy.asarray(incor_lr_output)
numpy.savetxt("incor_lr_out.csv", incor_lr_print, delimiter=",")

