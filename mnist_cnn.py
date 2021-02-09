import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy

import random
import time

# Based on https://nextjournal.com/gkoehler/pytorch-mnist

def getRandom(min, max):
    return random.randint(min, max)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

n_epochs = 10
batch_size_train = 128
batch_size_test = 128
correct_learning_rate = 0.017
cor_lr_change = correct_learning_rate * 0.01
incorrect_learning_rate = 0.003
incor_lr_change = incorrect_learning_rate * 0.01
momentum = 0.5
log_interval = 1000

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader =  torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])),
        batch_size=batch_size_train, shuffle=True)
        
test_loader =  torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])),
        batch_size=batch_size_test, shuffle=True)



network = Net()
optimizer = optim.Adagrad(network.parameters(), lr=correct_learning_rate)

cor_min_rate = 19995
cor_max_rate = 20005
incor_min_rate = 1995
incor_max_rate = 2005

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs * 1)]

test_epoch_acc_output = []
epoch_acc_output = []
cor_lr_output = []
incor_lr_output = []


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
  
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    acc = 100. * correct.item() / len(test_loader.dataset)

    test_epoch_acc_output.append(acc)

start_time = time.time()

# Training
for epoch in range(1, n_epochs + 1):
    train_correct = 0.
    train_total = 0.
    correct_count = 0
    incorrect_count = 0
    correct_rand_ratio = getRandom(cor_min_rate, cor_max_rate)
    incorrect_rand_ratio = getRandom(incor_min_rate, incor_max_rate)

    
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        # Checking for our extra reinforcement
        predicted = output.data.max(1, keepdim=True)[1]
        current_correct = predicted.eq(target.data.view_as(predicted)).sum()

        current_incorrect = batch_size_train - current_correct
        train_correct += current_correct
        train_total += Variable(target).size(0)

        ex_correct_or_incorrect = 0
        if current_correct >= current_incorrect:
            correct_count += current_correct
            incorrect_count += current_incorrect
            ex_correct_or_incorrect = 1
        else:
            correct_count += current_correct
            incorrect_count += current_incorrect
            ex_correct_or_incorrect = 0

        if correct_count >= correct_rand_ratio:
            correct_learning_rate = correct_learning_rate - cor_lr_change
            correct_count = 0
            correct_rand_ratio = getRandom(cor_min_rate, cor_max_rate)
        elif incorrect_count >= incorrect_rand_ratio:
            incorrect_learning_rate = incorrect_learning_rate - incor_lr_change
            incorrect_count = 0
            incorrect_rand_ratio = getRandom(incor_min_rate, incor_max_rate)

        for param_group in optimizer.param_groups:
            if ex_correct_or_incorrect:
                param_group['lr'] = correct_learning_rate
            else:
                param_group['lr'] = incorrect_learning_rate
        optimizer.step()

    cor_lr_output.append(correct_learning_rate)
    incor_lr_output.append(incorrect_learning_rate)

    e_acc = (100. * (train_correct.item() / train_total))
    print("Accuracy of network for epoch %d: %.4f %%" % (epoch, e_acc))
    epoch_acc_output.append(e_acc)

    test()

end_time = time.time()
print("Training time is", (end_time - start_time))
print("Training acc per epoch: ", epoch_acc_output)
print("Testing acc per epoch: ", test_epoch_acc_output)
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





