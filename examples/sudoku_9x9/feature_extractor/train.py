from network import MNISTNet
from dataset import load_data
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure save dir
save_dir = 'saved_model'
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Configure hyper-parameters
learning_rate = 1
n_epochs = 20
log_interval = 10

# Configure random seeds
random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Instantiate network and optimiser
network = MNISTNet()
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
network.to(dev)
optimizer = optim.Adadelta(network.parameters(), lr=learning_rate)

# Load data
train_loader, test_loader = load_data()

# Initialise lists for tracking training performance
train_losses = []
train_counter = []
test_losses = []
test_accuracies = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


# Method to train network
def train(e, log_file):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data.to(dev)
        output = network(data)
        lf = nn.CrossEntropyLoss()
        loss = lf(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            tc = (batch_idx*len(data)) + ((e-1)*len(train_loader.dataset))
            train_counter.append(tc)
            torch.save(network.state_dict(), save_dir+'/model.pth')
            torch.save(optimizer.state_dict(), save_dir+'/optimizer.pth')
            with open(log_file, 'a') as logger:
                logger.write('{0},{1}\n'.format(tc, loss.item()))


# Method to evaluate network
def test(log_file, counter_idx):
    network.eval()
    batch_test_losses = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data.to(dev)
            output = network(data)
            lf = nn.CrossEntropyLoss()
            batch_test_losses.append(lf(output, target).item())
            softmax_fn = nn.Softmax(dim=1)
            softmax_output = softmax_fn(output)
            pred = softmax_output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss = sum(batch_test_losses) / len(batch_test_losses)
    test_losses.append(test_loss)
    test_accuracies.append(correct / len(test_loader.dataset))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    with open(log_file, 'a') as logger:
        logger.write('{0},{1},{2},{3}\n'.format(test_counter[counter_idx],
                                                test_loss,
                                                correct,
                                                correct / len(test_loader.dataset)))


# Setup logging files
test_log_path = save_dir+'/test_log.txt'
train_log_path = save_dir+'/train_log.txt'
with open(test_log_path, 'w') as test_log:
    test_log.write('num_training_examples_observed,test_loss,num_correct,accuracy\n')

with open(train_log_path, 'w') as train_log:
    train_log.write('num_training_examples_observed,loss\n')

# Test model without training
test(test_log_path, 0)

# Run main training loop
for epoch in range(1, n_epochs + 1):
    train(epoch, train_log_path)
    test(test_log_path, epoch)

# Generate Training Graph
fig, ax1 = plt.subplots()

color = 'red'
ax1.set_xlabel('Number of observed training examples')
ax1.set_ylabel('Cross-Entropy Loss', color=color)
ax1.plot(train_counter, train_losses, color=color, label='Train Loss', zorder=1)
ax1.scatter(test_counter, test_losses, color='orange', label='Test Loss', zorder=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'blue'
ax2.set_ylabel('Test Accuracy', color=color)
ax2.plot(test_counter, test_accuracies, color=color, label='Test Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
lgd = fig.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
fig.savefig(save_dir+'/training_graph.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

# Print logs to console
print('Training finished.')
print('Train counter: ')
print(train_counter)
print('Train Losses')
print(train_losses)
print('Test Counter')
print(test_counter)
print('Test Losses')
print(test_losses)
print('Test Accuracies')
print(test_accuracies)
