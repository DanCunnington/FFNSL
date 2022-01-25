import torch
import torch.nn as nn
from network import PlayingCardNet
from dataset import load_data
import numpy as np

torch.set_printoptions(sci_mode=False)

# Instantiate network and load trained weights
net = PlayingCardNet()
network_state_dict = torch.load('saved_model/model.pth')
net.load_state_dict(network_state_dict)
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net.to(dev)
net.eval()

# Load dataset
train_batch_size = 32
test_batch_size = 32
train_loader, test_loader = load_data(train_batch_size=train_batch_size, test_batch_size=test_batch_size)

# Obtain predictions for the test data
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data.to(dev)
        output = net(data)
        softmax_fn = nn.Softmax(dim=1)
        softmax_output = softmax_fn(output)

        # Print first prediction in the batch for now
        print('Softmax output: ', softmax_output[0].tolist())
        print('Softmax prediction confidence: ', max(softmax_output[0].tolist()))
        print('Prediction: ', np.argmax(softmax_output[0].tolist()))
        print('Target: ', target[0])
        print('-----')
        pred = softmax_output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    print('\nTest set. Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                           100. * correct / len(test_loader.dataset)))
