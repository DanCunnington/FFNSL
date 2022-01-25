import torch.nn as nn


class PlayingCardNet(nn.Module):
    def __init__(self):
        super(PlayingCardNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)  # 3 channels; 64 is the output channel size; 3 is the kernel size;
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(17280, 512)
        self.fc2 = nn.Linear(512, 52)

        # self.fc_temp = nn.Linear(6656, 52)

        # self.pool = nn.MaxPool2d(20, 20)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.ReLU(self.conv1(x)))
        x = self.pool(self.ReLU(self.conv2(x)))
        x = self.pool(self.ReLU(self.conv3(x)))
        x = self.pool(self.ReLU(self.conv4(x)))

        x = self.flatten(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc_temp(x)
        return x
