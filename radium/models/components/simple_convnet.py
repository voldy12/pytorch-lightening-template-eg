from torch import nn
import torch.nn.functional as F
import torch


class SimpleConvNet(nn.Module):
    def __init__(
        self,
        conv_1_n_kernerls: int = 6,
        conv_2_n_kernerls: int = 16,
        lin1_size: int = 64,
        lin2_size: int = 64,
        output_size: int = 10,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv_1_n_kernerls, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv_1_n_kernerls, conv_2_n_kernerls, 5)
        self.fc1 = nn.Linear(conv_2_n_kernerls * 5 * 5, lin1_size)
        self.fc2 = nn.Linear(lin1_size, lin2_size)
        self.fc3 = nn.Linear(lin2_size, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
