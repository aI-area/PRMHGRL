import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(1173, 128)
        self.fc2 = nn.Linear(128, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out
