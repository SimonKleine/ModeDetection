import torch.nn as nn


class FirstNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d)
        self.out = nn.Linear(23, 42)

    def forward(self, x):
        x = self.conv(x)
        x = self.out(x)

        return x


CNN = FirstNet()
# Add actual training
