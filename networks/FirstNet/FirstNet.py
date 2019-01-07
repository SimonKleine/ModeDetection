
import torch
import torch.nn as nn


class FirstNet(nn.Module):

    def __init__(self):
        self.conv = nn.Sequential(nn.Conv1d
                        )
        self.out = nn.Linear

    def forward(self, x):
        x = self.conv
        x = self.out
        return x


CNN = FirstNet()

#Add actual training