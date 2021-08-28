import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.SyncBatchNorm(out_channels)
        self.activation = activation
        if activation:
            self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        if self.activation:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class spatial_branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256, activation=False)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x
