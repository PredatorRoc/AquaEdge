import torch
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)

        self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.norm1 = torch.nn.BatchNorm2d(out_planes)

        self.norm2 = torch.nn.BatchNorm2d(out_planes)

        # self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):

        # x = self.dropout(x)

        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))

        return x

    def forward_fuse(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x



