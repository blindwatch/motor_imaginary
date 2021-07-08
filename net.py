import torch
import torch.nn as nn


class BasicBlock(nn.Module):        #残差网络的基本块
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = ChannelAttention(out_channels)
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            self.shrinkage,       
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))


class Shrinkage(nn.Module):     #收缩残差网络的基本快
    def __init__(self, channel):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class DRSANet(nn.Module):       #深度收缩残差网络

    def __init__(self, channels, num_block, num_classes=2):
        super().__init__()
        self.block = BasicBlock
        self.channel = channels
        self.in_channels = channels[1]
        self.cbam = ChannelAttention(self.in_channels)
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(inplace=True),
            self.cbam)
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.RSBU1_x = self._make_layer(self.block, channels[2], num_block[0], 2)
        self.RSBU2_x = self._make_layer(self.block, channels[3], num_block[1], 2)
        self.RSBU3_x = self._make_layer(self.block, channels[4], num_block[2], 2)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.Sequential(
            ChannelAttention(self.in_channels),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.RSBU1_x(output)
        output = self.RSBU2_x(output)
        output = self.RSBU3_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class ChannelAttention(nn.Module):      #注意力机制
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_raw = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x_raw



class CANet(nn.Module):     #注意力卷积网络

    def __init__(self, channels, num_block=2, num_classes=2):
        super().__init__()
        self.channel = channels
        self.in_channels = channels[1]
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(inplace=True),
            ChannelAttention(self.in_channels, 4)
            )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(inplace=True),
            ChannelAttention(channels[2], 4))
        self.conv3 = nn.Sequential(
            nn.Conv1d(channels[2], channels[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(channels[3]),
            nn.ReLU(inplace=True),)
        self.avg_pool = nn.Sequential(
            ChannelAttention(channels[3], 4),
            nn.BatchNorm1d(channels[3]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1))
        self.fc2 = nn.Linear(channels[3], 2)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc2(output)

        return output



class BPnet(nn.Module):     #BP网络
    def __init__(self, layers):
        super(BPnet, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.bn1 = nn.BatchNorm1d(layers[1])
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(layers[1], layers[2])
        #self.bn2 = nn.BatchNorm1d(layers[2])
        #self.fc3 = nn.Linear(layers[2], layers[3])

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.bn2(out)
        #out = self.relu(out)
        #out = self.fc3(out)
        return out
