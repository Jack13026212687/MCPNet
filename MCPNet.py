import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import Backbone_ResNet152_in3

class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SA(nn.Module):
    def __init__(self, in_channels):
        super(SA, self).__init__()
        self.channel_attention = CA(in_channels)

    def forward(self, x):
        attn = self.channel_attention(x)
        return x * attn


class MCP(nn.Module):
    def __init__(self, outchannel):
        super(MCP, self).__init__()
        self.conv1 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=4, padding=4)
        self.conv0 = nn.Conv2d(outchannel, outchannel, kernel_size=1)
        self.conv = nn.Conv2d(5*outchannel, outchannel, kernel_size=1)
        self.rconv = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat((x0, x1, x2, x3, x4), dim=1)
        out = self.conv(out)
        out = out + x
        out = self.rconv(out)
        return out


class MCPNet(nn.Module):
    def __init__(self, n_classes):
        super(MCPNet, self).__init__()

        (self.layer1_rgb, self.layer2_rgb, self.layer3_rgb,
         self.layer4_rgb, self.layer5_rgb) = Backbone_ResNet152_in3(pretrained=True)

        self.rgbconv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.rgbconv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.rgbconv3 = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.rgbconv4 = nn.Conv2d(1024, 64, kernel_size=3, padding=1)
        self.rgbconv5 = nn.Conv2d(2048, 64, kernel_size=3, padding=1)

        self.aspp = MCP(64)
        self.semantic_attention = SA(64)

        self.finalconv = nn.Conv2d(64, n_classes, 1)

    def forward(self, rgb):
        x1 = self.layer1_rgb(rgb)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)

        x1 = self.rgbconv1(x1)
        x2 = self.rgbconv2(x2)
        x3 = self.rgbconv3(x3)
        x4 = self.rgbconv4(x4)
        x5 = self.rgbconv5(x5)

        out = self.aspp(x5)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.semantic_attention(out)
        out = self.finalconv(out)
        out = F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)

        return out


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224).cuda()
    model = MCPNet(n_classes=22).cuda()
    print(model(img).shape)