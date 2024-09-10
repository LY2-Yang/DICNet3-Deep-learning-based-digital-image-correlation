import  torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_,constant_

# 
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)  # 使用GroupNorm代替BatchNorm
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes=2, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.LeakyReLU1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.LeakyReLU1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.LeakyReLU1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class DICNet3(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=[32, 64, 128, 256, 512]):
        super(DICNet3, self).__init__()
        self.activation_function = nn.LeakyReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # ------------------Encoder-----------------
        self.conv1 = nn.Sequential(DepthwiseConv(in_channels, features[0], kernel_size=7),
                                   nn.GroupNorm(8, features[0]),
                                   nn.LeakyReLU(inplace=True),
                                   DepthwiseConv(features[0], features[0], kernel_size=5),
                                   nn.GroupNorm(8, features[0]),
                                   nn.LeakyReLU(inplace=True))  

        self.conv2 = nn.Sequential(DepthwiseConv(features[0], features[1], kernel_size=3),
                                   nn.GroupNorm(8, features[1]),
                                   nn.LeakyReLU(inplace=True),
                                   DepthwiseConv(features[1], features[1], kernel_size=3),
                                   nn.GroupNorm(8, features[1]),
                                   nn.LeakyReLU(inplace=True))

        self.conv3 = nn.Sequential(DepthwiseConv(features[1], features[2], kernel_size=3),
                                   nn.GroupNorm(8, features[2]),
                                   nn.LeakyReLU(inplace=True),
                                   ConvBlock(features[2], features[2]))

        self.conv4 = nn.Sequential(DepthwiseConv(features[2], features[3], kernel_size=3),
                                   nn.GroupNorm(8, features[3]),
                                   nn.LeakyReLU(inplace=True),
                                   ConvBlock(features[3], features[3]))

        self.conv5 = nn.Sequential(DepthwiseConv(features[3], features[4], kernel_size=3),
                                   nn.GroupNorm(8, features[4]),
                                   nn.LeakyReLU(inplace=True),
                                   ConvBlock(features[4], features[4]))

        self.conv5_1 = cbam_block(channel=(features[4]))

        # -------------------Decoder-----------------
        self.up_conv = nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1)

        # 第四层
        self.en_4_1 = nn.Sequential(nn.MaxPool2d(8, 8),
                                    nn.Conv2d(features[0], 32, kernel_size=3, padding=1),
                                    nn.GroupNorm(8, 32),
                                    nn.LeakyReLU(inplace=True))
        self.en_4_2 = nn.Sequential(nn.MaxPool2d(4, 4),
                                    nn.Conv2d(features[1], 32, kernel_size=3, padding=1),
                                    nn.GroupNorm(8, 32),
                                    nn.LeakyReLU(inplace=True))
        self.en_4_3 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    nn.Conv2d(features[2], 32, kernel_size=3, padding=1),
                                    nn.GroupNorm(8, 32),
                                    nn.LeakyReLU(inplace=True))
        self.en_4_4 = nn.Sequential(nn.Conv2d(features[3], 32, kernel_size=3, padding=1),
                                    nn.GroupNorm(8, 32),
                                    nn.LeakyReLU(inplace=True))
        self.en_4_5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    ConvBlock(features[4], 32))

        # 第三层
        self.en_3_1 = nn.Sequential(nn.MaxPool2d(4, 4),
                                    nn.Conv2d(features[0], 32, kernel_size=3, padding=1),
                                    nn.GroupNorm(8, 32),
                                    nn.LeakyReLU(inplace=True))
        self.en_3_2 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    nn.Conv2d(features[1], 32, kernel_size=3, padding=1),
                                    nn.GroupNorm(8, 32),
                                    nn.LeakyReLU(inplace=True))
        self.en_3_3 = nn.Sequential(nn.Conv2d(features[2], 32, kernel_size=3, padding=1),
                                    nn.GroupNorm(8, 32),
                                    nn.LeakyReLU(inplace=True))
        self.en_3_4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    ConvBlock(160, 32))
        self.en_3_5 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'),
                                    ConvBlock(features[4], 32))

        # 第二层
        self.en_2_1 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    nn.Conv2d(features[0], 32, kernel_size=3, padding=1),
                                    nn.GroupNorm(8, 32),
                                    nn.LeakyReLU(inplace=True))
        self.en_2_2 = nn.Sequential(nn.Conv2d(features[1], 32, kernel_size=3, padding=1),
                                    nn.GroupNorm(8, 32),
                                    nn.LeakyReLU(inplace=True))
        self.en_2_3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    ConvBlock(160, 32))
        self.en_2_4 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'),
                                    ConvBlock(160, 32))
        self.en_2_5 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear'),
                                    ConvBlock(features[4], 32))

        # 第一层
        self.en_1_1 = nn.Sequential(nn.Conv2d(features[0], 32, kernel_size=3, padding=1),
                                    nn.GroupNorm(8, 32),
                                    nn.LeakyReLU(inplace=True))
        self.en_1_2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    ConvBlock(160, 32))
        self.en_1_3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'),
                                    ConvBlock(160, 32))
        self.en_1_4 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear'),
                                    ConvBlock(160, 32))
        self.en_1_5 = nn.Sequential(nn.Upsample(scale_factor=16, mode='bilinear'),
                                    ConvBlock(features[4], 32))

        # 
        self.output1 = nn.Sequential(ConvBlock(160, 32),
                                     nn.Conv2d(features[0], out_channels,kernel_size=3,stride=1,padding=1),
                                     nn.BatchNorm2d(out_channels),
                                     nn.LeakyReLU(inplace=True) )

        # 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        # 
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        x5 = self.pool(x4)
        x5 = self.conv5(x5)
        x5 = self.conv5_1(x5)

        # 
        x_4_1 = self.en_4_1(x1)
        x_4_2 = self.en_4_2(x2)
        x_4_3 = self.en_4_3(x3)
        x_4_4 = self.en_4_4(x4)
        x_4_5 = self.en_4_5(x5)
        x_4 = self.activation_function(self.up_conv(torch.cat((x_4_1, x_4_2, x_4_3, x_4_4, x_4_5), dim=1)))

        # 
        x_3_1 = self.en_3_1(x1)
        x_3_2 = self.en_3_2(x2)
        x_3_3 = self.en_3_3(x3)
        x_3_4 = self.en_3_4(x_4)
        x_3_5 = self.en_3_5(x5)
        x_3 = self.activation_function(self.up_conv(torch.cat((x_3_1, x_3_2, x_3_3, x_3_4, x_3_5), dim=1)))

        # 
        x_2_1 = self.en_2_1(x1)
        x_2_2 = self.en_2_2(x2)
        x_2_3 = self.en_2_3(x_3)
        x_2_4 = self.en_2_4(x_4)
        x_2_5 = self.en_2_5(x5)
        x_2 = self.activation_function(self.up_conv(torch.cat((x_2_1, x_2_2, x_2_3, x_2_4, x_2_5), dim=1)))

        #
        x_1_1 = self.en_1_1(x1)
        x_1_2 = self.en_1_2(x_2)
        x_1_3 = self.en_1_3(x_3)
        x_1_4 = self.en_1_4(x_4)
        x_1_5 = self.en_1_5(x5)
        x_1 = self.activation_function(self.up_conv(torch.cat((x_1_1, x_1_2, x_1_3, x_1_4, x_1_5), dim=1)))

        # 
        output_x = self.output1(x_1)
        return output_x
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn((1, 2, 480, 480)).to(device)
    model =DICNet3(in_channels=2,out_channels=2).to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    outputs = model(inputs)
    print(outputs.shape)
#4.74M