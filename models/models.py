import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.srm import SRMConv2d_simple, SRMConv2d_Separate
from models.common.torch_dct import dct_2d, idct_2d


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.conv1x1 = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
                )
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, True)
        

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.lrelu(res)
        
        res = self.conv2(res)
        res = self.bn2(res)

        if hasattr(self, 'conv1x1'):
            x = self.conv1x1(x)
        
        out = self.lrelu(res + x)
        return out



class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_num=1, downsample: bool=True):
        super().__init__()
        if downsample == True:
            self.downsample = nn.Conv2d(in_channels, in_channels, 4, 2, 1)
        if block_num == 1:
            self.layers = ResBlock(in_channels, out_channels)
        else:
            layers = [ResBlock(in_channels, out_channels)]
            for i in range(block_num - 1):
                layers.append(ResBlock(out_channels, out_channels))
            self.layers = nn.Sequential(*layers)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        if hasattr(self, 'downsample'):
            x = self.downsample(x)
        out = self.layers(x)
        out = self.ca(out)
        out = self.sa(out)
        return out
            

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample: bool=True, up_mode='bilinear'):
        super().__init__()
        if upsample:
            if up_mode == 'deconv':
                self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
            elif up_mode == 'nearest':
                self.upsample = nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)
            elif up_mode == 'bilinear':
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                raise ValueError('Wrong upsample mode')

        self.layers = ResBlock(in_channels, out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        if hasattr(self, 'upsample'):
            x = self.upsample(x)
        out = self.layers(x)
        out = self.ca(out)
        out = self.sa(out)
        return out


class Classifier(nn.Module):
    def __init__(self, in_channels, n_class=2) -> None:
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_channels, n_class)

    def forward(self, x):
        y = self.aap(x)
        y = y.squeeze(3).squeeze(2)
        return self.fc(y)



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_channels = 32
        self.rates = [1, 2, 4, 8]
        self.block_num = [4, 4, 4]
        self.short_cut = False
        self.downsample_time = 2
        self.img_size = 256

        # self.freq = SRMConv2d_simple(3)
        # self.freq = BayarConv2d(3, 3, 5, padding=2)
        self.freq = FreqFilter(3, im_size=self.img_size, type='dct')

        self.input_layer = nn.Conv2d(3, self.conv_channels, 7, padding=3)
        self.freq_input_layer = nn.Conv2d(3, self.conv_channels, 7, padding=3)

        # encoder
        self.encoder = nn.ModuleList()

        channels = self.conv_channels
        for i in range(self.downsample_time):
            self.encoder.append(nn.Sequential(
                DownBlock(channels, channels * 2, 1, True),
                *[AOTBlock(channels * 2, self.rates) for _ in range(self.block_num[i])]
            ))
            channels *= 2

        self.freq_encoder = nn.ModuleList()

        for i in range(self.downsample_time):
            self.freq_encoder.append(nn.Sequential(
                DownBlock(self.conv_channels, self.conv_channels * 2, 1, True),
                *[AOTBlock(self.conv_channels * 2, self.rates) for _ in range(self.block_num[i])]
            ))
            self.conv_channels *= 2

        self.fusion = FusionModule(self.conv_channels * 2, self.conv_channels)

        # classifier
        self.classifier = Classifier(self.conv_channels, 2)

        # decoder
        self.decoder = nn.ModuleList()

        for _ in range(self.downsample_time):
            if self.short_cut:
                self.decoder.append(UpBlock(self.conv_channels * 3, self.conv_channels // 2, True))
            else:
                self.decoder.append(UpBlock(self.conv_channels, self.conv_channels // 2, True))
            self.conv_channels //= 2
        
        self.output_layer = nn.Conv2d(self.conv_channels, 1, 1)

        # init_weights
        # self.init_weights()


    def forward(self, image):
        x = self.input_layer(image)
        # encoder
        xs = []
        for layer in self.encoder:
            x = layer(x)
            xs.append(x)

        freq_f = self.freq(image)
        freq_x = self.freq_input_layer(freq_f)
        
        # encoder
        freq_xs = []
        for i, layer in enumerate(self.freq_encoder, 1):
            freq_x = layer(freq_x)
            freq_xs.append(freq_x)

        # fusion
        x = self.fusion(xs[-1], freq_xs[-1])

        # classifier
        pred = self.classifier(x)
        # decoder
        for i, layer in enumerate(self.decoder, 1):
            if self.short_cut:
                x = torch.concat([x, xs[-i], freq_xs[-i]], dim=1)
            x = layer(x)

        x = self.output_layer(x)
        return x, pred
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    conv3x3(dim, dim // len(rates), padding=0, dilation=rate),
                    nn.BatchNorm2d(dim // len(rates)),
                    nn.LeakyReLU(0.2, True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            conv3x3(dim, dim, padding=0, dilation=1),
            nn.BatchNorm2d(dim))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            conv3x3(dim, dim, padding=0, dilation=1),
            nn.BatchNorm2d(dim))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        # mask = my_layer_norm(self.gate(x))
        mask = self.gate(x)
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


"""
Channel Attention and Spaitial Attention from    
Woo, S., Park, J., Lee, J.Y., & Kweon, I. CBAM: Convolutional Block Attention Module. ECCV2018.
"""

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        attention = self.sigmoid(avgout + maxout)
        return attention * x + x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avgout, maxout], dim=1)
        y = self.conv(y)
        attention = self.sigmoid(y)
        return attention * x + x


class FusionModule(nn.Module):
    def __init__(self, in_channels=2048*2, out_channels=2048):
        super().__init__()
        self.convblk = nn.Sequential(
            conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        self.ca = ChannelAttention(out_channels, ratio=8)
        self.sa = SpatialAttention()
        # self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = self.ca(fuse_fea)
        fuse_fea = self.sa(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


# class BayarConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

#         super(BayarConv2d, self).__init__()
#         # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
#         self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
#                                    requires_grad=True)


#     def bayarConstraint(self):
#         self.kernel.data = self.kernel.permute(2, 0, 1)
#         self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
#         self.kernel.data = self.kernel.permute(1, 2, 0)
#         ctr = self.kernel_size ** 2 // 2
#         real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
#         real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
#         return real_kernel

#     def forward(self, x):
#         x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
#         return x


class FreqFilter(nn.Module):
    def __init__(self, dim, im_size=256, type='dct', fp64=False) -> None:
        super().__init__()
        self.type = type
        if type == 'dct':
            # self.freq_weight = nn.Parameter(
            #     torch.randn(dim, im_size, im_size, dtype=torch.float32) * 0.02
            # )
            mask = self.get_high_freq_mask(im_size)
            self.freq_weight = nn.Parameter(mask, requires_grad=False)
        else:
            self.freq_weight = nn.Parameter(
                torch.randn(dim, im_size, im_size // 2 + 1, 2, dtype=torch.float32) * 0.02
            )
        self.fp64 = fp64
    
    def forward(self, x):
        if self.fp64:
            dtype = x.dtype
            x = x.to(torch.float64)

        if self.type == 'dct':
            x = dct_2d(x, norm='ortho')
            x = x * self.freq_weight
            x = idct_2d(x, norm='ortho')
        else:
            x = torch.fft.rfft2(x, norm="ortho")
            weight = torch.view_as_complex(self.freq_weight)
            x = x * weight
            x = torch.fft.irfft2(x, norm='ortho')
        
        if self.fp64:
            x = x.to(dtype)

        return x
    
    def get_high_freq_mask(self, im_size, rate=8):
        mask = torch.ones((im_size, im_size))
        t = im_size // rate
        for i in range(t):
            mask[i, :t - i] = 0
        return mask