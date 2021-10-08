"""
References:
1. Udacity AI for Healthcare Nano Degree Exercise
2. https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation/code
3. https://arxiv.org/pdf/1505.04597.pdf
4. https://naokishibuya.medium.com/up-sampling-with-transposed-convolution-9ae4f2df52d0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

"""
U-Net
"""

class UNet(nn.Module):

    def __init__(self, dim_in=1, dim_out=1, d_hidden = 32):
        """
        Arguments:
            dim_in: number of channels of the input tensor (default: 1 == grayscale)
            dim_out: number of channels of the output tensor (default: 1 == grayscale)
            d_hidden: dimension of the hidden layers of the first convolution layer
        """

        super(UNet, self).__init__()

        self.d_hidden = d_hidden

        """
        Encoder
        """
        # 1st encode unet_block
        self.EncoderBlock1 = self.unet_block(dim_in=dim_in, dim_out=d_hidden, prefix="enc1")
        # 2nd encode unet_block
        self.EncoderBlock2 = self.unet_block(dim_in=d_hidden, dim_out=2*d_hidden, prefix="enc2")
        # 3rd encode unet_block
        self.EncoderBlock3 = self.unet_block(dim_in=2*d_hidden, dim_out=4*d_hidden, prefix="enc3")
        # 4th encode unet_block
        self.EncoderBlock4 = self.unet_block(dim_in=4*d_hidden, dim_out=8*d_hidden, prefix="enc4")
        # bottleneck (encoder)
        self.EncoderBottleNeck = self.unet_block(dim_in=8*d_hidden, dim_out=16*d_hidden, prefix="bottleneck")
        # 4th decoder transpose conv + block
        self.DecoderTransConv4 = nn.ConvTranspose2d(in_channels=16*d_hidden, out_channels=8*d_hidden, kernel_size=2, stride=2)
        self.DecoderBlock4 = self.unet_block(dim_in=2*8*d_hidden, dim_out=8*d_hidden, prefix="dec4")
        # 3rd decoder transpose conv + block
        self.DecoderTransConv3 = nn.ConvTranspose2d(in_channels=8*d_hidden, out_channels=4*d_hidden, kernel_size=2, stride=2)
        self.DecoderBlock3 = self.unet_block(dim_in=2*4*d_hidden, dim_out=4*d_hidden, prefix="dec3")
        # 2nd decoder transpose conv + block
        self.DecoderTransConv2 = nn.ConvTranspose2d(in_channels=4*d_hidden, out_channels=2*d_hidden, kernel_size=2, stride=2)
        self.DecoderBlock2 = self.unet_block(dim_in=2*2*d_hidden, dim_out=2*d_hidden, prefix="dec2")
        # 1st decoder transpose conv + block
        self.DecoderTransConv1 = nn.ConvTranspose2d(in_channels=2*d_hidden, out_channels=d_hidden, kernel_size=2, stride=2)
        self.DecoderBlock1 = self.unet_block(dim_in=2*d_hidden, dim_out=d_hidden, prefix="dec1")
        # last Conv2D
        self.LastConv = nn.ConvTranspose2d(in_channels=d_hidden, out_channels=dim_out, kernel_size=1)


    def forward(self, x):
        # encoders
        enc1_out = self.EncoderBlock1(x)
        enc2_out = self.EncoderBlock2(F.max_pool2d(enc1_out, kernel_size=2, stride=2))
        enc3_out = self.EncoderBlock3(F.max_pool2d(enc2_out, kernel_size=2, stride=2))
        enc4_out = self.EncoderBlock4(F.max_pool2d(enc3_out, kernel_size=2, stride=2))
        # bottle neck
        bottleneck_out = self.EncoderBottleNeck(F.max_pool2d(enc4_out, kernel_size=2, stride=2))
        # decoders
        dec4_out = self.DecoderBlock4(torch.cat((self.DecoderTransConv4(bottleneck_out), enc4_out), dim=1))
        dec3_out = self.DecoderBlock3(torch.cat((self.DecoderTransConv3(dec4_out), enc3_out), dim=1))
        dec2_out = self.DecoderBlock2(torch.cat((self.DecoderTransConv2(dec3_out), enc2_out), dim=1))
        dec1_out = self.DecoderBlock1(torch.cat((self.DecoderTransConv1(dec2_out), enc1_out), dim=1))
        # last conv2d
        y = F.softmax(self.LastConv(dec1_out), dim=1)

        return y


    def unet_block(self, dim_in, dim_out, prefix):
        """
        Create a block consisting of:
            Conv2d > BN > Relu > Conv2d > BN > Relu

        Arguments:
            dim_in: number of input channels
            dim_out: nuumber of output channels
            prefix: name prefix for each layer
        """
        # first conv2D layer
        conv1 = nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=3, padding='same', bias=False)
        # second conv2D layer
        conv2 = nn.Conv2d(in_channels=dim_out, out_channels=dim_out, kernel_size=3, padding='same', bias=False)

        u_block = nn.Sequential(OrderedDict([(prefix + "conv1", conv1),
                                             (prefix + "bn1", nn.BatchNorm2d(num_features=dim_out)),
                                             (prefix + "relu1", nn.ReLU(inplace=True)),
                                             (prefix + "conv2", conv2),
                                             (prefix + "bn2", nn.BatchNorm2d(num_features=dim_out)),
                                             (prefix + "relu2", nn.ReLU(inplace=True))]))

        return u_block
