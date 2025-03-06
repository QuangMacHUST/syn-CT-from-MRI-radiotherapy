#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Khối tích chập kép cho UNet"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer=nn.InstanceNorm2d):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Khối giảm kích thước trong UNet"""
    
    def __init__(self, in_channels, out_channels, norm_layer=nn.InstanceNorm2d):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_layer=norm_layer)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Khối tăng kích thước trong UNet"""
    
    def __init__(self, in_channels, out_channels, bilinear=True, norm_layer=nn.InstanceNorm2d):
        super(Up, self).__init__()
        
        # Lựa chọn phương pháp upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_layer=norm_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_layer=norm_layer)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Xử lý kích thước cho padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Nối các feature map
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Lớp tích chập đầu ra cuối cùng"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet cho chuyển đổi MRI sang CT"""
    
    def __init__(self, input_nc=1, output_nc=1, ngf=64, bilinear=True, norm_layer=nn.InstanceNorm2d):
        """
        Args:
            input_nc (int): Số kênh đầu vào (1 cho ảnh xám)
            output_nc (int): Số kênh đầu ra (1 cho ảnh xám)
            ngf (int): Số filter cơ bản
            bilinear (bool): Sử dụng upsampling bilinear hay transposed convolution
            norm_layer: Lớp chuẩn hóa (Instance, Batch, etc.)
        """
        super(UNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        
        # Khối encoder (downsampling)
        self.inc = DoubleConv(input_nc, ngf, norm_layer=norm_layer)
        self.down1 = Down(ngf, ngf * 2, norm_layer=norm_layer)
        self.down2 = Down(ngf * 2, ngf * 4, norm_layer=norm_layer)
        self.down3 = Down(ngf * 4, ngf * 8, norm_layer=norm_layer)
        factor = 2 if bilinear else 1
        self.down4 = Down(ngf * 8, ngf * 16 // factor, norm_layer=norm_layer)
        
        # Khối decoder (upsampling)
        self.up1 = Up(ngf * 16, ngf * 8 // factor, bilinear, norm_layer=norm_layer)
        self.up2 = Up(ngf * 8, ngf * 4 // factor, bilinear, norm_layer=norm_layer)
        self.up3 = Up(ngf * 4, ngf * 2 // factor, bilinear, norm_layer=norm_layer)
        self.up4 = Up(ngf * 2, ngf, bilinear, norm_layer=norm_layer)
        self.outc = OutConv(ngf, output_nc)
        
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path với skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Lớp cuối cùng
        x = self.outc(x)
        
        # Áp dụng tanh để đưa đầu ra về khoảng [-1, 1]
        return torch.tanh(x)

class UNetModel(nn.Module):
    """Mô hình UNet đầy đủ cho việc chuyển đổi MRI sang CT"""
    
    def __init__(self, input_nc=1, output_nc=1, ngf=64, bilinear=True):
        """
        Args:
            input_nc (int): Số kênh đầu vào
            output_nc (int): Số kênh đầu ra
            ngf (int): Số filter cơ bản
            bilinear (bool): Phương pháp upsampling
        """
        super(UNetModel, self).__init__()
        
        # Khởi tạo UNet
        self.netG = UNet(input_nc, output_nc, ngf, bilinear)
    
    def forward(self, real_A, real_B=None):
        """
        Forward pass
        Args:
            real_A (tensor): Ảnh MRI thật
            real_B (tensor, optional): Ảnh CT thật (không sử dụng trong forward)
        """
        # Tạo ảnh CT từ MRI
        fake_B = self.netG(real_A)
        
        # Trả về kết quả
        return {
            'real_A': real_A,
            'fake_B': fake_B,
            'real_B': real_B
        } 