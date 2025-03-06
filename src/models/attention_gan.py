#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Khối Residual cơ bản"""
    
    def __init__(self, dim, norm_layer=nn.InstanceNorm2d):
        super(ResidualBlock, self).__init__()
        self.conv_block = self._build_conv_block(dim, norm_layer)
    
    def _build_conv_block(self, dim, norm_layer):
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim)
        ]
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        return x + self.conv_block(x)

class AttentionModule(nn.Module):
    """Module tạo bản đồ attention"""
    
    def __init__(self, in_channels, out_channels, norm_layer=nn.InstanceNorm2d):
        super(AttentionModule, self).__init__()
        
        # Mạng tích chập cho attention
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.attention(x)

class AttentionGenerator(nn.Module):
    """Generator sử dụng cơ chế attention"""
    
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d):
        """
        Args:
            input_nc (int): Số kênh đầu vào (1 cho ảnh xám)
            output_nc (int): Số kênh đầu ra (1 cho ảnh xám)
            ngf (int): Số filter cơ bản
            n_blocks (int): Số khối residual
            norm_layer: Lớp chuẩn hóa (Instance, Batch, etc.)
        """
        super(AttentionGenerator, self).__init__()
        
        # Mã hóa ban đầu
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        # Residual blocks
        mult = 2 ** n_downsampling
        res_blocks = []
        for i in range(n_blocks):
            res_blocks += [ResidualBlock(ngf * mult, norm_layer=norm_layer)]
        
        # Upsampling
        decoder = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        
        # Đầu ra
        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        # Module attention
        self.attention = AttentionModule(ngf * mult, 1, norm_layer)
        
        # Mạng tạo ảnh gốc
        self.encoder = nn.Sequential(*encoder)
        self.res_blocks = nn.Sequential(*res_blocks)
        self.decoder = nn.Sequential(*decoder)
    
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Residual blocks
        features = self.res_blocks(x)
        
        # Tạo bản đồ attention
        attention_map = self.attention(features)
        
        # Decoder
        output = self.decoder(features)
        
        # Áp dụng attention
        # Bản đồ attention được kéo giãn để phù hợp với kích thước đầu ra
        attention_map_resized = F.interpolate(attention_map, size=output.size()[2:], mode='bilinear', align_corners=True)
        
        # Trả về cả bản đồ attention và ảnh đầu ra
        return output, attention_map_resized

class Discriminator(nn.Module):
    """Discriminator cho AttentionGAN"""
    
    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """
        Args:
            input_nc (int): Số kênh đầu vào
            ndf (int): Số filter cơ bản
            n_layers (int): Số lớp trong discriminator
            norm_layer: Lớp chuẩn hóa
        """
        super(Discriminator, self).__init__()
        
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.model(x)

class AttentionGANModel(nn.Module):
    """Mô hình AttentionGAN đầy đủ cho việc chuyển đổi MRI sang CT"""
    
    def __init__(self, input_nc=1, output_nc=1, ngf=64, ndf=64, n_blocks=9):
        """
        Args:
            input_nc (int): Số kênh đầu vào
            output_nc (int): Số kênh đầu ra
            ngf (int): Số filter cơ bản cho generator
            ndf (int): Số filter cơ bản cho discriminator
            n_blocks (int): Số khối residual trong generator
        """
        super(AttentionGANModel, self).__init__()
        
        # Khởi tạo các generator
        self.netG_A = AttentionGenerator(input_nc, output_nc, ngf, n_blocks)  # MRI -> CT
        self.netG_B = AttentionGenerator(output_nc, input_nc, ngf, n_blocks)  # CT -> MRI
        
        # Khởi tạo các discriminator
        self.netD_A = Discriminator(output_nc, ndf)  # Phân biệt ảnh CT thật/giả
        self.netD_B = Discriminator(input_nc, ndf)  # Phân biệt ảnh MRI thật/giả
    
    def forward(self, real_A, real_B=None):
        """
        Forward pass
        Args:
            real_A (tensor): Ảnh MRI thật
            real_B (tensor, optional): Ảnh CT thật (nếu có)
        """
        # MRI -> CT
        fake_B, attention_A = self.netG_A(real_A)
        
        # CT -> MRI (nếu cần cycle consistency)
        if real_B is not None:
            fake_A, attention_B = self.netG_B(real_B)
            # Cycle MRI -> CT -> MRI
            rec_A, _ = self.netG_B(fake_B)
            # Cycle CT -> MRI -> CT
            rec_B, _ = self.netG_A(fake_A)
            
            return {
                'real_A': real_A, 'fake_B': fake_B, 'rec_A': rec_A, 'attention_A': attention_A,
                'real_B': real_B, 'fake_A': fake_A, 'rec_B': rec_B, 'attention_B': attention_B
            }
            
        # Trường hợp chỉ chuyển đổi một chiều MRI -> CT
        return {
            'real_A': real_A,
            'fake_B': fake_B,
            'attention_A': attention_A,
            'real_B': real_B
        } 