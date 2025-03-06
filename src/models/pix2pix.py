#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class UNetGenerator(nn.Module):
    """Generator dựa trên kiến trúc UNet cho Pix2Pix"""
    
    def __init__(self, input_nc=1, output_nc=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        Args:
            input_nc (int): Số kênh đầu vào (1 cho ảnh xám)
            output_nc (int): Số kênh đầu ra (1 cho ảnh xám)
            ngf (int): Số filter cơ bản
            norm_layer: Lớp chuẩn hóa (Instance, Batch, etc.)
            use_dropout (bool): Sử dụng dropout hay không
        """
        super(UNetGenerator, self).__init__()
        
        # Tạo encoder blocks
        # Encoder block 1: input -> ngf
        self.enc1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        # Encoder block 2: ngf -> ngf*2
        self.enc2 = self._encoder_block(ngf, ngf*2, norm_layer)
        # Encoder block 3: ngf*2 -> ngf*4
        self.enc3 = self._encoder_block(ngf*2, ngf*4, norm_layer)
        # Encoder block 4: ngf*4 -> ngf*8
        self.enc4 = self._encoder_block(ngf*4, ngf*8, norm_layer)
        # Encoder block 5: ngf*8 -> ngf*8
        self.enc5 = self._encoder_block(ngf*8, ngf*8, norm_layer)
        # Encoder block 6: ngf*8 -> ngf*8
        self.enc6 = self._encoder_block(ngf*8, ngf*8, norm_layer)
        # Encoder block 7: ngf*8 -> ngf*8
        self.enc7 = self._encoder_block(ngf*8, ngf*8, norm_layer)
        # Encoder block 8: ngf*8 -> ngf*8 (không sử dụng normalization)
        self.enc8 = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1)
        
        # Tạo decoder blocks
        # Decoder block 1: ngf*8 -> ngf*8
        self.dec1 = self._decoder_block(ngf*8, ngf*8, norm_layer, use_dropout=True)
        # Decoder block 2: ngf*16 -> ngf*8 (Do skip connection nên input là ngf*16)
        self.dec2 = self._decoder_block(ngf*16, ngf*8, norm_layer, use_dropout=True)
        # Decoder block 3: ngf*16 -> ngf*8
        self.dec3 = self._decoder_block(ngf*16, ngf*8, norm_layer, use_dropout=True)
        # Decoder block 4: ngf*16 -> ngf*4
        self.dec4 = self._decoder_block(ngf*16, ngf*4, norm_layer)
        # Decoder block 5: ngf*8 -> ngf*2
        self.dec5 = self._decoder_block(ngf*8, ngf*2, norm_layer)
        # Decoder block 6: ngf*4 -> ngf
        self.dec6 = self._decoder_block(ngf*4, ngf, norm_layer)
        # Decoder block 7: ngf*2 -> output_nc
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def _encoder_block(self, in_channels, out_channels, norm_layer):
        """Tạo block encoder"""
        return nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            norm_layer(out_channels)
        )
    
    def _decoder_block(self, in_channels, out_channels, norm_layer, use_dropout=False):
        """Tạo block decoder"""
        layers = [
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            norm_layer(out_channels)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decoder path với skip connections
        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], 1))
        d3 = self.dec3(torch.cat([d2, e6], 1))
        d4 = self.dec4(torch.cat([d3, e5], 1))
        d5 = self.dec5(torch.cat([d4, e4], 1))
        d6 = self.dec6(torch.cat([d5, e3], 1))
        d7 = self.dec7(torch.cat([d6, e2], 1))
        
        return d7

class PatchGANDiscriminator(nn.Module):
    """Discriminator dựa trên PatchGAN cho Pix2Pix"""
    
    def __init__(self, input_nc=2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Args:
            input_nc (int): Số kênh đầu vào (2 cho cặp ảnh MRI và CT)
            ndf (int): Số filter cơ bản
            n_layers (int): Số lớp trong discriminator
            norm_layer: Lớp chuẩn hóa
        """
        super(PatchGANDiscriminator, self).__init__()
        
        # Lớp đầu tiên không có normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        
        # Xác định số kênh đầu vào và đầu ra cho các lớp sau
        nf_mult = 1
        nf_mult_prev = 1
        
        # Thêm các lớp trung gian
        layers = []
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.extend([
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ])
            
        # Thêm một lớp nữa giữ nguyên kích thước không gian
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.extend([
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ])
        
        # Lớp cuối cùng tạo ra map 1 kênh
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        return self.model(x)

class Pix2PixModel(nn.Module):
    """Mô hình Pix2Pix đầy đủ cho việc chuyển đổi MRI sang CT"""
    
    def __init__(self, input_nc=1, output_nc=1, ngf=64, ndf=64, norm_layer=nn.BatchNorm2d):
        """
        Args:
            input_nc (int): Số kênh đầu vào
            output_nc (int): Số kênh đầu ra
            ngf (int): Số filter cơ bản cho generator
            ndf (int): Số filter cơ bản cho discriminator
            norm_layer: Lớp chuẩn hóa
        """
        super(Pix2PixModel, self).__init__()
        
        # Generator
        self.netG = UNetGenerator(input_nc, output_nc, ngf, norm_layer)
        
        # Discriminator nhận vào ảnh MRI và ảnh CT (thật hoặc giả)
        self.netD = PatchGANDiscriminator(input_nc + output_nc, ndf, norm_layer=norm_layer)
    
    def forward(self, real_A, real_B=None):
        """
        Forward pass
        Args:
            real_A (tensor): Ảnh MRI thật
            real_B (tensor, optional): Ảnh CT thật
        """
        # Tạo ảnh CT từ MRI
        fake_B = self.netG(real_A)
        
        # Trả về kết quả
        return {
            'real_A': real_A,
            'fake_B': fake_B,
            'real_B': real_B
        }
    
    def discriminate(self, real_A, fake_B):
        """
        Phân biệt ảnh thật/giả
        Args:
            real_A (tensor): Ảnh MRI thật
            fake_B (tensor): Ảnh CT giả
        """
        # Nối đầu vào với đầu ra
        fake_AB = torch.cat([real_A, fake_B], 1)
        return self.netD(fake_AB) 