#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class ResidualBlock(nn.Module):
    """Khối Residual cho Generator"""
    
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

class Generator(nn.Module):
    """Generator cho việc chuyển đổi giữa các miền"""
    
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d):
        """
        Args:
            input_nc (int): Số kênh đầu vào (1 cho ảnh xám)
            output_nc (int): Số kênh đầu ra (1 cho ảnh xám)
            ngf (int): Số filter cơ bản
            n_blocks (int): Số khối residual
            norm_layer: Lớp chuẩn hóa (Instance, Batch, etc.)
        """
        super(Generator, self).__init__()
        
        # Mã hóa ban đầu
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        # Các khối Residual
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult, norm_layer)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        
        # Mã hóa cuối cùng
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """PatchGAN discriminator"""
    
    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """
        Args:
            input_nc (int): Số kênh đầu vào (1 cho ảnh xám)
            ndf (int): Số filter cơ bản
            n_layers (int): Số lớp trong mạng
            norm_layer: Lớp chuẩn hóa
        """
        super(Discriminator, self).__init__()
        
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.model(x)

class CycleGANModel(nn.Module):
    """Mô hình CycleGAN đầy đủ cho việc chuyển đổi MRI sang CT"""
    
    def __init__(self, input_nc=1, output_nc=1, ngf=64, ndf=64, pool_size=50):
        """
        Args:
            input_nc (int): Số kênh đầu vào
            output_nc (int): Số kênh đầu ra
            ngf (int): Số filter cơ bản cho generator
            ndf (int): Số filter cơ bản cho discriminator
            pool_size (int): Kích thước của bộ nhớ đệm ảnh
        """
        super(CycleGANModel, self).__init__()
        
        # Khởi tạo các generator
        self.netG_A = Generator(input_nc, output_nc, ngf)  # MRI -> CT
        self.netG_B = Generator(output_nc, input_nc, ngf)  # CT -> MRI
        
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
        # Chuyển từ MRI -> CT
        fake_B = self.netG_A(real_A)
        
        result = {'fake_B': fake_B}
        
        if real_B is not None:
            # Chu trình MRI -> CT -> MRI
            rec_A = self.netG_B(fake_B)
            result['rec_A'] = rec_A
            
            # Chuyển từ CT -> MRI
            fake_A = self.netG_B(real_B)
            result['fake_A'] = fake_A
            
            # Chu trình CT -> MRI -> CT
            rec_B = self.netG_A(fake_A)
            result['rec_B'] = rec_B
        
        return result

# Các hàm định nghĩa loss
def gan_loss(pred, target_is_real):
    """Hàm tính GAN loss"""
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)

def cycle_consistency_loss(real_img, reconstructed_img, lambda_cycle=10.0):
    """Hàm tính cycle consistency loss"""
    return lambda_cycle * F.l1_loss(real_img, reconstructed_img)

def identity_loss(real_img, same_img, lambda_identity=0.5):
    """Hàm tính identity loss"""
    return lambda_identity * F.l1_loss(real_img, same_img) 