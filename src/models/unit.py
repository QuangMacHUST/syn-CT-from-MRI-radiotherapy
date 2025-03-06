#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianNoiseLayer(nn.Module):
    """Lớp thêm nhiễu Gaussian"""
    
    def __init__(self, mean=0, std=0.1):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x

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

class Encoder(nn.Module):
    """Encoder cho UNIT, chuyển từ không gian ảnh sang không gian tiềm ẩn chung (shared latent space)"""
    
    def __init__(self, input_nc=1, nef=64, n_blocks=3, norm_layer=nn.InstanceNorm2d):
        super(Encoder, self).__init__()
        
        # Khối encoder ban đầu
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, nef, kernel_size=7, padding=0),
            norm_layer(nef),
            nn.ReLU(True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(nef * mult, nef * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(nef * mult * 2),
                nn.ReLU(True)
            ]
        
        # Các khối residual
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(nef * mult, norm_layer)]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    """Decoder cho UNIT, chuyển từ không gian tiềm ẩn chung sang không gian ảnh"""
    
    def __init__(self, output_nc=1, ndf=64, n_blocks=3, norm_layer=nn.InstanceNorm2d):
        super(Decoder, self).__init__()
        
        # Số kênh tính năng sau encoder
        mult = 4  # 2^2 từ 2 lớp downsampling
        
        # Khối residual
        model = []
        for i in range(n_blocks):
            model += [ResidualBlock(ndf * mult, norm_layer)]
        
        # Upsampling
        n_upsampling = 2
        for i in range(n_upsampling):
            mult = 2 ** (n_upsampling - i)
            model += [
                nn.ConvTranspose2d(ndf * mult, int(ndf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(int(ndf * mult / 2)),
                nn.ReLU(True)
            ]
        
        # Khối đầu ra
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ndf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class VAEEncoder(nn.Module):
    """VAE Encoder cho UNIT, tạo phân phối trong không gian tiềm ẩn"""
    
    def __init__(self, input_dim, output_dim, norm_layer=nn.InstanceNorm2d):
        super(VAEEncoder, self).__init__()
        
        # Mạng tính toán trung bình và phương sai
        self.fc_mu = nn.Conv2d(input_dim, output_dim, kernel_size=1, padding=0)
        self.fc_var = nn.Conv2d(input_dim, output_dim, kernel_size=1, padding=0)
        
        # Lớp nhiễu Gaussian để reparameterization trick
        self.noise_layer = GaussianNoiseLayer()
    
    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class Discriminator(nn.Module):
    """PatchGAN Discriminator cho UNIT"""
    
    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()
        
        # Mạng phân biệt
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

class UNITModel(nn.Module):
    """Mô hình UNIT đầy đủ cho việc chuyển đổi MRI sang CT"""
    
    def __init__(self, input_nc=1, output_nc=1, nef=64, ndf=64, n_blocks=3, latent_dim=512):
        """
        Args:
            input_nc (int): Số kênh đầu vào
            output_nc (int): Số kênh đầu ra
            nef (int): Số filter cơ bản cho encoder
            ndf (int): Số filter cơ bản cho decoder/discriminator
            n_blocks (int): Số khối residual trong encoder/decoder
            latent_dim (int): Số chiều của không gian tiềm ẩn
        """
        super(UNITModel, self).__init__()
        
        # Các thành phần cho MRI
        self.encoder_A = Encoder(input_nc, nef, n_blocks)
        self.gen_mu_var_A = VAEEncoder(nef * 4, latent_dim)  # 4 = 2^2
        
        # Các thành phần cho CT
        self.encoder_B = Encoder(output_nc, nef, n_blocks)
        self.gen_mu_var_B = VAEEncoder(nef * 4, latent_dim)
        
        # Các decoder cho cả hai miền
        self.decoder_A = Decoder(input_nc, ndf, n_blocks)
        self.decoder_B = Decoder(output_nc, ndf, n_blocks)
        
        # Các discriminator
        self.disc_A = Discriminator(input_nc, ndf)
        self.disc_B = Discriminator(output_nc, ndf)
    
    def encode(self, input_tensor, domain='A'):
        """Mã hóa ảnh thành không gian tiềm ẩn"""
        if domain == 'A':
            content = self.encoder_A(input_tensor)
            z, mu, log_var = self.gen_mu_var_A(content)
        else:
            content = self.encoder_B(input_tensor)
            z, mu, log_var = self.gen_mu_var_B(content)
        return content, z, mu, log_var
    
    def decode(self, content, domain='A'):
        """Giải mã từ không gian tiềm ẩn sang ảnh"""
        if domain == 'A':
            return self.decoder_A(content)
        else:
            return self.decoder_B(content)
    
    def forward(self, real_A, real_B=None):
        """
        Forward pass
        Args:
            real_A (tensor): Ảnh MRI thật
            real_B (tensor, optional): Ảnh CT thật (nếu có)
        """
        # Mã hóa MRI
        content_A, z_A, mu_A, log_var_A = self.encode(real_A, 'A')
        
        # Tái tạo MRI từ không gian tiềm ẩn
        recon_A = self.decode(content_A, 'A')
        
        # Chuyển đổi MRI sang CT qua không gian tiềm ẩn chung
        fake_B = self.decode(content_A, 'B')
        
        result = {
            'real_A': real_A,
            'recon_A': recon_A,
            'fake_B': fake_B,
            'mu_A': mu_A,
            'log_var_A': log_var_A,
            'real_B': real_B
        }
        
        # Nếu có ảnh CT thật, thực hiện cycle consistency
        if real_B is not None:
            # Mã hóa CT
            content_B, z_B, mu_B, log_var_B = self.encode(real_B, 'B')
            
            # Tái tạo CT từ không gian tiềm ẩn
            recon_B = self.decode(content_B, 'B')
            
            # Chuyển đổi CT sang MRI qua không gian tiềm ẩn chung
            fake_A = self.decode(content_B, 'A')
            
            # Thêm thông tin vào kết quả
            result.update({
                'fake_A': fake_A,
                'recon_B': recon_B,
                'mu_B': mu_B,
                'log_var_B': log_var_B
            })
        
        return result
    
    def discriminate(self, image, domain='A'):
        """
        Phân biệt ảnh thật/giả trong một miền
        Args:
            image (tensor): Ảnh cần phân biệt
            domain (str): Miền của ảnh ('A' cho MRI, 'B' cho CT)
        """
        if domain == 'A':
            return self.disc_A(image)
        else:
            return self.disc_B(image) 