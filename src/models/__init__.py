"""
Module chứa các mô hình học sâu.

Bao gồm các kiến trúc mạng nơ-ron như CycleGAN, các thành phần và hàm tính loss.
"""

from .cycle_gan import (
    CycleGANModel, 
    Generator, 
    Discriminator, 
    ResidualBlock, 
    gan_loss, 
    cycle_consistency_loss, 
    identity_loss
) 