"""
Module chứa các mô hình học sâu.

Bao gồm các kiến trúc mạng nơ-ron như CycleGAN, UNet, Pix2Pix, AttentionGAN, UNIT và các thành phần và hàm tính loss.
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

from .unet import (
    UNetModel,
    UNet,
    DoubleConv,
    Down,
    Up,
    OutConv
)

from .pix2pix import (
    Pix2PixModel,
    UNetGenerator,
    PatchGANDiscriminator
)

from .attention_gan import (
    AttentionGANModel,
    AttentionGenerator,
    AttentionModule
)

from .unit import (
    UNITModel,
    Encoder,
    Decoder,
    VAEEncoder,
    GaussianNoiseLayer
)

AVAILABLE_MODELS = {
    'cyclegan': CycleGANModel,
    'unet': UNetModel,
    'pix2pix': Pix2PixModel,
    'attentiongan': AttentionGANModel,
    'unit': UNITModel
} 