#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import tất cả các mô hình
from src.models import AVAILABLE_MODELS
from src.models.cycle_gan import gan_loss, cycle_consistency_loss, identity_loss
from src.data_processing.dataset import MRIToCTDataModule
from src.utils.visualization import save_images, visualize_results
from src.utils.metrics import calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình cho chuyển đổi MRI sang CT')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Đường dẫn đến file cấu hình')
    parser.add_argument('--resume', type=str, default=None,
                        help='Đường dẫn đến checkpoint để tiếp tục huấn luyện')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ModelTrainer:
    """Lớp huấn luyện chung cho tất cả các loại mô hình"""
    
    def __init__(self, config, resume_path=None):
        """
        Args:
            config (dict): Cấu hình huấn luyện
            resume_path (str, optional): Đường dẫn đến checkpoint để tiếp tục huấn luyện
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config['model']['type'].lower()
        
        # Kiểm tra mô hình có được hỗ trợ không
        if self.model_type not in AVAILABLE_MODELS:
            raise ValueError(f"Loại mô hình không hợp lệ: {self.model_type}. "
                            f"Các loại mô hình được hỗ trợ: {', '.join(AVAILABLE_MODELS.keys())}")
        
        # Cấu hình đường dẫn
        self.output_dir = Path(config['training']['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.results_dir = self.output_dir / 'results'
        self.log_dir = self.output_dir / 'logs'
        
        # Tạo thư mục đầu ra
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Khởi tạo data module
        self.data_module = MRIToCTDataModule(
            data_dir=config['data']['processed_dir'],
            batch_size=config['training']['batch_size'],
            slice_axis=config['data']['slice_axis'],
            slice_range=config['data']['slice_range'],
            num_workers=config['training']['num_workers'],
            paired=config['data']['paired']
        )
        
        # Khởi tạo dataloaders
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        # Khởi tạo mô hình dựa trên loại
        self._init_model()
        
        # Khởi tạo optimizers
        self._init_optimizers()
        
        # Khởi tạo learning rate schedulers
        self._init_schedulers()
        
        # Thiết lập trạng thái huấn luyện
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.best_val_loss = float('inf')
        
        # Tải checkpoint nếu có
        if resume_path:
            self._load_checkpoint(resume_path)
            
    def _init_model(self):
        """Khởi tạo mô hình dựa trên loại đã chọn"""
        model_class = AVAILABLE_MODELS[self.model_type]
        
        # Các tham số chung cho tất cả các mô hình
        common_params = {
            'input_nc': self.config['model']['input_nc'],
            'output_nc': self.config['model']['output_nc'],
        }
        
        # Tham số riêng cho từng loại mô hình
        if self.model_type == 'cyclegan':
            self.model = model_class(
                **common_params,
                ngf=self.config['model']['ngf'],
                ndf=self.config['model']['ndf']
            ).to(self.device)
        elif self.model_type == 'unet':
            bilinear = self.config['model'].get('unet', {}).get('bilinear', True)
            self.model = model_class(
                **common_params,
                ngf=self.config['model']['ngf'],
                bilinear=bilinear
            ).to(self.device)
        elif self.model_type == 'pix2pix':
            use_dropout = self.config['model'].get('pix2pix', {}).get('use_dropout', True)
            norm_layer_name = self.config['model'].get('pix2pix', {}).get('norm_layer', 'instance')
            
            # Chọn lớp normalization
            if norm_layer_name == 'instance':
                norm_layer = nn.InstanceNorm2d
            else:
                norm_layer = nn.BatchNorm2d
                
            self.model = model_class(
                **common_params,
                ngf=self.config['model']['ngf'],
                ndf=self.config['model']['ndf'],
                norm_layer=norm_layer
            ).to(self.device)
        elif self.model_type == 'attentiongan':
            self.model = model_class(
                **common_params,
                ngf=self.config['model']['ngf'],
                ndf=self.config['model']['ndf'],
                n_blocks=self.config['model']['n_blocks']
            ).to(self.device)
        elif self.model_type == 'unit':
            latent_dim = self.config['model'].get('unit', {}).get('latent_dim', 512)
            n_encoder_blocks = self.config['model'].get('unit', {}).get('n_encoder_blocks', 3)
            
            self.model = model_class(
                **common_params,
                nef=self.config['model']['ngf'],
                ndf=self.config['model']['ndf'],
                n_blocks=n_encoder_blocks,
                latent_dim=latent_dim
            ).to(self.device)
    
    def _init_optimizers(self):
        """Khởi tạo các optimizer cho mô hình"""
        if self.model_type == 'cyclegan':
            self.optimizer_G = optim.Adam(
                list(self.model.netG_A.parameters()) + list(self.model.netG_B.parameters()),
                lr=self.config['training']['lr'],
                betas=(self.config['training']['beta1'], 0.999)
            )
            
            self.optimizer_D = optim.Adam(
                list(self.model.netD_A.parameters()) + list(self.model.netD_B.parameters()),
                lr=self.config['training']['lr'],
                betas=(self.config['training']['beta1'], 0.999)
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['lr'],
                betas=(self.config['training']['beta1'], 0.999)
            )
    
    def _init_schedulers(self):
        """Khởi tạo các learning rate scheduler cho mô hình"""
        if self.model_type == 'cyclegan':
            self.scheduler_G = optim.lr_scheduler.StepLR(
                self.optimizer_G,
                step_size=self.config['training']['lr_decay_epochs'],
                gamma=self.config['training']['lr_decay_gamma']
            )
            
            self.scheduler_D = optim.lr_scheduler.StepLR(
                self.optimizer_D,
                step_size=self.config['training']['lr_decay_epochs'],
                gamma=self.config['training']['lr_decay_gamma']
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['lr_decay_epochs'],
                gamma=self.config['training']['lr_decay_gamma']
            )
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint từ đường dẫn"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model'])
        
        # Load optimizer states
        if self.model_type == 'cyclegan':
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scheduler states
        if self.model_type == 'cyclegan':
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])
        else:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.current_epoch = checkpoint['epoch']
        self.current_iter = checkpoint['iter']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Lưu checkpoint mô hình"""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        # Tạo state dict tùy theo loại mô hình
        if self.model_type == 'cyclegan':
            state_dict = {
                'epoch': epoch,
                'iter': self.current_iter,
                'netG_A': self.model.netG_A.state_dict(),
                'netG_B': self.model.netG_B.state_dict(),
                'netD_A': self.model.netD_A.state_dict(),
                'netD_B': self.model.netD_B.state_dict(),
                'optimizer_G': self.optimizer_G.state_dict(),
                'optimizer_D': self.optimizer_D.state_dict(),
                'scheduler_G': self.scheduler_G.state_dict(),
                'scheduler_D': self.scheduler_D.state_dict(),
                'best_val_loss': self.best_val_loss
            }
        else:
            state_dict = {
                'epoch': epoch,
                'iter': self.current_iter,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss
            }
        
        # Lưu checkpoint
        torch.save(state_dict, checkpoint_path)
        print(f"Đã lưu checkpoint tại {checkpoint_path}")
        
        # Lưu checkpoint tốt nhất nếu cần
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(state_dict, best_path)
            print(f"Đã lưu checkpoint tốt nhất tại {best_path}")
    
    def _set_train_mode(self):
        """Đặt mô hình ở chế độ huấn luyện"""
        self.model.train()
    
    def _set_eval_mode(self):
        """Đặt mô hình ở chế độ đánh giá"""
        self.model.eval()
    
    def train_step(self, batch):
        """Thực hiện một bước huấn luyện"""
        # Lấy dữ liệu từ batch
        real_A = batch['mri'].to(self.device)
        real_B = batch['ct'].to(self.device) if 'ct' in batch else None
        
        # Khởi tạo các biến loss
        loss_G = torch.tensor(0.0, device=self.device)
        loss_D = torch.tensor(0.0, device=self.device)
        loss_cycle_A = torch.tensor(0.0, device=self.device)
        loss_cycle_B = torch.tensor(0.0, device=self.device)
        loss_idt_A = torch.tensor(0.0, device=self.device)
        loss_idt_B = torch.tensor(0.0, device=self.device)
        loss_G_B = torch.tensor(0.0, device=self.device)
        loss_D_B = torch.tensor(0.0, device=self.device)
        loss_D_real = torch.tensor(0.0, device=self.device)
        loss_D_fake = torch.tensor(0.0, device=self.device)
        
        # Xử lý dựa trên loại mô hình
        if self.model_type == 'cyclegan':
            # CycleGAN có 2 generator và 2 discriminator
            self.optimizer_G.zero_grad()
            
            # Forward
            outputs = self.model(real_A, real_B)
            fake_B = outputs['fake_B']
            
            # Identity loss
            if self.config['training'].get('lambda_identity', 0) > 0 and real_B is not None:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                idt_A = self.model.netG_A(real_B)
                loss_idt_A = identity_loss(real_B, idt_A, self.config['training']['lambda_identity'])
                
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                idt_B = self.model.netG_B(real_A)
                loss_idt_B = identity_loss(real_A, idt_B, self.config['training']['lambda_identity'])
            
            # GAN loss D_A(G_A(A))
            pred_fake = self.model.netD_A(fake_B)
            loss_G = gan_loss(pred_fake, True)
            
            # Cycle loss
            if real_B is not None:
                fake_A = outputs['fake_A']
                rec_A = outputs['rec_A']
                rec_B = outputs['rec_B']
                
                # Forward cycle loss || G_B(G_A(A)) - A||
                loss_cycle_A = cycle_consistency_loss(real_A, rec_A, self.config['training']['lambda_A'])
                
                # Backward cycle loss || G_A(G_B(B)) - B||
                loss_cycle_B = cycle_consistency_loss(real_B, rec_B, self.config['training']['lambda_B'])
                
                # GAN loss D_B(G_B(B))
                pred_fake_B = self.model.netD_B(fake_A)
                loss_G_B = gan_loss(pred_fake_B, True)
            
            # Tổng hợp tất cả generator losses
            loss_G = loss_G + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            
            # Backward và optimize
            loss_G.backward()
            self.optimizer_G.step()
            
            # Huấn luyện Discriminator
            self.optimizer_D.zero_grad()
            
            # Discriminator A
            # Thật
            pred_real = self.model.netD_A(real_B) if real_B is not None else None
            loss_D_real = gan_loss(pred_real, True) if pred_real is not None else 0
            
            # Giả
            pred_fake = self.model.netD_A(fake_B.detach())
            loss_D_fake = gan_loss(pred_fake, False)
            
            # Kết hợp loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5 if real_B is not None else loss_D_fake
            
            # Discriminator B
            if real_B is not None:
                # Thật
                pred_real = self.model.netD_B(real_A)
                loss_D_B_real = gan_loss(pred_real, True)
                
                # Giả
                pred_fake = self.model.netD_B(fake_A.detach())
                loss_D_B_fake = gan_loss(pred_fake, False)
                
                # Kết hợp
                loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            
            # Tổng hợp tất cả discriminator losses
            loss_D = loss_D + loss_D_B if real_B is not None else loss_D
            
            # Backward và optimize
            loss_D.backward()
            self.optimizer_D.step()
            
        elif self.model_type in ['unet', 'pix2pix', 'attentiongan', 'unit']:
            # Các mô hình khác có quy trình huấn luyện đơn giản hơn
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(real_A, real_B)
            fake_B = outputs['fake_B']
            
            # Loss cho generator
            if self.model_type == 'unet':
                # UNet chỉ sử dụng L1 loss
                loss_G = nn.L1Loss()(fake_B, real_B) if real_B is not None else 0
            elif self.model_type == 'pix2pix':
                # Pix2Pix sử dụng kết hợp L1 và GAN loss
                loss_l1 = nn.L1Loss()(fake_B, real_B) * self.config['training'].get('lambda_L1', 100)
                loss_gan = self.model.discriminate(real_A, fake_B)
                loss_G = loss_l1 + loss_gan
            elif self.model_type == 'attentiongan':
                # AttentionGAN tương tự như CycleGAN
                fake_A = outputs['fake_A'] if 'fake_A' in outputs else None
                rec_A = outputs['rec_A'] if 'rec_A' in outputs else None
                rec_B = outputs['rec_B'] if 'rec_B' in outputs else None
                
                # Tính toán tất cả các loss cần thiết
                if real_B is not None:
                    loss_cycle_A = cycle_consistency_loss(real_A, rec_A, self.config['training']['lambda_A'])
                    loss_cycle_B = cycle_consistency_loss(real_B, rec_B, self.config['training']['lambda_B'])
                    loss_G = loss_G + loss_cycle_A + loss_cycle_B
            elif self.model_type == 'unit':
                # UNIT sử dụng kết hợp nhiều loss
                loss_recon = 0
                if real_B is not None:
                    loss_recon = nn.L1Loss()(fake_B, real_B) * self.config['training'].get('lambda_recon', 10)
                
                loss_gan = 0
                if 'disc_loss' in outputs:
                    loss_gan = outputs['disc_loss']
                
                loss_kl = 0
                if 'kl_loss' in outputs:
                    loss_kl = outputs['kl_loss'] * self.config['training'].get('lambda_kl', 0.1)
                
                loss_G = loss_recon + loss_gan + loss_kl
            
            # Optimize generator
            loss_G.backward()
            self.optimizer.step()
            
            # Huấn luyện discriminator cho các mô hình cần
            if self.model_type in ['pix2pix', 'attentiongan', 'unit']:
                self.optimizer.zero_grad()
                
                # Discriminator loss
                if real_B is not None:
                    if self.model_type == 'pix2pix':
                        loss_D_real = gan_loss(self.model.discriminator(real_A, real_B), True)
                        loss_D_fake = gan_loss(self.model.discriminator(real_A, fake_B.detach()), False)
                    elif self.model_type == 'attentiongan':
                        loss_D_real = gan_loss(self.model.netD_A(real_B), True)
                        loss_D_fake = gan_loss(self.model.netD_A(fake_B.detach()), False)
                    elif self.model_type == 'unit':
                        loss_D_real = gan_loss(self.model.discriminate(real_B, domain='B'), True)
                        loss_D_fake = gan_loss(self.model.discriminate(fake_B.detach(), domain='B'), False)
                    
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                    loss_D.backward()
                    self.optimizer.step()
        
        # Trả về losses để logging
        losses = {
            'loss_G': loss_G.item(),
            'loss_cycle_A': loss_cycle_A.item() if isinstance(loss_cycle_A, torch.Tensor) and real_B is not None else 0,
            'loss_cycle_B': loss_cycle_B.item() if isinstance(loss_cycle_B, torch.Tensor) and real_B is not None else 0,
            'loss_idt_A': loss_idt_A.item() if isinstance(loss_idt_A, torch.Tensor) else 0,
            'loss_idt_B': loss_idt_B.item() if isinstance(loss_idt_B, torch.Tensor) else 0,
            'loss_D': loss_D.item() if isinstance(loss_D, torch.Tensor) else 0,
            'loss_D_real': loss_D_real.item() if isinstance(loss_D_real, torch.Tensor) else 0,
            'loss_D_fake': loss_D_fake.item() if isinstance(loss_D_fake, torch.Tensor) else 0
        }
        
        return losses, outputs
    
    def validate(self):
        """Đánh giá mô hình trên tập validation"""
        self._set_eval_mode()
        val_losses = {
            'loss_G': 0, 'loss_cycle_A': 0, 'loss_cycle_B': 0,
            'loss_idt_A': 0, 'loss_idt_B': 0,
            'loss_D': 0, 'loss_D_real': 0, 'loss_D_fake': 0
        }
        
        metrics = {'mae': 0, 'psnr': 0, 'ssim': 0}
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                real_A = batch['mri'].to(self.device)
                real_B = batch['ct'].to(self.device) if 'ct' in batch else None
                
                # Forward pass
                outputs = self.model(real_A, real_B)
                fake_B = outputs['fake_B']
                
                # Tính loss (giả lập train_step ở chế độ evaluation)
                # Note: Code này có thể được refactor để tránh lặp lại logic từ train_step
                
                # Tính metrics nếu có ground truth CT
                if real_B is not None:
                    batch_metrics = calculate_metrics(fake_B.cpu().numpy(), real_B.cpu().numpy())
                    metrics['mae'] += batch_metrics['mae'] * real_A.size(0)
                    metrics['psnr'] += batch_metrics['psnr'] * real_A.size(0)
                    metrics['ssim'] += batch_metrics['ssim'] * real_A.size(0)
                    
                    num_samples += real_A.size(0)
        
        # Tính trung bình metrics
        if num_samples > 0:
            metrics['mae'] /= num_samples
            metrics['psnr'] /= num_samples
            metrics['ssim'] /= num_samples
        
        # Lưu một số ảnh kết quả
        if real_B is not None:
            save_path = self.results_dir / f"val_epoch_{self.current_epoch}.png"
            visualize_results(real_A.cpu(), fake_B.cpu(), real_B.cpu(), save_path)
        
        self._set_train_mode()
        return val_losses, metrics
    
    def train(self):
        """Huấn luyện mô hình"""
        num_epochs = self.config['training']['num_epochs']
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            self._set_train_mode()
            train_losses = {
                'loss_G': 0, 'loss_cycle_A': 0, 'loss_cycle_B': 0,
                'loss_idt_A': 0, 'loss_idt_B': 0,
                'loss_D': 0, 'loss_D_real': 0, 'loss_D_fake': 0
            }
            
            # Tạo progress bar
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs-1}")
            for i, batch in enumerate(pbar):
                losses, outputs = self.train_step(batch)
                
                # Cập nhật training losses
                for key in train_losses:
                    train_losses[key] += losses[key]
                
                # Update progress bar
                pbar.set_postfix({
                    'loss_G': f"{losses['loss_G']:.4f}",
                    'loss_D': f"{losses['loss_D']:.4f}"
                })
                
                # Log to tensorboard (every n iterations)
                log_interval = self.config['training']['log_interval']
                if self.current_iter % log_interval == 0:
                    # Log losses
                    for key, value in losses.items():
                        self.writer.add_scalar(f'train/{key}', value, self.current_iter)
                    
                    # Log images
                    if self.current_iter % (log_interval * 5) == 0:
                        real_A = batch['mri'][:4]  # Chỉ lấy 4 ảnh đầu tiên
                        fake_B = outputs['fake_B'][:4].cpu().detach()
                        
                        self.writer.add_images('train/real_MRI', real_A, self.current_iter)
                        self.writer.add_images('train/fake_CT', fake_B, self.current_iter)
                        
                        if 'ct' in batch:
                            real_B = batch['ct'][:4]
                            fake_A = outputs['fake_A'][:4].cpu().detach()
                            rec_A = outputs['rec_A'][:4].cpu().detach()
                            rec_B = outputs['rec_B'][:4].cpu().detach()
                            
                            self.writer.add_images('train/real_CT', real_B, self.current_iter)
                            self.writer.add_images('train/fake_MRI', fake_A, self.current_iter)
                            self.writer.add_images('train/rec_MRI', rec_A, self.current_iter)
                            self.writer.add_images('train/rec_CT', rec_B, self.current_iter)
                
                self.current_iter += 1
            
            # Tính trung bình loss cho epoch
            num_batches = len(self.train_loader)
            for key in train_losses:
                train_losses[key] /= num_batches
            
            # Validation
            val_losses, val_metrics = self.validate()
            
            # Log validation metrics
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
            
            # Update learning rates
            if self.model_type == 'cyclegan':
                self.scheduler_G.step()
                self.scheduler_D.step()
            else:
                self.scheduler.step()
            
            # Log learning rates
            if self.model_type == 'cyclegan':
                self.writer.add_scalar('lr/G', self.scheduler_G.get_last_lr()[0], self.current_epoch)
                self.writer.add_scalar('lr/D', self.scheduler_D.get_last_lr()[0], self.current_epoch)
            else:
                self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.current_epoch)
            
            # Save model checkpoint
            is_best = val_metrics['mae'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['mae']
            
            self._save_checkpoint(epoch, is_best)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            print(f"  Train Loss G: {train_losses['loss_G']:.4f}, D: {train_losses['loss_D']:.4f}")
            if 'mae' in val_metrics:
                print(f"  Val   MAE: {val_metrics['mae']:.4f}, PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
        
        self.writer.close()
        print("Training completed!")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    trainer = ModelTrainer(config, args.resume)
    trainer.train()

if __name__ == '__main__':
    main() 