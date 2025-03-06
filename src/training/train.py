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

from src.models.cycle_gan import CycleGANModel, gan_loss, cycle_consistency_loss, identity_loss
from src.data_processing.dataset import MRIToCTDataModule
from src.utils.visualization import save_images, visualize_results
from src.utils.metrics import calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình CycleGAN cho chuyển đổi MRI sang CT')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Đường dẫn đến file cấu hình')
    parser.add_argument('--resume', type=str, default=None,
                        help='Đường dẫn đến checkpoint để tiếp tục huấn luyện')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class CycleGANTrainer:
    """Lớp huấn luyện mô hình CycleGAN"""
    
    def __init__(self, config, resume_path=None):
        """
        Args:
            config (dict): Cấu hình huấn luyện
            resume_path (str, optional): Đường dẫn đến checkpoint để tiếp tục huấn luyện
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        # Khởi tạo mô hình
        self.model = CycleGANModel(
            input_nc=config['model']['input_nc'],
            output_nc=config['model']['output_nc'],
            ngf=config['model']['ngf'],
            ndf=config['model']['ndf']
        ).to(self.device)
        
        # Khởi tạo optimizers
        self.optimizer_G = optim.Adam(
            list(self.model.netG_A.parameters()) + list(self.model.netG_B.parameters()),
            lr=config['training']['lr'],
            betas=(config['training']['beta1'], 0.999)
        )
        
        self.optimizer_D = optim.Adam(
            list(self.model.netD_A.parameters()) + list(self.model.netD_B.parameters()),
            lr=config['training']['lr'],
            betas=(config['training']['beta1'], 0.999)
        )
        
        # Khởi tạo learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G,
            step_size=config['training']['lr_decay_epochs'],
            gamma=config['training']['lr_decay_gamma']
        )
        
        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D,
            step_size=config['training']['lr_decay_epochs'],
            gamma=config['training']['lr_decay_gamma']
        )
        
        # Hyperparameters
        self.lambda_A = config['training']['lambda_A']  # weight for cycle loss A -> B -> A
        self.lambda_B = config['training']['lambda_B']  # weight for cycle loss B -> A -> B
        self.lambda_identity = config['training']['lambda_identity']  # weight for identity loss
        
        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.best_val_loss = float('inf')
        
        # Load checkpoint nếu có
        if resume_path is not None:
            self._load_checkpoint(resume_path)
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint từ đường dẫn"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.netG_A.load_state_dict(checkpoint['netG_A'])
        self.model.netG_B.load_state_dict(checkpoint['netG_B'])
        self.model.netD_A.load_state_dict(checkpoint['netD_A'])
        self.model.netD_B.load_state_dict(checkpoint['netD_B'])
        
        # Load optimizer states
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        
        # Load scheduler states
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.current_epoch = checkpoint['epoch']
        self.current_iter = checkpoint['iter']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Lưu checkpoint hiện tại"""
        checkpoint = {
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
        
        # Lưu checkpoint thông thường
        filename = f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
        # Lưu checkpoint tốt nhất
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            print(f"Saved best model checkpoint at epoch {epoch}")
    
    def _set_train_mode(self):
        """Đặt mô hình ở chế độ huấn luyện"""
        self.model.netG_A.train()
        self.model.netG_B.train()
        self.model.netD_A.train()
        self.model.netD_B.train()
    
    def _set_eval_mode(self):
        """Đặt mô hình ở chế độ đánh giá"""
        self.model.netG_A.eval()
        self.model.netG_B.eval()
        self.model.netD_A.eval()
        self.model.netD_B.eval()
    
    def train_step(self, batch):
        """Thực hiện một bước huấn luyện"""
        real_A = batch['mri'].to(self.device)
        real_B = batch['ct'].to(self.device) if 'ct' in batch else None
        
        # -----------------
        # Huấn luyện Generator
        # -----------------
        self.optimizer_G.zero_grad()
        
        # Forward
        outputs = self.model(real_A, real_B)
        fake_B = outputs['fake_B']
        
        # Identity loss
        if self.lambda_identity > 0 and real_B is not None:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = self.model.netG_A(real_B)
            loss_idt_A = identity_loss(real_B, idt_A, self.lambda_identity)
            
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = self.model.netG_B(real_A)
            loss_idt_B = identity_loss(real_A, idt_B, self.lambda_identity)
        else:
            loss_idt_A = 0
            loss_idt_B = 0
        
        # GAN loss D_A(G_A(A))
        pred_fake = self.model.netD_A(fake_B)
        loss_G_A = gan_loss(pred_fake, True)
        
        # Cycle loss
        if real_B is not None:
            fake_A = outputs['fake_A']
            rec_A = outputs['rec_A']
            rec_B = outputs['rec_B']
            
            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_cycle_A = cycle_consistency_loss(real_A, rec_A, self.lambda_A)
            
            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_cycle_B = cycle_consistency_loss(real_B, rec_B, self.lambda_B)
            
            # GAN loss D_B(G_B(B))
            pred_fake = self.model.netD_B(fake_A)
            loss_G_B = gan_loss(pred_fake, True)
        else:
            loss_cycle_A = 0
            loss_cycle_B = 0
            loss_G_B = 0
        
        # Tổng hợp tất cả generator losses
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        
        # Backward và optimize
        loss_G.backward()
        self.optimizer_G.step()
        
        # -----------------
        # Huấn luyện Discriminator
        # -----------------
        self.optimizer_D.zero_grad()
        
        # Discriminator A
        # Thật
        pred_real = self.model.netD_A(real_B) if real_B is not None else None
        loss_D_real = gan_loss(pred_real, True) if pred_real is not None else 0
        
        # Giả (cập nhật lại để không tính gradient cho generators)
        fake_B = self.model.netG_A(real_A).detach()
        pred_fake = self.model.netD_A(fake_B)
        loss_D_fake = gan_loss(pred_fake, False)
        
        # Kết hợp loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        
        # Discriminator B
        if real_B is not None:
            # Thật
            pred_real = self.model.netD_B(real_A)
            loss_D_real = gan_loss(pred_real, True)
            
            # Giả
            fake_A = self.model.netG_B(real_B).detach()
            pred_fake = self.model.netD_B(fake_A)
            loss_D_fake = gan_loss(pred_fake, False)
            
            # Kết hợp loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        else:
            loss_D_B = 0
        
        # Tổng hợp tất cả discriminator losses
        loss_D = loss_D_A + loss_D_B
        
        # Backward và optimize
        loss_D.backward()
        self.optimizer_D.step()
        
        # Trả về losses để logging
        losses = {
            'loss_G': loss_G.item(),
            'loss_G_A': loss_G_A.item(),
            'loss_G_B': loss_G_B.item() if real_B is not None else 0,
            'loss_cycle_A': loss_cycle_A.item() if real_B is not None else 0,
            'loss_cycle_B': loss_cycle_B.item() if real_B is not None else 0,
            'loss_idt_A': loss_idt_A.item() if isinstance(loss_idt_A, torch.Tensor) else 0,
            'loss_idt_B': loss_idt_B.item() if isinstance(loss_idt_B, torch.Tensor) else 0,
            'loss_D': loss_D.item(),
            'loss_D_A': loss_D_A.item(),
            'loss_D_B': loss_D_B.item() if real_B is not None else 0
        }
        
        return losses, outputs
    
    def validate(self):
        """Đánh giá mô hình trên tập validation"""
        self._set_eval_mode()
        val_losses = {
            'loss_G': 0, 'loss_G_A': 0, 'loss_G_B': 0,
            'loss_cycle_A': 0, 'loss_cycle_B': 0,
            'loss_idt_A': 0, 'loss_idt_B': 0,
            'loss_D': 0, 'loss_D_A': 0, 'loss_D_B': 0
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
                'loss_G': 0, 'loss_G_A': 0, 'loss_G_B': 0,
                'loss_cycle_A': 0, 'loss_cycle_B': 0,
                'loss_idt_A': 0, 'loss_idt_B': 0,
                'loss_D': 0, 'loss_D_A': 0, 'loss_D_B': 0
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
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Log learning rates
            self.writer.add_scalar('lr/G', self.scheduler_G.get_last_lr()[0], self.current_epoch)
            self.writer.add_scalar('lr/D', self.scheduler_D.get_last_lr()[0], self.current_epoch)
            
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
    
    trainer = CycleGANTrainer(config, args.resume)
    trainer.train()

if __name__ == '__main__':
    main() 