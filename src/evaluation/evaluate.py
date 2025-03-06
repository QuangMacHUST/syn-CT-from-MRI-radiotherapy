#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import h5py
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import AVAILABLE_MODELS
from src.utils.metrics import calculate_metrics, calculate_tissue_metrics, calculate_hu_metrics
from src.utils.visualization import visualize_results, save_images
from src.data_processing.dataset import MRIToCTDataModule

def parse_args():
    parser = argparse.ArgumentParser(description='Đánh giá mô hình chuyển đổi MRI sang CT')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Đường dẫn đến file cấu hình')
    parser.add_argument('--model', type=str, required=True,
                        help='Đường dẫn đến checkpoint mô hình')
    parser.add_argument('--model_type', type=str, default='cyclegan',
                        help='Loại mô hình (cyclegan, unet, pix2pix, attentiongan, unit)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Đường dẫn đến thư mục chứa dữ liệu test')
    parser.add_argument('--output_dir', type=str, default='data/output/evaluation',
                        help='Đường dẫn đến thư mục lưu kết quả đánh giá')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Thiết bị tính toán (cuda hoặc cpu)')
    return parser.parse_args()

def load_config(config_path):
    """Load cấu hình từ file YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def segment_tissues(ct_image):
    """
    Phân đoạn ảnh CT thành các loại mô dựa trên giá trị Hounsfield Units (HU)
    
    Args:
        ct_image (numpy.ndarray): Ảnh CT đơn vị HU
    
    Returns:
        dict: Dictionary chứa mask cho từng loại mô
    """
    # Đảm bảo ảnh là 2D
    if ct_image.ndim > 2:
        ct_image = ct_image.squeeze()
    
    # Tạo các mask dựa trên khoảng HU
    air_mask = (ct_image <= -950)  # Không khí
    lung_mask = (ct_image > -950) & (ct_image <= -700)  # Phổi
    fat_mask = (ct_image > -700) & (ct_image <= -30)  # Mỡ
    soft_tissue_mask = (ct_image > -30) & (ct_image <= 100)  # Mô mềm
    bone_mask = (ct_image > 100) & (ct_image <= 1500)  # Xương
    metal_mask = (ct_image > 1500)  # Kim loại/implant
    
    # Tạo mask tổng hợp
    tissue_masks = {
        'air': air_mask,
        'lung': lung_mask,
        'fat': fat_mask,
        'soft_tissue': soft_tissue_mask,
        'bone': bone_mask,
        'metal': metal_mask,
        'body': ~air_mask  # Toàn bộ cơ thể (không bao gồm không khí)
    }
    
    return tissue_masks

def load_model(model_path, model_type, device, config):
    """
    Tải mô hình từ checkpoint
    
    Args:
        model_path (str): Đường dẫn đến checkpoint
        model_type (str): Loại mô hình (cyclegan, unet, pix2pix, attentiongan, unit)
        device (torch.device): Thiết bị tính toán
        config (dict): Cấu hình mô hình
    
    Returns:
        nn.Module: Mô hình đã tải
    """
    # Kiểm tra xem loại mô hình có được hỗ trợ không
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"Loại mô hình không hợp lệ: {model_type}. "
                        f"Các loại mô hình được hỗ trợ: {', '.join(AVAILABLE_MODELS.keys())}")
    
    # Tải checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Khởi tạo mô hình dựa trên loại
    model_class = AVAILABLE_MODELS[model_type]
    
    # Các tham số chung cho tất cả các mô hình
    common_params = {
        'input_nc': config['model']['input_nc'],
        'output_nc': config['model']['output_nc'],
    }
    
    # Tham số riêng cho từng loại mô hình
    if model_type == 'cyclegan':
        model = model_class(
            **common_params,
            ngf=config['model']['ngf'],
            ndf=config['model']['ndf']
        ).to(device)
        # CycleGAN lưu các generator và discriminator riêng lẻ
        if 'netG_A' in checkpoint:
            model.netG_A.load_state_dict(checkpoint['netG_A'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
    elif model_type == 'unet':
        bilinear = config['model'].get('unet', {}).get('bilinear', True)
        model = model_class(
            **common_params,
            ngf=config['model']['ngf'],
            bilinear=bilinear
        ).to(device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    elif model_type == 'pix2pix':
        model = model_class(
            **common_params,
            ngf=config['model']['ngf'],
            ndf=config['model']['ndf']
        ).to(device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    elif model_type == 'attentiongan':
        model = model_class(
            **common_params,
            ngf=config['model']['ngf'],
            ndf=config['model']['ndf'],
            n_blocks=config['model']['n_blocks']
        ).to(device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    elif model_type == 'unit':
        latent_dim = config['model'].get('unit', {}).get('latent_dim', 512)
        n_blocks = config['model'].get('unit', {}).get('n_encoder_blocks', 3)
        model = model_class(
            **common_params,
            nef=config['model']['ngf'],
            ndf=config['model']['ndf'],
            n_blocks=n_blocks,
            latent_dim=latent_dim
        ).to(device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def evaluate_model(model, data_loader, device, config, output_dir, model_type):
    """
    Đánh giá hiệu quả của mô hình trên tập dữ liệu test
    
    Args:
        model (torch.nn.Module): Mô hình cần đánh giá
        data_loader (DataLoader): DataLoader chứa dữ liệu test
        device (torch.device): Thiết bị tính toán
        config (dict): Cấu hình đánh giá
        output_dir (str): Đường dẫn thư mục đầu ra
        model_type (str): Loại mô hình đang được đánh giá
    
    Returns:
        dict: Dictionary chứa kết quả đánh giá
    """
    model.eval()
    results = {
        'global_metrics': {
            'mae': [],
            'rmse': [],
            'psnr': [],
            'ssim': []
        },
        'tissue_metrics': {},
        'hu_range_metrics': {}
    }
    
    # Tạo thư mục kết quả hình ảnh
    image_dir = Path(output_dir) / 'images'
    image_dir.mkdir(exist_ok=True, parents=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Đánh giá")):
            # Lấy dữ liệu từ batch
            mri_images = batch['mri'].to(device)
            real_ct_images = batch['ct'].to(device) if 'ct' in batch else None
            
            # Nếu không có ảnh CT thật, bỏ qua
            if real_ct_images is None:
                continue
            
            # Chuyển đổi MRI sang CT dựa trên loại mô hình
            if model_type == 'cyclegan':
                output = model(mri_images)
                synth_ct_images = output['fake_B']
            elif model_type == 'unit':
                content = model.encode(mri_images, domain='A')
                synth_ct_images = model.decode(content, domain='B')
            else:  # unet, pix2pix, attentiongan
                output = model(mri_images)
                synth_ct_images = output['fake_B'] if isinstance(output, dict) else output
            
            # Chuyển tensor sang numpy
            mri_np = mri_images.cpu().numpy()
            synth_ct_np = synth_ct_images.cpu().numpy()
            real_ct_np = real_ct_images.cpu().numpy()
            
            # Thêm kết quả trực quan cho một số mẫu
            if i % 10 == 0:
                for b in range(min(mri_images.size(0), 3)):  # Chỉ lưu tối đa 3 ảnh mỗi batch
                    save_path = image_dir / f'sample_{i}_{b}.png'
                    visualize_results(
                        mri_np[b], 
                        synth_ct_np[b], 
                        real_ct_np[b],
                        save_path=save_path
                    )
            
            # Đánh giá từng ảnh trong batch
            for b in range(mri_images.size(0)):
                # Giả định dữ liệu CT là HU trong khoảng [-1000, 3000]
                # Chuyển đổi CT mô phỏng về thang HU nếu nó được chuẩn hóa
                if synth_ct_np[b].min() >= -1.1 and synth_ct_np[b].max() <= 1.1:
                    synth_ct_hu = (synth_ct_np[b] + 1) / 2 * 4000 - 1000  # Map [-1, 1] sang [-1000, 3000]
                elif synth_ct_np[b].min() >= 0 and synth_ct_np[b].max() <= 1:
                    synth_ct_hu = synth_ct_np[b] * 4000 - 1000  # Map [0, 1] sang [-1000, 3000]
                else:
                    synth_ct_hu = synth_ct_np[b]
                
                if real_ct_np[b].min() >= -1.1 and real_ct_np[b].max() <= 1.1:
                    real_ct_hu = (real_ct_np[b] + 1) / 2 * 4000 - 1000  # Map [-1, 1] sang [-1000, 3000]
                elif real_ct_np[b].min() >= 0 and real_ct_np[b].max() <= 1:
                    real_ct_hu = real_ct_np[b] * 4000 - 1000  # Map [0, 1] sang [-1000, 3000]
                else:
                    real_ct_hu = real_ct_np[b]
                
                # Tính metrics tổng thể
                metrics = calculate_metrics(synth_ct_hu, real_ct_hu)
                
                # Lưu kết quả
                for key, value in metrics.items():
                    if key in results['global_metrics']:
                        results['global_metrics'][key].append(value)
                
                # Phân đoạn các loại mô trên CT thật
                tissue_masks = segment_tissues(real_ct_hu)
                
                # Tính metrics cho từng loại mô
                tissue_metrics = calculate_tissue_metrics(synth_ct_hu, real_ct_hu, tissue_masks)
                
                # Lưu kết quả
                for key, value in tissue_metrics.items():
                    if key not in results['tissue_metrics']:
                        results['tissue_metrics'][key] = []
                    results['tissue_metrics'][key].append(value)
                
                # Tính metrics cho các khoảng HU
                hu_ranges = config['evaluation']['hu_ranges']
                hu_metrics = calculate_hu_metrics(synth_ct_hu, real_ct_hu, hu_ranges)
                
                # Lưu kết quả
                for key, value in hu_metrics.items():
                    if key not in results['hu_range_metrics']:
                        results['hu_range_metrics'][key] = []
                    results['hu_range_metrics'][key].append(value)
    
    # Tính trung bình của các metrics
    for category in results:
        if isinstance(results[category], dict):
            for key in results[category]:
                values = results[category][key]
                if values:  # Kiểm tra nếu list không rỗng
                    results[category][key] = np.mean(values)
                else:
                    results[category][key] = 0
    
    return results

def generate_evaluation_report(results, output_dir, model_type):
    """
    Tạo báo cáo đánh giá
    
    Args:
        results (dict): Kết quả đánh giá
        output_dir (str): Đường dẫn thư mục đầu ra
        model_type (str): Loại mô hình đang được đánh giá
    """
    # Tạo thư mục kết quả
    output_dir = Path(output_dir) / model_type
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Lưu metrics dưới dạng CSV
    metrics_df = pd.DataFrame()
    
    # Thêm metrics tổng thể
    for key, value in results['global_metrics'].items():
        metrics_df.loc['Global', key] = value
    
    # Thêm metrics cho từng loại mô
    tissue_metrics = {}
    for key, value in results['tissue_metrics'].items():
        tissue_name, metric_name = key.split('_', 1)
        if tissue_name not in tissue_metrics:
            tissue_metrics[tissue_name] = {}
        tissue_metrics[tissue_name][metric_name] = value
    
    for tissue_name, metrics in tissue_metrics.items():
        for metric_name, value in metrics.items():
            metrics_df.loc[f'Tissue_{tissue_name}', metric_name] = value
    
    # Thêm metrics cho các khoảng HU
    hu_metrics = {}
    for key, value in results['hu_range_metrics'].items():
        hu_metrics[key] = value
    
    for range_name, value in hu_metrics.items():
        metrics_df.loc[range_name, 'mae'] = value
    
    # Lưu DataFrame thành CSV
    metrics_csv_path = output_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_csv_path)
    print(f"Đã lưu metrics vào: {metrics_csv_path}")
    
    # Tạo biểu đồ cho metrics tổng thể
    plt.figure(figsize=(10, 6))
    plt.bar(results['global_metrics'].keys(), results['global_metrics'].values())
    plt.title(f'Global Metrics - {model_type.upper()}')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(output_dir / 'global_metrics.png')
    plt.close()
    
    # Tạo biểu đồ cho MAE theo loại mô
    tissue_mae = {k: v for k, v in results['tissue_metrics'].items() if k.endswith('_mae')}
    if tissue_mae:
        plt.figure(figsize=(12, 6))
        plt.bar(tissue_mae.keys(), tissue_mae.values())
        plt.title(f'MAE by Tissue Type - {model_type.upper()}')
        plt.ylabel('MAE (HU)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'tissue_mae.png')
        plt.close()
    
    # Tạo biểu đồ cho MAE theo khoảng HU
    if results['hu_range_metrics']:
        plt.figure(figsize=(12, 6))
        plt.bar(results['hu_range_metrics'].keys(), results['hu_range_metrics'].values())
        plt.title(f'MAE by HU Range - {model_type.upper()}')
        plt.ylabel('MAE (HU)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'hu_range_mae.png')
        plt.close()
    
    print(f"Đã tạo báo cáo đánh giá tại: {output_dir}")

def main():
    """Hàm chính đánh giá mô hình"""
    # Parse arguments
    args = parse_args()
    
    # Load cấu hình
    config = load_config(args.config)
    
    # Thiết lập device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    # Tạo data loader
    data_module = MRIToCTDataModule(
        data_dir=args.data_dir,
        batch_size=config['evaluation']['test_batch_size'],
        slice_axis=config['data']['slice_axis'],
        paired=True  # Cần dữ liệu ghép cặp để đánh giá
    )
    test_loader = data_module.test_dataloader()
    
    # Tải mô hình
    model = load_model(args.model, args.model_type, device, config)
    
    print(f"Đã tải mô hình {args.model_type} từ: {args.model}")
    
    # Đánh giá mô hình
    print(f"Bắt đầu đánh giá mô hình {args.model_type}...")
    results = evaluate_model(model, test_loader, device, config, args.output_dir, args.model_type)
    
    # Tạo báo cáo
    generate_evaluation_report(results, args.output_dir, args.model_type)
    
    print("Đánh giá hoàn tất!")

if __name__ == '__main__':
    main() 