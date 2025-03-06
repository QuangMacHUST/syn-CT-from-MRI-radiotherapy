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

from src.models import CycleGANModel
from src.utils.metrics import calculate_metrics, calculate_tissue_metrics, calculate_hu_metrics
from src.utils.visualization import visualize_results, save_images
from src.data_processing.dataset import MRIToCTDataModule

def parse_args():
    parser = argparse.ArgumentParser(description='Đánh giá mô hình chuyển đổi MRI sang CT')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Đường dẫn đến file cấu hình')
    parser.add_argument('--model', type=str, required=True,
                        help='Đường dẫn đến checkpoint mô hình')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Đường dẫn đến thư mục chứa dữ liệu test')
    parser.add_argument('--output_dir', type=str, default='data/output/evaluation',
                        help='Đường dẫn đến thư mục lưu kết quả đánh giá')
    parser.add_argument('--device', type=str, default=None,
                        help='Thiết bị để thực hiện đánh giá (cuda/cpu)')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def segment_tissues(ct_image):
    """
    Phân đoạn ảnh CT thành các loại mô khác nhau
    
    Args:
        ct_image (numpy.ndarray): Ảnh CT với giá trị HU
    
    Returns:
        dict: Dictionary chứa các mask cho từng loại mô
    """
    # Giả định giá trị đã ở dạng HU
    masks = {}
    
    # Không khí
    masks['air'] = (ct_image < -900)
    
    # Phổi
    masks['lung'] = (ct_image >= -900) & (ct_image < -500)
    
    # Mỡ
    masks['fat'] = (ct_image >= -500) & (ct_image < -50)
    
    # Mô mềm
    masks['soft_tissue'] = (ct_image >= -50) & (ct_image < 100)
    
    # Xương
    masks['bone'] = (ct_image >= 100)
    
    return masks

def evaluate_model(model, data_loader, device, config, output_dir):
    """
    Đánh giá mô hình trên tập dữ liệu test
    
    Args:
        model (CycleGANModel): Mô hình đã huấn luyện
        data_loader: DataLoader cho tập dữ liệu test
        device (torch.device): Thiết bị để thực hiện đánh giá
        config (dict): Cấu hình đánh giá
        output_dir (str): Thư mục đầu ra
    
    Returns:
        dict: Kết quả đánh giá
    """
    model.eval()
    
    # Danh sách metrics cần đánh giá
    metrics_list = config['evaluation']['metrics']
    hu_ranges = config['evaluation']['hu_ranges']
    
    all_metrics = {}
    for metric in metrics_list:
        all_metrics[metric] = []
    
    # Metrics cho các loại mô
    tissue_metrics = {
        'air_mae': [], 'lung_mae': [], 'fat_mae': [], 
        'soft_tissue_mae': [], 'bone_mae': []
    }
    
    # Metrics cho các khoảng HU
    hu_metrics = {}
    for lower, upper in hu_ranges:
        hu_metrics[f'hu_{lower}_{upper}_mae'] = []
    
    # Thư mục lưu ảnh kết quả
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            mri_images = batch['mri'].to(device)
            real_ct_images = batch['ct'].to(device) if 'ct' in batch else None
            
            if real_ct_images is None:
                continue  # Bỏ qua các batch không có ảnh CT thật
            
            # Sinh ảnh CT
            outputs = model(mri_images)
            synth_ct_images = outputs['fake_B']
            
            # Chuyển về numpy để tính metrics
            mri_np = mri_images.cpu().numpy()
            synth_ct_np = synth_ct_images.cpu().numpy()
            real_ct_np = real_ct_images.cpu().numpy()
            
            # Chuyển đổi giá trị về dạng HU cho CT
            synth_ct_hu = synth_ct_np * 4000 - 1000  # Giả định [0, 1] -> [-1000, 3000] HU
            real_ct_hu = real_ct_np * 4000 - 1000
            
            # Tính metrics tổng thể
            batch_metrics = calculate_metrics(synth_ct_hu, real_ct_hu)
            
            # Lưu metrics
            for metric in metrics_list:
                if metric in batch_metrics:
                    all_metrics[metric].append(batch_metrics[metric])
            
            # Phân đoạn mô trên ảnh CT thật
            tissue_masks = segment_tissues(real_ct_hu)
            
            # Tính metrics cho các loại mô
            batch_tissue_metrics = calculate_tissue_metrics(synth_ct_hu, real_ct_hu, tissue_masks)
            for key, value in batch_tissue_metrics.items():
                if key.endswith('_mae') and key in tissue_metrics:
                    tissue_metrics[key].append(value)
            
            # Tính metrics cho các khoảng HU
            batch_hu_metrics = calculate_hu_metrics(synth_ct_hu, real_ct_hu, hu_ranges)
            for key, value in batch_hu_metrics.items():
                hu_key = f'hu_{key}'
                if hu_key in hu_metrics:
                    hu_metrics[hu_key].append(value)
            
            # Lưu một số ảnh kết quả
            if i < 10:  # Chỉ lưu 10 mẫu đầu tiên
                for j in range(min(5, mri_images.size(0))):  # Lưu tối đa 5 ảnh từ mỗi batch
                    save_path = os.path.join(output_images_dir, f'sample_{i}_{j}.png')
                    visualize_results(
                        mri_np[j], synth_ct_np[j], real_ct_np[j],
                        save_path=save_path
                    )
    
    # Tính giá trị trung bình của các metrics
    results = {}
    for metric in metrics_list:
        if all_metrics[metric]:
            results[metric] = np.mean(all_metrics[metric])
    
    # Tính giá trị trung bình cho tissue metrics
    for key, values in tissue_metrics.items():
        if values:
            results[key] = np.mean(values)
    
    # Tính giá trị trung bình cho HU range metrics
    for key, values in hu_metrics.items():
        if values:
            results[key] = np.mean(values)
    
    return results

def generate_evaluation_report(results, output_dir):
    """
    Tạo báo cáo đánh giá
    
    Args:
        results (dict): Kết quả đánh giá
        output_dir (str): Thư mục đầu ra
    """
    # Tạo thư mục báo cáo
    report_dir = os.path.join(output_dir, 'report')
    os.makedirs(report_dir, exist_ok=True)
    
    # Lưu kết quả dưới dạng CSV
    df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    csv_path = os.path.join(report_dir, 'evaluation_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Tạo biểu đồ cho metrics chung
    general_metrics = ['mae', 'mse', 'rmse', 'psnr', 'ssim']
    general_values = [results.get(m, 0) for m in general_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.bar(general_metrics, general_values)
    plt.title('General Metrics')
    plt.ylabel('Value')
    plt.savefig(os.path.join(report_dir, 'general_metrics.png'))
    plt.close()
    
    # Tạo biểu đồ cho tissue metrics
    tissue_keys = [k for k in results.keys() if any(t in k for t in ['air', 'lung', 'fat', 'soft_tissue', 'bone'])]
    if tissue_keys:
        tissue_labels = [k.split('_')[0] for k in tissue_keys]
        tissue_values = [results[k] for k in tissue_keys]
        
        plt.figure(figsize=(12, 6))
        plt.bar(tissue_labels, tissue_values)
        plt.title('MAE by Tissue Type')
        plt.ylabel('MAE (HU)')
        plt.savefig(os.path.join(report_dir, 'tissue_mae.png'))
        plt.close()
    
    # Tạo biểu đồ cho HU range metrics
    hu_keys = [k for k in results.keys() if 'hu_' in k]
    if hu_keys:
        hu_labels = [k.replace('hu_', '').replace('_mae', '') for k in hu_keys]
        hu_values = [results[k] for k in hu_keys]
        
        plt.figure(figsize=(12, 6))
        plt.bar(hu_labels, hu_values)
        plt.title('MAE by HU Range')
        plt.ylabel('MAE (HU)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'hu_range_mae.png'))
        plt.close()
    
    # Tạo file Markdown tổng hợp
    with open(os.path.join(report_dir, 'evaluation_report.md'), 'w') as f:
        f.write('# Báo cáo đánh giá mô hình Synthetic CT\n\n')
        
        f.write('## Metrics tổng thể\n\n')
        f.write('| Metric | Value |\n')
        f.write('|--------|-------|\n')
        for metric in general_metrics:
            if metric in results:
                f.write(f'| {metric} | {results[metric]:.4f} |\n')
        
        f.write('\n## Metrics theo loại mô\n\n')
        f.write('| Tissue | MAE (HU) |\n')
        f.write('|--------|----------|\n')
        for key in tissue_keys:
            tissue = key.split('_')[0]
            f.write(f'| {tissue} | {results[key]:.4f} |\n')
        
        f.write('\n## Metrics theo khoảng HU\n\n')
        f.write('| HU Range | MAE |\n')
        f.write('|----------|-----|\n')
        for key in hu_keys:
            hu_range = key.replace('hu_', '').replace('_mae', '')
            f.write(f'| {hu_range} | {results[key]:.4f} |\n')
        
        f.write('\n\n*Báo cáo được tạo tự động*\n')
    
    print(f"Evaluation report saved to {report_dir}")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Xác định thiết bị
    device_name = args.device if args.device else config['inference']['device']
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Khởi tạo data loader
    data_module = MRIToCTDataModule(
        data_dir=args.data_dir,
        batch_size=config['evaluation']['test_batch_size'],
        slice_axis=config['data']['slice_axis'],
        paired=True
    )
    test_loader = data_module.test_dataloader()
    
    # Load mô hình
    checkpoint = torch.load(args.model, map_location=device)
    model = CycleGANModel().to(device)
    model.netG_A.load_state_dict(checkpoint['netG_A'])
    model.eval()
    
    # Tạo thư mục đầu ra
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Đánh giá mô hình
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, config, output_dir)
    
    # Tạo báo cáo
    print("Generating evaluation report...")
    generate_evaluation_report(results, output_dir)
    
    print("Evaluation completed!")

if __name__ == '__main__':
    main() 