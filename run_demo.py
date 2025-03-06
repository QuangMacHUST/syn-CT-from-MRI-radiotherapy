#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import CycleGANModel
from src.utils.visualization import visualize_results, save_dicom
from src.inference import assign_tissue_densities

def parse_args():
    parser = argparse.ArgumentParser(description='Demo chuyển đổi MRI sang CT mô phỏng')
    parser.add_argument('--output_dir', type=str, default='data/output/demo', help='Thư mục đầu ra')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='File cấu hình')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Thiết bị tính toán')
    return parser.parse_args()

def load_config(config_path):
    """Tải cấu hình từ file YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_synthetic_mri_data(num_slices=15, size=256):
    """Tạo dữ liệu MRI giả lập"""
    print("Tạo dữ liệu MRI giả lập...")
    
    # Tạo background
    mri_data = np.zeros((num_slices, size, size))
    
    # Tạo vòng tròn bên ngoài mô phỏng đầu
    center = (size // 2, size // 2)
    radius = size // 2 - 10
    
    for z in range(num_slices):
        slice_ratio = z / num_slices
        r_scale = 0.5 + slice_ratio if slice_ratio < 0.5 else 1.5 - slice_ratio
        
        # Tạo đường viền đầu
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if dist < radius * r_scale:
                    mri_data[z, i, j] = 0.8  # Mô mềm
        
        # Tạo vùng tủy xương
        center_radius = radius * r_scale * 0.8
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if dist < center_radius:
                    mri_data[z, i, j] = 0.6  # Tủy xương
        
        # Tạo vùng xương
        bone_radius = radius * r_scale * 0.9
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if bone_radius * 0.95 < dist < bone_radius * 1.05:
                    mri_data[z, i, j] = 0.3  # Xương
        
        # Tạo một số đốm sáng ngẫu nhiên
        num_spots = 5
        for _ in range(num_spots):
            x = np.random.randint(center[0] - center_radius//2, center[0] + center_radius//2)
            y = np.random.randint(center[1] - center_radius//2, center[1] + center_radius//2)
            spot_size = np.random.randint(5, 15)
            
            for i in range(max(0, x-spot_size), min(size, x+spot_size)):
                for j in range(max(0, y-spot_size), min(size, y+spot_size)):
                    if np.sqrt((i-x)**2 + (j-y)**2) < spot_size:
                        mri_data[z, i, j] = 0.9  # Điểm sáng
    
    # Thêm nhiễu nhẹ
    mri_data += np.random.normal(0, 0.05, mri_data.shape)
    mri_data = np.clip(mri_data, 0, 1)
    
    return mri_data

def generate_synthetic_ct_data(mri_data):
    """Tạo dữ liệu CT giả lập từ MRI"""
    print("Tạo dữ liệu CT giả lập...")
    
    # Map các khoảng giá trị MRI sang CT (HU)
    ct_data = np.zeros_like(mri_data)
    
    # Áp dụng các ngưỡng để chuyển đổi MRI sang CT
    # Không khí: -1000 HU
    ct_data[mri_data < 0.1] = -1000
    
    # Phổi: -700 HU
    ct_data[(0.1 <= mri_data) & (mri_data < 0.3)] = -700
    
    # Mỡ: -100 HU
    ct_data[(0.3 <= mri_data) & (mri_data < 0.5)] = -100
    
    # Mô mềm: 50 HU
    ct_data[(0.5 <= mri_data) & (mri_data < 0.7)] = 50
    
    # Mô mềm đặc: 100 HU
    ct_data[(0.7 <= mri_data) & (mri_data < 0.8)] = 100
    
    # Xương: 700 HU
    ct_data[mri_data >= 0.8] = 700
    
    # Thêm nhiễu nhẹ
    ct_data += np.random.normal(0, 30, ct_data.shape)
    
    # Chuẩn hóa về khoảng [-1000, 3000] HU
    ct_data = np.clip(ct_data, -1000, 3000)
    
    return ct_data

def main():
    args = parse_args()
    
    # Tạo thư mục đầu ra
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tải cấu hình
    config = load_config(args.config)
    
    # Tạo dữ liệu giả lập
    mri_data = generate_synthetic_mri_data()
    real_ct_data = generate_synthetic_ct_data(mri_data)
    
    # Khởi tạo mô hình đơn giản để demo
    device = torch.device(args.device)
    print(f"Sử dụng thiết bị: {device}")
    
    # Tạo một mô hình giả định cho demo
    model = CycleGANModel().to(device)
    
    # Thay thế quá trình suy luận thực tế bằng việc sử dụng dữ liệu CT giả lập có thêm nhiễu
    print("Mô phỏng quá trình chuyển đổi MRI sang CT...")
    synth_ct_data = real_ct_data + np.random.normal(0, 50, real_ct_data.shape)
    synth_ct_data = np.clip(synth_ct_data, -1000, 3000)
    
    # Gán hệ số mô
    print("Gán hệ số mô...")
    synth_ct_data_hu = synth_ct_data  # Đã là HU, không cần gán lại
    
    # Hiển thị kết quả
    mid_slice = mri_data.shape[0] // 2
    
    # Chuẩn hóa MRI về khoảng [0, 1] để hiển thị
    mri_slice_norm = (mri_data[mid_slice] - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
    
    # Chuẩn hóa CT về khoảng [0, 1] cho việc hiển thị
    real_ct_slice_norm = (real_ct_data[mid_slice] - (-1000)) / (3000 - (-1000))
    synth_ct_slice_norm = (synth_ct_data[mid_slice] - (-1000)) / (3000 - (-1000))
    
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(mri_slice_norm, cmap='gray')
    plt.title('MRI Input')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(synth_ct_slice_norm, cmap='gray')
    plt.title('Synthetic CT')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(real_ct_slice_norm, cmap='gray')
    plt.title('Real CT (Ground Truth)')
    plt.axis('off')
    
    plt.tight_layout()
    output_image_path = output_dir / 'demo_result.png'
    plt.savefig(output_image_path)
    plt.close()
    
    print(f"Đã lưu kết quả trực quan hóa tại: {output_image_path}")
    
    # Lưu dữ liệu dưới dạng DICOM
    metadata = {
        "PatientName": "DEMO",
        "PatientID": "DEMO-001",
        "PatientBirthDate": "20000101",
        "PatientSex": "O",
        "StudyDate": "20230101",
    }
    
    # Lưu slice giữa làm ví dụ
    output_dicom_path = output_dir / 'demo_synthetic_ct.dcm'
    save_dicom(synth_ct_data[mid_slice], metadata, str(output_dicom_path), series_desc="Demo Synthetic CT")
    
    print(f"Đã lưu kết quả DICOM tại: {output_dicom_path}")
    print("Demo hoàn thành!")

if __name__ == "__main__":
    main() 