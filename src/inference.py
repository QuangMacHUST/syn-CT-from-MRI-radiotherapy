#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py
import pydicom
import time
import SimpleITK as sitk

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AVAILABLE_MODELS
from data_processing.preprocess import read_dicom_series, normalize_image
from utils.visualization import visualize_results, save_dicom

def parse_args():
    parser = argparse.ArgumentParser(description='Chuyển đổi ảnh MRI thành CT mô phỏng')
    parser.add_argument('--input', type=str, required=True,
                        help='Đường dẫn đến thư mục chứa ảnh DICOM MRI')
    parser.add_argument('--output', type=str, default='data/output/synthetic_ct',
                        help='Đường dẫn đến thư mục lưu kết quả')
    parser.add_argument('--model', type=str, default='data/output/models/checkpoints/best_model.pth',
                        help='Đường dẫn đến checkpoint mô hình')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Đường dẫn đến file cấu hình')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Thiết bị tính toán (cuda hoặc cpu)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Kích thước batch cho inference')
    parser.add_argument('--save_dicom', action='store_true',
                        help='Lưu kết quả dưới dạng DICOM')
    return parser.parse_args()

def load_config(config_path):
    """Tải cấu hình từ file YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_path, device, model_type='cyclegan'):
    """
    Tải mô hình từ checkpoint
    
    Args:
        model_path (str): Đường dẫn đến checkpoint mô hình
        device (torch.device): Thiết bị tính toán
        model_type (str): Loại mô hình (cyclegan, unet, pix2pix, attentiongan, unit)
    
    Returns:
        nn.Module: Mô hình đã tải
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Kiểm tra xem loại mô hình có được hỗ trợ không
        if model_type.lower() not in AVAILABLE_MODELS:
            raise ValueError(f"Loại mô hình không hợp lệ: {model_type}. Các loại mô hình được hỗ trợ: {', '.join(AVAILABLE_MODELS.keys())}")
        
        # Khởi tạo mô hình dựa trên loại
        model_class = AVAILABLE_MODELS[model_type.lower()]
        model = model_class().to(device)
        
        # Tải trọng số tùy thuộc vào loại mô hình
        if model_type.lower() == 'cyclegan':
            # CycleGAN sử dụng netG_A để chuyển đổi MRI -> CT
            if 'netG_A' in checkpoint:
                model.netG_A.load_state_dict(checkpoint['netG_A'])
            elif 'generator' in checkpoint:
                model.netG_A.load_state_dict(checkpoint['generator'])
            else:
                # Nếu không tìm thấy key, thử tải trực tiếp
                model.netG_A.load_state_dict(checkpoint)
        elif model_type.lower() == 'unet':
            # UNet sử dụng một generator duy nhất
            if 'netG' in checkpoint:
                model.netG.load_state_dict(checkpoint['netG'])
            else:
                model.netG.load_state_dict(checkpoint)
        elif model_type.lower() == 'pix2pix':
            # Pix2Pix sử dụng một generator UNet
            if 'netG' in checkpoint:
                model.netG.load_state_dict(checkpoint['netG'])
            else:
                model.netG.load_state_dict(checkpoint)
        elif model_type.lower() == 'attentiongan':
            # AttentionGAN sử dụng netG_A như CycleGAN
            if 'netG_A' in checkpoint:
                model.netG_A.load_state_dict(checkpoint['netG_A'])
            else:
                model.netG_A.load_state_dict(checkpoint)
        elif model_type.lower() == 'unit':
            # UNIT có nhiều thành phần
            if all(k in checkpoint for k in ['encoder_A', 'decoder_B']):
                model.encoder_A.load_state_dict(checkpoint['encoder_A'])
                model.gen_mu_var_A.load_state_dict(checkpoint['gen_mu_var_A'])
                model.decoder_B.load_state_dict(checkpoint['decoder_B'])
            else:
                model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        sys.exit(1)

def process_mri_volume(mri_data, model, device, batch_size=4, slice_axis=0, model_type='cyclegan'):
    """
    Chuyển đổi volume MRI thành volume CT mô phỏng
    
    Args:
        mri_data (numpy.ndarray): Dữ liệu MRI 3D
        model (nn.Module): Mô hình đã huấn luyện
        device (torch.device): Thiết bị tính toán
        batch_size (int): Kích thước batch
        slice_axis (int): Trục lấy slice (0: axial, 1: coronal, 2: sagittal)
        model_type (str): Loại mô hình (cyclegan, unet, pix2pix, attentiongan, unit)
    
    Returns:
        numpy.ndarray: Volume CT mô phỏng
    """
    # Đảm bảo mô hình ở chế độ đánh giá
    model.eval()
    
    # Tạo tensor đầu ra
    shape = list(mri_data.shape)
    ct_shape = shape.copy()
    
    # Trích xuất từng slice dọc theo trục đã chọn
    num_slices = shape[slice_axis]
    slices = []
    
    # Trích xuất tất cả slice dọc theo trục đã chọn
    for i in range(num_slices):
        if slice_axis == 0:
            mri_slice = mri_data[i, :, :]
        elif slice_axis == 1:
            mri_slice = mri_data[:, i, :]
        else:  # slice_axis == 2
            mri_slice = mri_data[:, :, i]
        
        # Thêm chiều cho channel và batch
        mri_slice = torch.from_numpy(mri_slice).float().unsqueeze(0).unsqueeze(0)
        slices.append(mri_slice)
    
    # Xử lý theo batch
    ct_slices = []
    with torch.no_grad():
        for i in tqdm(range(0, len(slices), batch_size), desc="Chuyển đổi MRI sang CT"):
            batch = torch.cat(slices[i:i+batch_size], dim=0).to(device)
            
            # Xử lý tùy thuộc vào loại mô hình
            if model_type.lower() == 'cyclegan':
                # CycleGAN trả về dict với 'fake_B' là kết quả CT
                output = model(batch)
                synth_ct_batch = output['fake_B']
            
            elif model_type.lower() == 'unet':
                # UNet trả về trực tiếp CT từ netG
                synth_ct_batch = model.netG(batch)
            
            elif model_type.lower() == 'pix2pix':
                # Pix2Pix cũng trả về dict với 'fake_B'
                output = model(batch)
                synth_ct_batch = output['fake_B']
            
            elif model_type.lower() == 'attentiongan':
                # AttentionGAN trả về tuple (ảnh, attention) từ netG_A
                fake_B, _ = model.netG_A(batch)
                synth_ct_batch = fake_B
            
            elif model_type.lower() == 'unit':
                # UNIT cần xử lý thông qua encode và decode
                output = model(batch)
                synth_ct_batch = output['fake_B']
            
            else:
                # Mặc định xử lý như một forward pass thông thường
                output = model(batch)
                if isinstance(output, dict) and 'fake_B' in output:
                    synth_ct_batch = output['fake_B']
                else:
                    synth_ct_batch = output
            
            # Thêm kết quả vào danh sách
            for j in range(synth_ct_batch.size(0)):
                ct_slices.append(synth_ct_batch[j, 0].cpu().numpy())
    
    # Tái tạo volume CT
    ct_data = np.zeros(shape)
    for i in range(num_slices):
        if slice_axis == 0:
            ct_data[i, :, :] = ct_slices[i]
        elif slice_axis == 1:
            ct_data[:, i, :] = ct_slices[i]
        else:  # slice_axis == 2
            ct_data[:, :, i] = ct_slices[i]
    
    return ct_data

def assign_tissue_densities(synth_ct_data, config):
    """
    Gán hệ số mô cho ảnh CT mô phỏng
    
    Args:
        synth_ct_data (numpy.ndarray): Dữ liệu CT mô phỏng, có thể ở dạng chuẩn hóa [0, 1]
        config (dict): Cấu hình
    
    Returns:
        numpy.ndarray: Dữ liệu CT mô phỏng với giá trị tính bằng HU
    """
    # Kiểm tra xem dữ liệu đã ở dạng HU chưa
    if synth_ct_data.min() >= -100 and synth_ct_data.max() <= 100 and synth_ct_data.mean() < 10:
        # Có vẻ như giá trị đã được chuẩn hóa, chuyển đổi về thang HU
        print("Chuyển đổi giá trị chuẩn hóa sang thang Hounsfield Units (HU)...")
        synth_ct_hu = synth_ct_data * 4000 - 1000  # Map [0, 1] sang [-1000, 3000]
    else:
        # Giá trị có thể đã ở dạng HU
        synth_ct_hu = synth_ct_data
    
    # Gán giá trị HU chuẩn cho các loại mô
    # Có thể tùy chỉnh thêm nếu cần
    print("Gán hệ số mô cho các vùng khác nhau...")
    
    # Giới hạn giá trị trong phạm vi HU hợp lệ
    synth_ct_hu = np.clip(synth_ct_hu, -1000, 3000)
    
    # Tạo bản sao để áp dụng gán hệ số mô
    synth_ct_tissue = synth_ct_hu.copy()
    
    # Định nghĩa các ngưỡng và giá trị HU cho từng loại mô
    tissue_thresholds = [
        # (min_hu, max_hu, assigned_hu, description)
        (-1000, -950, -1000, "Không khí"),  # Không khí: -1000 HU
        (-950, -700, -800, "Phổi"),        # Phổi: -800 HU
        (-700, -30, -100, "Mỡ"),          # Mỡ: -100 HU
        (-30, 100, 50, "Mô mềm"),         # Mô mềm: 50 HU
        (100, 300, 200, "Cơ"),            # Cơ: 200 HU
        (300, 1500, 700, "Xương"),        # Xương: 700 HU
        (1500, 3000, 2000, "Implant")     # Implant: 2000 HU
    ]
    
    # Áp dụng gán hệ số mô nếu được cấu hình
    if config.get('inference', {}).get('apply_tissue_mapping', False):
        print("Áp dụng bảng ánh xạ hệ số mô...")
        for min_hu, max_hu, assigned_hu, desc in tissue_thresholds:
            mask = (synth_ct_hu >= min_hu) & (synth_ct_hu <= max_hu)
            if np.any(mask):
                synth_ct_tissue[mask] = assigned_hu
                print(f"  - Gán {desc}: {np.sum(mask)} voxel")
        
        # Trả về dữ liệu đã gán hệ số mô
        return synth_ct_tissue
    else:
        # Không áp dụng gán hệ số mô, trả về dữ liệu gốc
        return synth_ct_hu

def save_result(synth_ct_data, metadata, output_dir, save_dicom=True):
    """
    Lưu kết quả CT mô phỏng
    
    Args:
        synth_ct_data (numpy.ndarray): Dữ liệu CT mô phỏng 3D
        metadata (dict): Metadata của MRI gốc
        output_dir (str): Thư mục đầu ra
        save_dicom (bool): Lưu dưới dạng DICOM
    """
    # Tạo thư mục đầu ra
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Giới hạn giá trị HU
    synth_ct_data = np.clip(synth_ct_data, -1000, 3000)
    
    # Lưu dưới dạng DICOM
    if save_dicom:
        dicom_dir = output_dir / 'dicom'
        dicom_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Lưu kết quả dưới dạng DICOM vào: {dicom_dir}")
        
        # Lưu từng slice thành một file DICOM
        for i in tqdm(range(synth_ct_data.shape[0]), desc="Lưu DICOM"):
            # Lấy slice và đảm bảo nó là 2D
            slice_data = synth_ct_data[i]
            if slice_data.ndim > 2:
                slice_data = slice_data.squeeze()
            
            # Lưu file DICOM với sử dụng metadata từ MRI gốc
            dicom_path = dicom_dir / f'synth_ct_{i:03d}.dcm'
            save_dicom(
                slice_data, 
                metadata, 
                str(dicom_path),
                series_desc=f"Synthetic CT {i}"
            )
    
    # Lưu dưới dạng numpy array
    np_path = output_dir / 'synth_ct_volume.npy'
    np.save(np_path, synth_ct_data)
    print(f"Đã lưu volume CT mô phỏng dưới dạng numpy array tại: {np_path}")
    
    # Lưu một số slice minh họa dưới dạng ảnh
    img_dir = output_dir / 'images'
    img_dir.mkdir(exist_ok=True, parents=True)
    
    # Lưu một số slice đại diện
    slice_indices = [
        synth_ct_data.shape[0] // 4,
        synth_ct_data.shape[0] // 2,
        synth_ct_data.shape[0] * 3 // 4
    ]
    
    for idx in slice_indices:
        slice_data = synth_ct_data[idx]
        if slice_data.ndim > 2:
            slice_data = slice_data.squeeze()
        
        # Chuẩn hóa để hiển thị
        slice_normalized = (slice_data - (-1000)) / (3000 - (-1000))
        
        # Lưu ảnh
        plt_path = img_dir / f'slice_{idx}.png'
        plt.figure(figsize=(8, 8))
        plt.imshow(slice_normalized, cmap='gray')
        plt.colorbar(label='HU')
        plt.title(f'Synthetic CT - Slice {idx}')
        plt.savefig(plt_path)
        plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Sử dụng model_type từ cấu hình nếu được cung cấp, nếu không sử dụng giá trị mặc định là cyclegan
    model_type = config['inference'].get('model_type', 'cyclegan')
    
    # Setup device
    device = torch.device(config['inference']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    # Load model
    model_path = Path(args.model) if args.model else Path(config['inference']['checkpoint'])
    print(f"Đang tải mô hình từ: {model_path}")
    model = load_model(model_path, device, model_type)
    
    # Đọc dữ liệu MRI
    print(f"Đọc dữ liệu MRI từ: {args.input}")
    try:
        mri_data, metadata = read_dicom_series(args.input)
        
        # Hiển thị thông tin cơ bản về dữ liệu
        print(f"Kích thước dữ liệu MRI: {mri_data.shape}")
        print(f"Giá trị min/max: {mri_data.min():.2f}/{mri_data.max():.2f}")
        
        # Chuẩn hóa dữ liệu MRI
        print("Chuẩn hóa dữ liệu MRI...")
        mri_data_norm = normalize_image(mri_data, 'mri')
        
        # Chuyển đổi MRI sang CT
        print("Chuyển đổi MRI sang CT mô phỏng...")
        synth_ct_data = process_mri_volume(
            mri_data_norm, 
            model, 
            device, 
            batch_size=args.batch_size, 
            slice_axis=config['data']['slice_axis'],
            model_type=model_type
        )
        
        # Gán hệ số mô
        print("Gán hệ số mô...")
        synth_ct_data_hu = assign_tissue_densities(synth_ct_data, config)
        
        # Lưu kết quả
        print(f"Lưu kết quả vào: {args.output}")
        save_result(synth_ct_data_hu, metadata, args.output, save_dicom=args.save_dicom)
        
        print("Chuyển đổi hoàn tất!")
        
    except Exception as e:
        print(f"Lỗi: {e}")
        sys.exit(1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()