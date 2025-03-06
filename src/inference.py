#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import pydicom
import SimpleITK as sitk
from pathlib import Path
import sys
import h5py
import torch.nn.functional as F

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cycle_gan import CycleGANModel
from src.data_processing.preprocess import read_dicom_series, normalize_image
from src.utils.visualization import save_dicom, visualize_results

def parse_args():
    parser = argparse.ArgumentParser(description='Chuyển đổi ảnh MRI sang CT mô phỏng')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Đường dẫn đến file cấu hình')
    parser.add_argument('--input', type=str, required=True,
                        help='Đường dẫn đến thư mục chứa ảnh DICOM MRI hoặc file h5')
    parser.add_argument('--output', type=str, required=True,
                        help='Đường dẫn đến thư mục lưu kết quả')
    parser.add_argument('--model', type=str, default=None,
                        help='Đường dẫn đến checkpoint mô hình (ghi đè cấu hình)')
    parser.add_argument('--device', type=str, default=None,
                        help='Thiết bị để thực hiện inference (cuda/cpu)')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_path, device):
    """Load mô hình từ checkpoint"""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Khởi tạo mô hình
    model = CycleGANModel().to(device)
    
    # Load weights
    model.netG_A.load_state_dict(checkpoint['netG_A'])
    
    # Đặt mô hình ở chế độ evaluation
    model.netG_A.eval()
    
    return model

def process_mri_volume(mri_data, model, device, batch_size=4, slice_axis=0):
    """
    Xử lý volume MRI để tạo ra volume CT mô phỏng
    
    Args:
        mri_data (numpy.ndarray): Dữ liệu MRI dạng 3D
        model (CycleGANModel): Mô hình CycleGAN đã huấn luyện
        device (torch.device): Thiết bị để thực hiện inference
        batch_size (int): Kích thước batch
        slice_axis (int): Trục lấy slice (0: axial, 1: coronal, 2: sagittal)
    
    Returns:
        numpy.ndarray: Volume CT mô phỏng
    """
    # Lấy kích thước volume
    if slice_axis == 0:
        num_slices = mri_data.shape[0]
        H, W = mri_data.shape[1], mri_data.shape[2]
    elif slice_axis == 1:
        num_slices = mri_data.shape[1]
        H, W = mri_data.shape[0], mri_data.shape[2]
    else:  # slice_axis == 2
        num_slices = mri_data.shape[2]
        H, W = mri_data.shape[0], mri_data.shape[1]
    
    # Khởi tạo volume CT mô phỏng
    synth_ct_data = np.zeros_like(mri_data)
    
    # Xử lý từng batch các slices
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_slices, batch_size), desc="Processing slices"):
            end_idx = min(start_idx + batch_size, num_slices)
            batch_slices = []
            
            # Lấy các slices trong batch
            for slice_idx in range(start_idx, end_idx):
                if slice_axis == 0:
                    slice_data = mri_data[slice_idx, :, :]
                elif slice_axis == 1:
                    slice_data = mri_data[:, slice_idx, :]
                else:  # slice_axis == 2
                    slice_data = mri_data[:, :, slice_idx]
                
                # Chuẩn bị slice cho model
                slice_tensor = torch.from_numpy(slice_data).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # Resize nếu cần để phù hợp với mô hình
                if slice_tensor.shape[2] % 4 != 0 or slice_tensor.shape[3] % 4 != 0:
                    new_h = ((slice_tensor.shape[2] // 4) + 1) * 4
                    new_w = ((slice_tensor.shape[3] // 4) + 1) * 4
                    slice_tensor = F.interpolate(slice_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                batch_slices.append(slice_tensor)
            
            # Ghép các slices thành một batch
            batch_tensor = torch.cat(batch_slices, dim=0).to(device)
            
            # Thực hiện inference
            synth_ct_batch = model.netG_A(batch_tensor)
            
            # Lưu kết quả vào volume CT mô phỏng
            for i, slice_idx in enumerate(range(start_idx, end_idx)):
                # Resize về kích thước ban đầu nếu cần
                if synth_ct_batch[i:i+1].shape[2] != H or synth_ct_batch[i:i+1].shape[3] != W:
                    synth_ct_slice = F.interpolate(synth_ct_batch[i:i+1], size=(H, W), mode='bilinear', align_corners=False)
                    synth_ct_slice = synth_ct_slice.squeeze().cpu().numpy()
                else:
                    synth_ct_slice = synth_ct_batch[i].squeeze().cpu().numpy()
                
                # Lưu slice vào volume
                if slice_axis == 0:
                    synth_ct_data[slice_idx, :, :] = synth_ct_slice
                elif slice_axis == 1:
                    synth_ct_data[:, slice_idx, :] = synth_ct_slice
                else:  # slice_axis == 2
                    synth_ct_data[:, :, slice_idx] = synth_ct_slice
    
    return synth_ct_data

def assign_tissue_densities(synth_ct_data, config):
    """
    Gán hệ số mô cho ảnh CT mô phỏng
    
    Args:
        synth_ct_data (numpy.ndarray): Volume CT mô phỏng với giá trị [0, 1]
        config (dict): Cấu hình với thông tin các vùng HU
    
    Returns:
        numpy.ndarray: Volume CT mô phỏng với giá trị HU
    """
    # Giả định rằng synth_ct_data nằm trong khoảng [0, 1]
    
    # Chuyển đổi về thang HU (Hounsfield Units) cho CT
    # Thông thường, không khí ~ -1000 HU, nước ~ 0 HU, xương ~ 1000 HU
    
    # Cách đơn giản: Ánh xạ tuyến tính từ [0, 1] sang [-1000, 1000]
    hu_data = synth_ct_data * 2000 - 1000
    
    # Cách nâng cao hơn: Phân đoạn mô dựa trên ngưỡng
    # và gán giá trị HU cụ thể cho từng loại mô
    # Ví dụ:
    # - Không khí: < 0.1 -> -1000 HU
    # - Phổi: 0.1-0.2 -> -700 HU
    # - Mỡ: 0.2-0.3 -> -100 HU
    # - Mô mềm: 0.3-0.6 -> 50 HU
    # - Xương nhẹ: 0.6-0.8 -> 300 HU
    # - Xương đặc: > 0.8 -> 800 HU
    
    # Áp dụng phân đoạn và gán giá trị HU
    hu_data_segmented = np.zeros_like(synth_ct_data)
    hu_data_segmented[synth_ct_data < 0.1] = -1000  # Không khí
    hu_data_segmented[(synth_ct_data >= 0.1) & (synth_ct_data < 0.2)] = -700  # Phổi
    hu_data_segmented[(synth_ct_data >= 0.2) & (synth_ct_data < 0.3)] = -100  # Mỡ
    hu_data_segmented[(synth_ct_data >= 0.3) & (synth_ct_data < 0.6)] = 50    # Mô mềm
    hu_data_segmented[(synth_ct_data >= 0.6) & (synth_ct_data < 0.8)] = 300   # Xương nhẹ
    hu_data_segmented[synth_ct_data >= 0.8] = 800  # Xương đặc
    
    # Phiên bản kết hợp: 80% phân đoạn + 20% tuyến tính để duy trì sự mượt mà
    alpha = 0.8
    combined_hu = alpha * hu_data_segmented + (1 - alpha) * hu_data
    
    return combined_hu

def save_result(synth_ct_data, metadata, output_dir, save_dicom=True):
    """
    Lưu kết quả CT mô phỏng
    
    Args:
        synth_ct_data (numpy.ndarray): Volume CT mô phỏng với giá trị HU
        metadata (dict): Metadata từ MRI gốc
        output_dir (str): Thư mục đầu ra
        save_dicom (bool): Có lưu dưới dạng DICOM hay không
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu dưới dạng file npy
    np.save(output_dir / "synthetic_ct.npy", synth_ct_data)
    
    # Lưu dưới dạng file DICOM nếu cần
    if save_dicom:
        dicom_dir = output_dir / "dicom"
        dicom_dir.mkdir(exist_ok=True)
        
        for i in tqdm(range(synth_ct_data.shape[0]), desc="Saving DICOM files"):
            # Tạo metadata cho slice hiện tại
            slice_metadata = metadata.copy()
            if "SliceLocation" in slice_metadata:
                slice_metadata["SliceLocation"] = metadata["SliceLocation"] + i * metadata.get("SliceThickness", 1.0)
            
            # Tạo tên file
            dicom_path = dicom_dir / f"slice_{i:04d}.dcm"
            
            # Lưu DICOM
            save_dicom(
                synth_ct_data[i],
                slice_metadata,
                str(dicom_path),
                series_desc="Synthetic CT from MRI"
            )
    
    print(f"Results saved to {output_dir}")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Xác định thiết bị
    device_name = args.device if args.device else config['inference']['device']
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = args.model if args.model else os.path.join(config['training']['output_dir'], 'checkpoints', config['inference']['checkpoint'])
    model = load_model(model_path, device)
    
    # Xác định đường dẫn đầu vào và đầu ra
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Đọc dữ liệu MRI
    if input_path.is_dir():
        # Đọc từ thư mục DICOM
        mri_data, metadata = read_dicom_series(str(input_path))
        mri_data_norm = normalize_image(mri_data, 'mri')
    elif input_path.suffix == '.h5':
        # Đọc từ file h5
        with h5py.File(input_path, 'r') as f:
            mri_data = f['mri'][:]
            mri_data_norm = mri_data  # Giả định đã được chuẩn hóa
            metadata = {}
            if 'mri_metadata' in f:
                for key in f['mri_metadata'].attrs:
                    metadata[key] = f['mri_metadata'].attrs[key]
    else:
        raise ValueError(f"Unsupported input format: {input_path}")
    
    # Xử lý volume MRI để tạo ra volume CT mô phỏng
    slice_axis = config['data']['slice_axis']
    batch_size = config['inference']['batch_size']
    synth_ct_data = process_mri_volume(mri_data_norm, model, device, batch_size, slice_axis)
    
    # Gán hệ số mô
    synth_ct_data_hu = assign_tissue_densities(synth_ct_data, config)
    
    # Lưu kết quả
    save_result(
        synth_ct_data_hu,
        metadata,
        output_dir,
        save_dicom=config['inference']['save_dicom']
    )
    
    # Tạo ảnh trực quan hóa
    sample_slice = mri_data.shape[0] // 2
    visualize_results(
        mri_data_norm[sample_slice],
        synth_ct_data[sample_slice],
        save_path=output_dir / "visualization.png"
    )
    
    print("Conversion completed successfully!")

if __name__ == '__main__':
    main()