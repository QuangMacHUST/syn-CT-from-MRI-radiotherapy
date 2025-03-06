#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.utils as vutils
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import datetime

def tensor_to_numpy(tensor):
    """Chuyển đổi tensor sang numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    return tensor

def normalize_for_display(image, min_val=None, max_val=None):
    """Chuẩn hóa ảnh để hiển thị"""
    image = tensor_to_numpy(image)
    
    if min_val is None:
        min_val = np.min(image)
    if max_val is None:
        max_val = np.max(image)
    
    image = (image - min_val) / (max_val - min_val + 1e-8)
    return np.clip(image, 0, 1)

def save_images(images, save_path, nrow=4, padding=2, normalize=True):
    """Lưu nhiều ảnh tensor thành một grid"""
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)
    
    # Đảm bảo định dạng [N, C, H, W]
    if len(images.shape) == 2:  # [H, W]
        images = images.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif len(images.shape) == 3:  # [N, H, W] or [C, H, W]
        if images.shape[0] == 1 or images.shape[0] == 3:  # [C, H, W]
            images = images.unsqueeze(0)  # [1, C, H, W]
        else:  # [N, H, W]
            images = images.unsqueeze(1)  # [N, 1, H, W]
    
    # Tạo grid
    grid = vutils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize)
    
    # Chuyển đổi sang numpy để lưu với matplotlib
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().detach().numpy().transpose((1, 2, 0))
    
    # Điều chỉnh thành ảnh grayscale nếu kênh = 1
    if grid.shape[2] == 1:
        grid = grid.squeeze(2)
    
    # Lưu ảnh
    plt.figure(figsize=(12, 12))
    if len(grid.shape) == 2:  # grayscale
        plt.imshow(grid, cmap='gray')
    else:  # RGB
        plt.imshow(grid)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_results(mri_images, synth_ct_images, real_ct_images=None, save_path=None, slice_idx=None):
    """Trực quan hóa kết quả chuyển đổi MRI sang CT"""
    mri_images = tensor_to_numpy(mri_images)
    synth_ct_images = tensor_to_numpy(synth_ct_images)
    
    if real_ct_images is not None:
        real_ct_images = tensor_to_numpy(real_ct_images)
    
    # Chọn slice để hiển thị nếu là volume 3D
    if len(mri_images.shape) == 4:  # [N, C, H, W]
        if slice_idx is None:
            slice_idx = mri_images.shape[0] // 2
        
        mri_image = mri_images[slice_idx, 0]  # Lấy kênh đầu tiên
        synth_ct_image = synth_ct_images[slice_idx, 0]
        real_ct_image = real_ct_images[slice_idx, 0] if real_ct_images is not None else None
    else:  # Đã là một slice
        mri_image = mri_images.squeeze()
        synth_ct_image = synth_ct_images.squeeze()
        real_ct_image = real_ct_images.squeeze() if real_ct_images is not None else None
    
    # Tạo figure
    if real_ct_image is not None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Hiển thị ảnh MRI
        axs[0].imshow(normalize_for_display(mri_image), cmap='gray')
        axs[0].set_title('MRI Input')
        axs[0].axis('off')
        
        # Hiển thị ảnh CT tổng hợp
        axs[1].imshow(normalize_for_display(synth_ct_image), cmap='gray')
        axs[1].set_title('Synthetic CT')
        axs[1].axis('off')
        
        # Hiển thị ảnh CT thật nếu có
        axs[2].imshow(normalize_for_display(real_ct_image), cmap='gray')
        axs[2].set_title('Real CT')
        axs[2].axis('off')
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Hiển thị ảnh MRI
        axs[0].imshow(normalize_for_display(mri_image), cmap='gray')
        axs[0].set_title('MRI Input')
        axs[0].axis('off')
        
        # Hiển thị ảnh CT tổng hợp
        axs[1].imshow(normalize_for_display(synth_ct_image), cmap='gray')
        axs[1].set_title('Synthetic CT')
        axs[1].axis('off')
    
    plt.tight_layout()
    
    # Lưu ảnh nếu có đường dẫn
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    
    plt.close()

def save_dicom(image_data, metadata, output_path, series_desc="Synthetic CT"):
    """
    Lưu ảnh dưới dạng file DICOM
    
    Args:
        image_data (numpy.ndarray): Dữ liệu ảnh CT mô phỏng
        metadata (dict): Metadata cần bao gồm trong file DICOM
        output_path (str): Đường dẫn lưu file DICOM
        series_desc (str): Mô tả series
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Chuyển đổi ảnh sang định dạng phù hợp với DICOM
    image_data = tensor_to_numpy(image_data)
    
    # Rescale về Hounsfield Units cho CT (nếu cần)
    if not (-1500 < np.min(image_data) < 3500) or not (-1500 < np.max(image_data) < 3500):
        # Giả định là ảnh đã được chuẩn hóa, chuyển sang thang HU
        image_data = image_data * 4000 - 1000  # Map [0, 1] sang [-1000, 3000]
    
    # Giới hạn giá trị HU và chuyển sang uint16
    image_data = np.clip(image_data, -1000, 3000)
    # Chuyển đổi sang định dạng uint16 cho DICOM
    image_data_uint16 = ((image_data + 1000) / 4000 * 65535).astype(np.uint16)
    
    try:
        # Tạo đối tượng DICOM mới
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Tạo dataset
        ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Thiết lập thời gian
        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        ds.ContentTime = dt.strftime('%H%M%S.%f')
        
        # Thiết lập các thông tin chung
        ds.Modality = "CT"
        ds.SeriesDescription = series_desc
        ds.Manufacturer = "Synthetic CT"
        ds.ManufacturerModelName = "MRI-to-CT"
        ds.PatientName = metadata.get("PatientName", "SYNTHETIC")
        ds.PatientID = metadata.get("PatientID", "SYN-CT-001")
        ds.PatientBirthDate = metadata.get("PatientBirthDate", "")
        ds.PatientSex = metadata.get("PatientSex", "")
        ds.StudyDate = metadata.get("StudyDate", dt.strftime('%Y%m%d'))
        ds.StudyTime = metadata.get("StudyTime", dt.strftime('%H%M%S.%f'))
        
        # Copy các metadata khác nếu có
        for key, value in metadata.items():
            if hasattr(ds, key) and key not in ['PixelData']:
                try:
                    setattr(ds, key, value)
                except:
                    pass  # Bỏ qua các trường không thể thiết lập
        
        # Thiết lập các thông số ảnh
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # unsigned
        
        # Thiết lập thông tin CT
        ds.RescaleIntercept = -1000.0
        ds.RescaleSlope = 1.0
        ds.RescaleType = "HU"
        
        # Thiết lập kích thước ảnh
        if len(image_data.shape) == 3:
            # 3D volume, lấy slice đầu tiên
            ds.Rows, ds.Columns = image_data.shape[1], image_data.shape[2]
            image_data_uint16 = image_data_uint16[0]
        else:
            # 2D image
            ds.Rows, ds.Columns = image_data.shape
        
        # Thiết lập dữ liệu pixel
        ds.PixelData = image_data_uint16.tobytes()
        
        # Lưu file
        ds.save_as(output_path)
        print(f"Saved DICOM file to {output_path}")
        
    except Exception as e:
        print(f"Error saving DICOM file: {e}")
        # Tạo một file DICOM đơn giản hơn nếu có lỗi
        try:
            ds = Dataset()
            ds.Modality = "CT"
            ds.SeriesDescription = series_desc
            ds.PatientName = "SYNTHETIC"
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.RescaleIntercept = -1000.0
            ds.RescaleSlope = 1.0
            ds.RescaleType = "HU"
            
            if len(image_data.shape) == 3:
                ds.Rows, ds.Columns = image_data.shape[1], image_data.shape[2]
                image_data_uint16 = image_data_uint16[0]
            else:
                ds.Rows, ds.Columns = image_data.shape
            
            ds.PixelData = image_data_uint16.tobytes()
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            
            ds.save_as(output_path, write_like_original=False)
            print(f"Saved simplified DICOM file to {output_path}")
        except Exception as e2:
            print(f"Failed to save even simplified DICOM: {e2}")
            # Lưu ảnh dạng numpy nếu DICOM thất bại
            np.save(output_path.replace('.dcm', '.npy'), image_data)