#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import h5py
import nibabel as nib

def parse_args():
    parser = argparse.ArgumentParser(description='Tiền xử lý dữ liệu DICOM MRI và CT')
    parser.add_argument('--input_mri', type=str, default='data/raw/mri',
                        help='Đường dẫn đến thư mục chứa ảnh DICOM MRI')
    parser.add_argument('--input_ct', type=str, default='data/raw/ct',
                        help='Đường dẫn đến thư mục chứa ảnh DICOM CT (nếu có, cho đánh giá)')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Đường dẫn đến thư mục lưu dữ liệu đã xử lý')
    parser.add_argument('--paired', action='store_true',
                        help='Dữ liệu MRI và CT đã được ghép cặp')
    return parser.parse_args()

def read_dicom_series(directory):
    """Đọc tập ảnh DICOM từ thư mục và trả về dưới dạng mảng 3D"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Chuyển đổi sang định dạng numpy
    array = sitk.GetArrayFromImage(image)
    
    # Trả về thông tin hình ảnh và mảng pixel
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    
    return array, {'spacing': spacing, 'origin': origin, 'direction': direction}

def normalize_image(image, modality='mri'):
    """Chuẩn hóa giá trị pixel của ảnh"""
    if modality.lower() == 'mri':
        # Chuẩn hóa MRI về khoảng [0, 1]
        min_val = np.percentile(image, 1)
        max_val = np.percentile(image, 99)
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val + 1e-8)
    elif modality.lower() == 'ct':
        # Chuẩn hóa CT về khoảng HU thích hợp
        # Thông thường HU từ -1000 đến 3000
        image = np.clip(image, -1000, 3000)
        image = (image + 1000) / 4000.0
    
    return image

def preprocess_and_save(mri_dir, ct_dir, output_dir, paired=False):
    """Xử lý và lưu dữ liệu MRI và CT"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo thư mục đầu ra
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Danh sách các thư mục con trong thư mục MRI
    if paired:
        # Giả định rằng MRI và CT có tên thư mục con tương ứng
        patient_dirs = [d for d in os.listdir(mri_dir) if os.path.isdir(os.path.join(mri_dir, d))]
        
        # Chia thành tập train, validation và test (70%, 15%, 15%)
        np.random.shuffle(patient_dirs)
        train_split = int(len(patient_dirs) * 0.7)
        val_split = int(len(patient_dirs) * 0.85)
        
        train_patients = patient_dirs[:train_split]
        val_patients = patient_dirs[train_split:val_split]
        test_patients = patient_dirs[val_split:]
        
        print(f"Số lượng bệnh nhân: Train={len(train_patients)}, Val={len(val_patients)}, Test={len(test_patients)}")
        
        # Hàm xử lý và lưu dữ liệu cho một bệnh nhân
        def process_patient(patient_id, subset):
            out_dir = train_dir if subset == 'train' else val_dir if subset == 'val' else test_dir
            
            mri_patient_dir = os.path.join(mri_dir, patient_id)
            ct_patient_dir = os.path.join(ct_dir, patient_id) if ct_dir else None
            
            # Đọc và xử lý MRI
            try:
                mri_array, mri_metadata = read_dicom_series(mri_patient_dir)
                mri_norm = normalize_image(mri_array, 'mri')
                
                # Đọc và xử lý CT nếu có
                if ct_patient_dir and os.path.exists(ct_patient_dir):
                    ct_array, ct_metadata = read_dicom_series(ct_patient_dir)
                    ct_norm = normalize_image(ct_array, 'ct')
                    
                    # Kiểm tra kích thước ảnh MRI và CT
                    if mri_norm.shape != ct_norm.shape:
                        print(f"Cảnh báo: Kích thước MRI và CT không khớp cho bệnh nhân {patient_id}")
                        # Có thể thêm code resampling ở đây
                else:
                    ct_array = None
                    ct_metadata = None
                    ct_norm = None
                
                # Lưu dữ liệu đã xử lý
                h5_file = out_dir / f"{patient_id}.h5"
                with h5py.File(h5_file, 'w') as f:
                    f.create_dataset('mri', data=mri_norm)
                    if ct_norm is not None:
                        f.create_dataset('ct', data=ct_norm)
                        f.create_dataset('ct_raw', data=ct_array)
                    
                    # Lưu metadata
                    mri_meta_grp = f.create_group('mri_metadata')
                    for key, value in mri_metadata.items():
                        if isinstance(value, tuple):
                            mri_meta_grp.create_dataset(key, data=np.array(value))
                        else:
                            mri_meta_grp.attrs[key] = value
                    
                    if ct_metadata is not None:
                        ct_meta_grp = f.create_group('ct_metadata')
                        for key, value in ct_metadata.items():
                            if isinstance(value, tuple):
                                ct_meta_grp.create_dataset(key, data=np.array(value))
                            else:
                                ct_meta_grp.attrs[key] = value
                
                return True
            except Exception as e:
                print(f"Lỗi khi xử lý bệnh nhân {patient_id}: {e}")
                return False
        
        # Xử lý các tập dữ liệu
        for subset, patients in [('train', train_patients), ('val', val_patients), ('test', test_patients)]:
            successful = 0
            for patient in tqdm(patients, desc=f"Đang xử lý tập {subset}"):
                if process_patient(patient, subset):
                    successful += 1
            print(f"Hoàn thành tập {subset}: {successful}/{len(patients)} thành công")
    else:
        # Xử lý không ghép cặp (điều này có thể thay đổi tùy thuộc vào cấu trúc dữ liệu của bạn)
        pass

def main():
    args = parse_args()
    print("Bắt đầu tiền xử lý dữ liệu...")
    preprocess_and_save(args.input_mri, args.input_ct, args.output, args.paired)
    print("Hoàn thành tiền xử lý dữ liệu!")

if __name__ == '__main__':
    main() 