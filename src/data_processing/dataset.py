# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from skimage import transform
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class MRIToCTDataset(Dataset):
    """Dataset cho việc chuyển đổi ảnh MRI sang CT"""
    
    def __init__(self, data_dir, mode='train', slice_axis=0, slice_range=None, transform=None, paired=True):
        """
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu đã tiền xử lý
            mode (str): 'train', 'val', hoặc 'test'
            slice_axis (int): Trục lấy slice (0: axial, 1: coronal, 2: sagittal)
            slice_range (tuple): Khoảng slices cần lấy (start, end)
            transform (callable, optional): Transform tùy chọn
            paired (bool): Dữ liệu MRI và CT đã được ghép cặp
        """
        self.data_dir = Path(data_dir) / mode
        self.mode = mode
        self.slice_axis = slice_axis
        self.slice_range = slice_range
        self.transform = transform
        self.paired = paired
        
        # Kiểm tra thư mục dữ liệu tồn tại
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {self.data_dir}")
        
        self.file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
        
        # Kiểm tra có file dữ liệu hay không
        if not self.file_list:
            raise ValueError(f"Không tìm thấy file h5 nào trong thư mục: {self.data_dir}")
            
        self.samples = []
        
        # Đọc trước thông tin các slice từ file h5
        for file_name in self.file_list:
            file_path = self.data_dir / file_name
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'mri' not in f:
                        print(f"Cảnh báo: File {file_path} không chứa dữ liệu MRI, bỏ qua.")
                        continue
                    
                    # Lấy dữ liệu MRI và chuyển thành numpy array
                    mri_data = np.array(f['mri'])
                    
                    # Kiểm tra xem file có dữ liệu CT không nếu là dữ liệu ghép cặp
                    if self.paired and 'ct' not in f:
                        print(f"Cảnh báo: File {file_path} không chứa dữ liệu CT mặc dù chế độ ghép cặp được kích hoạt. Bỏ qua.")
                        continue
                    
                    # Xác định range slices nếu không được cung cấp
                    if self.slice_range is None:
                        start_slice = 0
                        end_slice = mri_data.shape[self.slice_axis]
                    else:
                        start_slice, end_slice = self.slice_range
                        # Kiểm tra giá trị hợp lệ
                        if start_slice < 0:
                            start_slice = 0
                        end_slice = min(end_slice, mri_data.shape[self.slice_axis])
                    
                    # Thêm thông tin các slice vào danh sách mẫu
                    for slice_idx in range(start_slice, end_slice):
                        self.samples.append((str(file_path), slice_idx))
            except Exception as e:
                print(f"Lỗi khi đọc file {file_path}: {e}")
        
        # Kiểm tra số lượng mẫu
        if not self.samples:
            raise ValueError(f"Không tìm thấy mẫu nào trong thư mục: {self.data_dir} với cấu hình đã chọn")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, slice_idx = self.samples[idx]
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Lấy dữ liệu và chuyển thành numpy array
                mri_volume = np.array(f['mri'])
                
                # Kiểm tra kích thước dữ liệu
                if self.slice_axis >= len(mri_volume.shape):
                    raise ValueError(f"Slice axis {self.slice_axis} vượt quá số chiều của dữ liệu {len(mri_volume.shape)}")
                
                # Lấy slice MRI theo trục đã chọn
                if self.slice_axis == 0:
                    mri_slice = mri_volume[slice_idx, :, :]
                elif self.slice_axis == 1:
                    mri_slice = mri_volume[:, slice_idx, :]
                else:  # self.slice_axis == 2
                    mri_slice = mri_volume[:, :, slice_idx]
                
                # Đọc slice CT tương ứng nếu dữ liệu được ghép cặp
                ct_slice = None
                if self.paired and 'ct' in f:
                    ct_volume = np.array(f['ct'])
                    
                    # Kiểm tra kích thước dữ liệu CT và MRI khớp nhau
                    if ct_volume.shape != mri_volume.shape:
                        print(f"Cảnh báo: Kích thước CT {ct_volume.shape} và MRI {mri_volume.shape} không khớp nhau trong file {file_path}")
                    
                    if self.slice_axis == 0:
                        ct_slice = ct_volume[slice_idx, :, :]
                    elif self.slice_axis == 1:
                        ct_slice = ct_volume[:, slice_idx, :]
                    else:  # self.slice_axis == 2
                        ct_slice = ct_volume[:, :, slice_idx]
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path} tại slice {slice_idx}: {e}")
            # Trả về mẫu ngẫu nhiên khác nếu có lỗi
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        
        # Xử lý giá trị NaN hoặc Inf
        mri_slice = np.nan_to_num(mri_slice)
        
        # Chuyển đổi thành tensor
        mri_slice = torch.from_numpy(mri_slice).float().unsqueeze(0)  # Thêm kênh
        
        # Biến đổi dữ liệu
        if self.transform:
            mri_slice = self.transform(mri_slice)
        
        # Trả về cặp MRI-CT nếu có dữ liệu CT
        if ct_slice is not None:
            # Xử lý giá trị NaN hoặc Inf cho CT
            ct_slice = np.nan_to_num(ct_slice)
            
            ct_slice = torch.from_numpy(ct_slice).float().unsqueeze(0)  # Thêm kênh
            
            if self.transform:
                # Áp dụng cùng một transform cho CT
                state = torch.get_rng_state()
                mri_slice = self.transform(mri_slice)
                torch.set_rng_state(state)
                ct_slice = self.transform(ct_slice)
            
            return {'mri': mri_slice, 'ct': ct_slice}
        else:
            return {'mri': mri_slice}


class MRIToCTDataModule:
    """Lớp quản lý dữ liệu cho việc huấn luyện"""
    
    def __init__(self, data_dir, batch_size=8, slice_axis=0, slice_range=None, num_workers=4, paired=True):
        """
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu đã tiền xử lý
            batch_size (int): Kích thước batch
            slice_axis (int): Trục lấy slice (0: axial, 1: coronal, 2: sagittal)
            slice_range (tuple): Khoảng slices cần lấy (start, end)
            num_workers (int): Số lượng worker cho DataLoader
            paired (bool): Dữ liệu MRI và CT đã được ghép cặp
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.slice_axis = slice_axis
        self.slice_range = slice_range
        self.num_workers = num_workers
        self.paired = paired
        
        # Kiểm tra thư mục dữ liệu
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {self.data_dir}")
        
        for subset in ['train', 'val', 'test']:
            subset_dir = self.data_dir / subset
            if not os.path.exists(subset_dir):
                os.makedirs(subset_dir, exist_ok=True)
                print(f"Đã tạo thư mục con {subset_dir}")
        
        # Định nghĩa các biến đổi
        self.setup_transforms()
    
    def setup_transforms(self):
        """Thiết lập các biến đổi cho dữ liệu"""
        # Transform cho dữ liệu huấn luyện
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0),
        ])
        
        # Transform cho dữ liệu validation và test (không augment)
        self.val_transform = None
    
    def train_dataloader(self):
        """Tạo DataLoader cho tập huấn luyện"""
        try:
            train_dataset = MRIToCTDataset(
                self.data_dir,
                mode='train',
                slice_axis=self.slice_axis,
                slice_range=self.slice_range,
                transform=self.train_transform,
                paired=self.paired
            )
            return DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=(self.num_workers > 0)
            )
        except Exception as e:
            print(f"Lỗi khi tạo train_dataloader: {e}")
            return None
    
    def val_dataloader(self):
        """Tạo DataLoader cho tập validation"""
        try:
            val_dataset = MRIToCTDataset(
                self.data_dir,
                mode='val',
                slice_axis=self.slice_axis,
                slice_range=self.slice_range,
                transform=self.val_transform,
                paired=self.paired
            )
            return DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=(self.num_workers > 0)
            )
        except Exception as e:
            print(f"Lỗi khi tạo val_dataloader: {e}")
            return None
    
    def test_dataloader(self):
        """Tạo DataLoader cho tập test"""
        try:
            test_dataset = MRIToCTDataset(
                self.data_dir,
                mode='test',
                slice_axis=self.slice_axis,
                slice_range=self.slice_range,
                transform=self.val_transform,
                paired=self.paired
            )
            return DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=(self.num_workers > 0)
            )
        except Exception as e:
            print(f"Lỗi khi tạo test_dataloader: {e}")
            return None 