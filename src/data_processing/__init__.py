"""
Module xử lý dữ liệu MRI và CT.

Bao gồm các hàm và lớp để xử lý dữ liệu DICOM, chuẩn hóa ảnh và tạo datasets.
"""

from .preprocess import read_dicom_series, normalize_image, preprocess_and_save
from .dataset import MRIToCTDataset, MRIToCTDataModule 