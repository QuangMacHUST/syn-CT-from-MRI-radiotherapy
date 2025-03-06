#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error

def calculate_metrics(pred, gt, mask=None):
    """
    Tính toán các metrics đánh giá chất lượng chuyển đổi
    
    Args:
        pred (numpy.ndarray): Ảnh dự đoán
        gt (numpy.ndarray): Ảnh ground truth
        mask (numpy.ndarray, optional): Mặt nạ cho vùng quan tâm
    
    Returns:
        dict: Dictionary chứa các metrics
    """
    # Chắc chắn rằng pred và gt có cùng shape
    assert pred.shape == gt.shape, f"Shapes don't match: {pred.shape} vs {gt.shape}"
    
    # Áp dụng mask nếu có
    if mask is not None:
        assert mask.shape == pred.shape, f"Mask shape {mask.shape} doesn't match pred shape {pred.shape}"
        pred = pred * mask
        gt = gt * mask
    
    # Tính Mean Absolute Error (MAE)
    mae = np.mean(np.abs(pred - gt))
    
    # Tính Mean Squared Error (MSE)
    mse = mean_squared_error(gt, pred)
    
    # Tính Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Tính Peak Signal-to-Noise Ratio (PSNR)
    try:
        psnr_val = psnr(gt, pred, data_range=gt.max() - gt.min())
    except:
        psnr_val = 0
    
    # Tính Structural Similarity Index (SSIM)
    try:
        ssim_val = ssim(gt, pred, data_range=gt.max() - gt.min(), multichannel=False)
    except:
        ssim_val = 0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'psnr': psnr_val,
        'ssim': ssim_val
    }

def calculate_tissue_metrics(pred, gt, tissue_mask):
    """
    Tính toán metrics cho các loại mô khác nhau
    
    Args:
        pred (numpy.ndarray): Ảnh dự đoán
        gt (numpy.ndarray): Ảnh ground truth
        tissue_mask (dict): Dictionary chứa các mặt nạ cho từng loại mô
                            Ví dụ: {'bone': bone_mask, 'soft_tissue': soft_mask, ...}
    
    Returns:
        dict: Dictionary chứa các metrics cho từng loại mô
    """
    results = {}
    
    for tissue_name, mask in tissue_mask.items():
        # Tính metrics cho từng loại mô
        tissue_metrics = calculate_metrics(pred, gt, mask)
        
        # Lưu kết quả với tên mô
        for metric_name, value in tissue_metrics.items():
            results[f"{tissue_name}_{metric_name}"] = value
    
    return results

def calculate_hu_metrics(pred_hu, gt_hu, hu_ranges):
    """
    Tính toán sai số trong các khoảng HU khác nhau
    
    Args:
        pred_hu (numpy.ndarray): Giá trị HU dự đoán
        gt_hu (numpy.ndarray): Giá trị HU ground truth
        hu_ranges (list): Danh sách các khoảng HU cần tính
                          Ví dụ: [(-1000, -100), (-100, 100), (100, 1000)]
    
    Returns:
        dict: Dictionary chứa các metrics cho các khoảng HU
    """
    results = {}
    
    for i, (lower, upper) in enumerate(hu_ranges):
        # Tạo mask cho khoảng HU
        mask = (gt_hu >= lower) & (gt_hu <= upper)
        
        # Bỏ qua nếu không có pixel nào trong khoảng
        if np.sum(mask) == 0:
            continue
        
        # Tính MAE cho khoảng HU
        mae = np.mean(np.abs(pred_hu[mask] - gt_hu[mask]))
        
        # Lưu kết quả
        results[f"hu_range_{lower}_{upper}_mae"] = mae
    
    return results 