#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import skimage.metrics
from functools import partial

def calculate_metrics(pred, gt, mask=None):
    """
    Tính toán các chỉ số MAE, RMSE, PSNR, SSIM
    
    Args:
        pred (numpy.ndarray): Ảnh dự đoán (CT mô phỏng)
        gt (numpy.ndarray): Ảnh ground truth (CT thật)
        mask (numpy.ndarray, optional): Mask để tính toán metrics trên vùng cụ thể
        
    Returns:
        dict: Dictionary chứa các chỉ số đánh giá
    """
    # Chuyển tensor thành numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor) and mask is not None:
        mask = mask.detach().cpu().numpy()
    
    # Đảm bảo data là 2D
    if pred.ndim > 2:
        pred = pred.squeeze()
    if gt.ndim > 2:
        gt = gt.squeeze()
    if mask is not None and mask.ndim > 2:
        mask = mask.squeeze()
    
    # Áp dụng mask nếu có
    if mask is not None:
        pred = pred[mask > 0]
        gt = gt[mask > 0]
    
    # Tính MAE
    mae = np.mean(np.abs(pred - gt))
    
    # Tính RMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    
    # Tính PSNR (cần chuẩn hóa ảnh về khoảng [0, 1] cho skimage)
    # Đối với ảnh CT (HU), ta thường chuyển từ [-1000, 3000] sang [0, 1]
    pred_normalized = (pred - pred.min()) / (pred.max() - pred.min())
    gt_normalized = (gt - gt.min()) / (gt.max() - gt.min())
    
    try:
        psnr_value = skimage.metrics.peak_signal_noise_ratio(gt_normalized, pred_normalized)
    except:
        psnr_value = 0
    
    # Tính SSIM
    try:
        ssim_value = skimage.metrics.structural_similarity(
            pred_normalized, 
            gt_normalized,
            data_range=1.0
        )
    except:
        ssim_value = 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'psnr': psnr_value,
        'ssim': ssim_value
    }

def calculate_tissue_metrics(pred, gt, tissue_masks):
    """
    Tính toán chỉ số MAE cho từng loại mô
    
    Args:
        pred (numpy.ndarray): Ảnh dự đoán (CT mô phỏng)
        gt (numpy.ndarray): Ảnh ground truth (CT thật)
        tissue_masks (dict): Dictionary chứa các mask cho từng loại mô
        
    Returns:
        dict: Dictionary chứa MAE cho từng loại mô
    """
    metrics = {}
    
    for tissue_name, mask in tissue_masks.items():
        tissue_metrics = calculate_metrics(pred, gt, mask)
        metrics[f"{tissue_name}_mae"] = tissue_metrics['mae']
        metrics[f"{tissue_name}_rmse"] = tissue_metrics['rmse']
    
    return metrics

def calculate_hu_metrics(pred_hu, gt_hu, hu_ranges):
    """
    Tính toán chỉ số MAE trong các khoảng HU khác nhau
    
    Args:
        pred_hu (numpy.ndarray): Ảnh dự đoán (CT mô phỏng) đơn vị HU
        gt_hu (numpy.ndarray): Ảnh ground truth (CT thật) đơn vị HU
        hu_ranges (list): Danh sách các khoảng HU, ví dụ: [(-1000, -100), (-100, 100)]
        
    Returns:
        dict: Dictionary chứa MAE cho từng khoảng HU
    """
    metrics = {}
    
    for i, (min_hu, max_hu) in enumerate(hu_ranges):
        # Tạo mask cho khoảng HU
        mask = (gt_hu >= min_hu) & (gt_hu <= max_hu)
        
        # Nếu không có pixel nào trong khoảng, bỏ qua
        if np.sum(mask) == 0:
            metrics[f"hu_range_{min_hu}_{max_hu}_mae"] = 0
            continue
        
        # Tính metrics trên mask
        range_metrics = calculate_metrics(pred_hu, gt_hu, mask)
        metrics[f"hu_range_{min_hu}_{max_hu}_mae"] = range_metrics['mae']
    
    return metrics 