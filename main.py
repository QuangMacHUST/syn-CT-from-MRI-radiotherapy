#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Chuyển đổi MRI sang CT mô phỏng cho xạ trị')
    subparsers = parser.add_subparsers(dest='command', help='Các lệnh', required=True)
    
    # Lệnh tiền xử lý dữ liệu
    preprocess_parser = subparsers.add_parser('preprocess', help='Tiền xử lý dữ liệu MRI và CT')
    preprocess_parser.add_argument('--mri_dir', type=str, required=True, help='Thư mục chứa dữ liệu MRI')
    preprocess_parser.add_argument('--ct_dir', type=str, help='Thư mục chứa dữ liệu CT (nếu có)')
    preprocess_parser.add_argument('--output_dir', type=str, default='data/processed', help='Thư mục đầu ra')
    preprocess_parser.add_argument('--paired', action='store_true', help='Dữ liệu MRI và CT đã ghép cặp')
    preprocess_parser.add_argument('--config', type=str, default='configs/default.yaml', help='File cấu hình')
    
    # Lệnh huấn luyện mô hình
    train_parser = subparsers.add_parser('train', help='Huấn luyện mô hình CycleGAN')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml', help='File cấu hình')
    train_parser.add_argument('--resume', type=str, help='Đường dẫn đến checkpoint để tiếp tục huấn luyện')
    
    # Lệnh đánh giá mô hình
    eval_parser = subparsers.add_parser('evaluate', help='Đánh giá hiệu quả mô hình')
    eval_parser.add_argument('--model', type=str, required=True, help='Đường dẫn đến mô hình đã huấn luyện')
    eval_parser.add_argument('--data_dir', type=str, default='data/processed/test', help='Thư mục chứa dữ liệu test')
    eval_parser.add_argument('--output_dir', type=str, default='data/output/evaluation', help='Thư mục đầu ra')
    eval_parser.add_argument('--config', type=str, default='configs/default.yaml', help='File cấu hình')
    eval_parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Thiết bị tính toán')
    
    # Lệnh chuyển đổi MRI sang CT
    infer_parser = subparsers.add_parser('infer', help='Sinh ảnh CT mô phỏng từ ảnh MRI')
    infer_parser.add_argument('--input', type=str, required=True, help='Đường dẫn đến dữ liệu MRI đầu vào')
    infer_parser.add_argument('--output', type=str, default='data/output/synthetic_ct', help='Thư mục đầu ra')
    infer_parser.add_argument('--model', type=str, default='data/output/models/checkpoints/best_model.pth', help='Đường dẫn đến mô hình')
    infer_parser.add_argument('--config', type=str, default='configs/default.yaml', help='File cấu hình')
    infer_parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Thiết bị tính toán')
    infer_parser.add_argument('--save_dicom', action='store_true', help='Lưu kết quả dưới dạng DICOM')
    
    return parser.parse_args()

def load_config(config_path):
    """Tải cấu hình từ file YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    
    # Đảm bảo các thư mục cần thiết tồn tại
    os.makedirs('data/raw/mri', exist_ok=True)
    os.makedirs('data/raw/ct', exist_ok=True)
    os.makedirs('data/processed/train', exist_ok=True)
    os.makedirs('data/processed/val', exist_ok=True)
    os.makedirs('data/processed/test', exist_ok=True)
    os.makedirs('data/output/models/checkpoints', exist_ok=True)
    os.makedirs('data/output/synthetic_ct', exist_ok=True)
    
    if args.command == 'preprocess':
        from src.data_processing.preprocess import preprocess_and_save
        
        # Tiền xử lý dữ liệu
        config = load_config(args.config)
        print(f"Tiền xử lý dữ liệu MRI và CT từ {args.mri_dir} và {args.ct_dir if args.ct_dir else 'không có CT'}")
        preprocess_and_save(
            mri_dir=args.mri_dir,
            ct_dir=args.ct_dir,
            output_dir=args.output_dir,
            paired=args.paired
        )
    
    elif args.command == 'train':
        from src.training.train import CycleGANTrainer, load_config
        
        # Huấn luyện mô hình
        config = load_config(args.config)
        print(f"Huấn luyện mô hình với cấu hình từ {args.config}")
        
        # Khởi tạo trainer
        trainer = CycleGANTrainer(config, resume_path=args.resume)
        
        # Huấn luyện mô hình
        trainer.train()
    
    elif args.command == 'evaluate':
        from src.evaluation.evaluate import evaluate_model, generate_evaluation_report
        from src.models import CycleGANModel
        from src.data_processing.dataset import MRIToCTDataModule
        
        # Đánh giá mô hình
        config = load_config(args.config)
        print(f"Đánh giá mô hình {args.model} trên dữ liệu {args.data_dir}")
        
        # Chọn thiết bị
        device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        print(f"Sử dụng thiết bị: {device}")
        
        # Tải mô hình
        checkpoint = torch.load(args.model, map_location=device)
        model = CycleGANModel().to(device)
        model.netG_A.load_state_dict(checkpoint['netG_A'])
        model.eval()
        
        # Tạo data loader cho tập test
        data_module = MRIToCTDataModule(
            data_dir=args.data_dir,
            batch_size=config['evaluation']['test_batch_size'],
            slice_axis=config['data']['slice_axis'],
            paired=True
        )
        test_loader = data_module.test_dataloader()
        
        # Đánh giá mô hình
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        results = evaluate_model(model, test_loader, device, config, output_dir)
        generate_evaluation_report(results, output_dir)
    
    elif args.command == 'infer':
        from src.inference import process_mri_volume, assign_tissue_densities, save_result
        from src.data_processing.preprocess import read_dicom_series, normalize_image
        from src.models import CycleGANModel
        
        # Chuyển đổi MRI sang CT
        config = load_config(args.config)
        print(f"Chuyển đổi MRI từ {args.input} sang CT mô phỏng")
        
        # Chọn thiết bị
        device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        print(f"Sử dụng thiết bị: {device}")
        
        # Đọc dữ liệu MRI
        print(f"Đọc dữ liệu MRI từ: {args.input}")
        mri_data, metadata = read_dicom_series(args.input)
        mri_data_norm = normalize_image(mri_data, 'mri')
        
        # Tải mô hình
        print(f"Tải mô hình từ: {args.model}")
        checkpoint = torch.load(args.model, map_location=device)
        model = CycleGANModel().to(device)
        model.netG_A.load_state_dict(checkpoint['netG_A'])
        model.eval()
        
        # Chuyển đổi MRI sang CT
        print("Chuyển đổi MRI sang CT mô phỏng...")
        synth_ct_data = process_mri_volume(
            mri_data_norm, 
            model, 
            device, 
            batch_size=config['inference']['batch_size'], 
            slice_axis=config['data']['slice_axis']
        )
        
        # Gán hệ số mô
        print("Gán hệ số mô...")
        synth_ct_data_hu = assign_tissue_densities(synth_ct_data, config)
        
        # Lưu kết quả
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        print(f"Lưu kết quả vào: {output_dir}")
        save_result(synth_ct_data_hu, metadata, output_dir, save_dicom=args.save_dicom)
        
        print("Hoàn thành chuyển đổi!")

if __name__ == "__main__":
    main() 