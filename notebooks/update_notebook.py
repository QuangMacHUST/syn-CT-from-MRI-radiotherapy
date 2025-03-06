import json

# Đọc notebook hiện tại
with open('MRI_to_CT_Tutorial.ipynb', 'r') as f:
    notebook = json.load(f)

# Thêm các ô mới
notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## Nội dung\n1. Thiết lập môi trường\n2. Tải mô hình đã huấn luyện\n3. Chuẩn bị dữ liệu MRI\n4. Chuyển đổi MRI sang CT tổng hợp\n5. Hiển thị kết quả\n6. Phân đoạn mô dựa trên đơn vị Hounsfield (HU)\n7. Lưu kết quả\n8. Đánh giá chất lượng chuyển đổi (tùy chọn)'
})

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## 1. Thiết lập môi trường\n\nĐầu tiên, chúng ta cần cài đặt và nhập các thư viện cần thiết.'
})

notebook['cells'].append({
    'cell_type': 'code',
    'metadata': {},
    'source': '# Cài đặt các thư viện cần thiết (nếu chưa có)\n# !pip install torch torchvision numpy matplotlib pydicom nibabel SimpleITK scikit-image pyyaml\n\nimport os\nimport sys\nimport yaml\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nfrom pathlib import Path\n\n# Thêm thư mục gốc vào đường dẫn để có thể nhập các module\nsys.path.append(\'..\')\n\n# Nhập các module cần thiết từ dự án\nfrom models.cycle_gan import CycleGANModel\nfrom utils.data_utils import load_dicom_series, save_dicom_series, normalize_image\nfrom utils.visualization import display_slices, compare_images\nfrom utils.config import load_config\n\n# Kiểm tra GPU\ndevice = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')\nprint(f"Sử dụng thiết bị: {device}")',
    'execution_count': None,
    'outputs': []
})

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## 2. Tải cấu hình và mô hình đã huấn luyện\n\nTiếp theo, chúng ta sẽ tải cấu hình và mô hình CycleGAN đã được huấn luyện.'
})

notebook['cells'].append({
    'cell_type': 'code',
    'metadata': {},
    'source': '# Tải cấu hình\nconfig_path = \'../configs/default.yaml\'\nconfig = load_config(config_path)\n\n# Đường dẫn đến checkpoint mô hình\ncheckpoint_path = \'../data/output/models/checkpoints/best_model.pth\'\n\n# Khởi tạo mô hình\nmodel = CycleGANModel(\n    input_channels=config[\'model\'][\'input_channels\'],\n    output_channels=config[\'model\'][\'output_channels\'],\n    generator_filters=config[\'model\'][\'generator_filters\'],\n    discriminator_filters=config[\'model\'][\'discriminator_filters\'],\n    n_residual_blocks=config[\'model\'][\'n_residual_blocks\']\n)\n\n# Tải trọng số mô hình\nif os.path.exists(checkpoint_path):\n    checkpoint = torch.load(checkpoint_path, map_location=device)\n    model.load_state_dict(checkpoint[\'model_state_dict\'])\n    print(f"Đã tải mô hình từ epoch {checkpoint[\'epoch\']}")\nelse:\n    print(f"Không tìm thấy checkpoint tại {checkpoint_path}")\n    print("Vui lòng huấn luyện mô hình trước hoặc tải mô hình đã huấn luyện")\n\nmodel.to(device)\nmodel.eval()  # Đặt mô hình ở chế độ đánh giá',
    'execution_count': None,
    'outputs': []
})

# Lưu notebook đã cập nhật
with open('MRI_to_CT_Tutorial.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 