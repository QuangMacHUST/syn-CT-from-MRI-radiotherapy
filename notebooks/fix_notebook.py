import json

# Đọc notebook hiện tại
with open('MRI_to_CT_Tutorial.ipynb', 'r') as f:
    notebook = json.load(f)

# Loại bỏ các ô trùng lặp
unique_cells = []
seen_sources = set()

for cell in notebook['cells']:
    source = cell.get('source', '')
    if source not in seen_sources:
        seen_sources.add(source)
        unique_cells.append(cell)

notebook['cells'] = unique_cells

# Thêm các ô mới
notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## 3. Chuẩn bị dữ liệu MRI\n\nBây giờ chúng ta sẽ tải và chuẩn bị dữ liệu MRI để chuyển đổi.'
})

notebook['cells'].append({
    'cell_type': 'code',
    'metadata': {},
    'source': '# Đường dẫn đến thư mục chứa ảnh MRI DICOM\nmri_dir = \'../data/test/mri\'\n\n# Tải dữ liệu MRI\nmri_volume, mri_metadata = load_dicom_series(mri_dir)\nprint(f"Kích thước khối MRI: {mri_volume.shape}")\n\n# Chuẩn hóa dữ liệu MRI\nmri_normalized = normalize_image(mri_volume)\n\n# Hiển thị một số lát cắt MRI\nmiddle_slice = mri_volume.shape[0] // 2\ndisplay_slices(mri_normalized, start_slice=middle_slice-5, end_slice=middle_slice+6, step=2, \n               title="Lát cắt MRI", cmap="gray")',
    'execution_count': None,
    'outputs': []
})

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## 4. Chuyển đổi MRI sang CT tổng hợp\n\nSử dụng mô hình CycleGAN để chuyển đổi ảnh MRI sang ảnh CT tổng hợp.'
})

notebook['cells'].append({
    'cell_type': 'code',
    'metadata': {},
    'source': '# Hàm để xử lý từng lát cắt và tạo CT tổng hợp\ndef generate_synthetic_ct(mri_volume, model, device, batch_size=4):\n    model.eval()\n    synthetic_ct = np.zeros_like(mri_volume)\n    \n    with torch.no_grad():\n        for i in range(0, mri_volume.shape[0], batch_size):\n            batch_end = min(i + batch_size, mri_volume.shape[0])\n            batch = mri_volume[i:batch_end]\n            \n            # Chuyển đổi sang tensor và thêm kênh\n            batch_tensor = torch.from_numpy(batch).float().unsqueeze(1).to(device)\n            \n            # Tạo CT tổng hợp\n            synthetic_batch = model.generator_MRI_to_CT(batch_tensor)\n            \n            # Chuyển về numpy và loại bỏ kênh\n            synthetic_batch = synthetic_batch.cpu().numpy().squeeze(1)\n            synthetic_ct[i:batch_end] = synthetic_batch\n            \n            print(f"Đã xử lý {batch_end}/{mri_volume.shape[0]} lát cắt", end=\'\\r\')\n    \n    print(f"\\nĐã hoàn thành chuyển đổi {mri_volume.shape[0]} lát cắt MRI sang CT tổng hợp")\n    return synthetic_ct\n\n# Tạo CT tổng hợp từ MRI\nsynthetic_ct = generate_synthetic_ct(mri_normalized, model, device)\n\n# Chuyển đổi giá trị pixel sang đơn vị Hounsfield (HU)\n# Giả sử mô hình tạo ra giá trị trong khoảng [-1, 1] cần được ánh xạ sang khoảng HU thích hợp\nhu_min, hu_max = -1000, 3000  # Khoảng HU điển hình\nsynthetic_ct_hu = (synthetic_ct + 1) / 2 * (hu_max - hu_min) + hu_min',
    'execution_count': None,
    'outputs': []
})

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## 5. Hiển thị kết quả\n\nSo sánh ảnh MRI gốc với ảnh CT tổng hợp.'
})

notebook['cells'].append({
    'cell_type': 'code',
    'metadata': {},
    'source': '# Hiển thị một số lát cắt CT tổng hợp\nmiddle_slice = synthetic_ct_hu.shape[0] // 2\ndisplay_slices(synthetic_ct_hu, start_slice=middle_slice-5, end_slice=middle_slice+6, step=2, \n               title="Lát cắt CT tổng hợp", cmap="gray", vmin=-200, vmax=400)\n\n# So sánh MRI và CT tổng hợp\nfor slice_idx in range(middle_slice-4, middle_slice+5, 2):\n    compare_images(\n        mri_normalized[slice_idx], synthetic_ct_hu[slice_idx],\n        titles=[f"MRI (lát cắt {slice_idx})", f"CT tổng hợp (lát cắt {slice_idx})"],\n        cmaps=["gray", "gray"],\n        vmin_vmax=[(None, None), (-200, 400)]\n    )',
    'execution_count': None,
    'outputs': []
})

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## 6. Phân đoạn mô dựa trên đơn vị Hounsfield (HU)\n\nPhân đoạn các loại mô khác nhau dựa trên giá trị HU của chúng.'
})

notebook['cells'].append({
    'cell_type': 'code',
    'metadata': {},
    'source': '# Định nghĩa các khoảng HU cho các loại mô khác nhau\ntissue_ranges = {\n    \'Không khí\': (-1000, -950),\n    \'Phổi\': (-950, -700),\n    \'Mỡ\': (-700, -100),\n    \'Nước/Mô mềm\': (-100, 100),\n    \'Xương\': (100, 3000)\n}\n\n# Tạo mặt nạ cho từng loại mô\ntissue_masks = {}\nfor tissue, (min_hu, max_hu) in tissue_ranges.items():\n    mask = (synthetic_ct_hu >= min_hu) & (synthetic_ct_hu < max_hu)\n    tissue_masks[tissue] = mask\n\n# Hiển thị phân đoạn mô cho một lát cắt\ndef display_tissue_segmentation(ct_slice, masks, slice_idx):\n    fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n    axes = axes.flatten()\n    \n    # Hiển thị CT gốc\n    axes[0].imshow(ct_slice, cmap=\'gray\', vmin=-200, vmax=400)\n    axes[0].set_title(f"CT tổng hợp (lát cắt {slice_idx})")\n    axes[0].axis(\'off\')\n    \n    # Hiển thị từng loại mô\n    for i, (tissue, mask) in enumerate(masks.items(), 1):\n        if i < len(axes):\n            axes[i].imshow(mask[slice_idx], cmap=\'viridis\')\n            axes[i].set_title(f"{tissue}")\n            axes[i].axis(\'off\')\n    \n    plt.tight_layout()\n    plt.show()\n\n# Hiển thị phân đoạn mô cho lát cắt giữa\ndisplay_tissue_segmentation(synthetic_ct_hu[middle_slice], tissue_masks, middle_slice)',
    'execution_count': None,
    'outputs': []
})

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## 7. Lưu kết quả\n\nLưu ảnh CT tổng hợp dưới dạng tệp DICOM.'
})

notebook['cells'].append({
    'cell_type': 'code',
    'metadata': {},
    'source': '# Tạo thư mục đầu ra\noutput_dir = \'../data/output/notebook_demo\'\nos.makedirs(output_dir, exist_ok=True)\n\n# Lưu CT tổng hợp dưới dạng tệp DICOM\nsave_dicom_series(\n    synthetic_ct_hu,\n    output_dir,\n    reference_metadata=mri_metadata,\n    modality=\'CT\',\n    series_description=\'Synthetic CT from MRI\',\n    window_center=40,  # Giá trị cửa sổ phù hợp cho CT\n    window_width=400   # Độ rộng cửa sổ phù hợp cho CT\n)\n\nprint(f"Đã lưu CT tổng hợp tại: {output_dir}")',
    'execution_count': None,
    'outputs': []
})

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## 8. Đánh giá chất lượng chuyển đổi (tùy chọn)\n\nNếu có sẵn dữ liệu CT thực, chúng ta có thể đánh giá chất lượng của CT tổng hợp.'
})

notebook['cells'].append({
    'cell_type': 'code',
    'metadata': {},
    'source': '# Kiểm tra xem có dữ liệu CT thực hay không\nreal_ct_dir = \'../data/test/ct\'\nif os.path.exists(real_ct_dir):\n    # Tải dữ liệu CT thực\n    real_ct, real_ct_metadata = load_dicom_series(real_ct_dir)\n    print(f"Kích thước khối CT thực: {real_ct.shape}")\n    \n    # Đảm bảo CT thực và CT tổng hợp có cùng kích thước\n    if real_ct.shape == synthetic_ct_hu.shape:\n        # Tính toán các chỉ số đánh giá\n        from utils.metrics import calculate_mae, calculate_psnr, calculate_ssim\n        \n        mae = calculate_mae(real_ct, synthetic_ct_hu)\n        psnr = calculate_psnr(real_ct, synthetic_ct_hu)\n        ssim = calculate_ssim(real_ct, synthetic_ct_hu)\n        \n        print(f"Đánh giá chất lượng CT tổng hợp:")\n        print(f"MAE: {mae:.2f} HU")\n        print(f"PSNR: {psnr:.2f} dB")\n        print(f"SSIM: {ssim:.4f}")\n        \n        # So sánh CT thực và CT tổng hợp\n        for slice_idx in range(middle_slice-4, middle_slice+5, 2):\n            compare_images(\n                real_ct[slice_idx], synthetic_ct_hu[slice_idx],\n                titles=[f"CT thực (lát cắt {slice_idx})", f"CT tổng hợp (lát cắt {slice_idx})"],\n                cmaps=["gray", "gray"],\n                vmin_vmax=[(-200, 400), (-200, 400)]\n            )\n            \n            # Hiển thị bản đồ sai số\n            error_map = np.abs(real_ct[slice_idx] - synthetic_ct_hu[slice_idx])\n            plt.figure(figsize=(8, 6))\n            plt.imshow(error_map, cmap=\'hot\', vmin=0, vmax=200)\n            plt.colorbar(label=\'Sai số tuyệt đối (HU)\')\n            plt.title(f"Bản đồ sai số (lát cắt {slice_idx})")\n            plt.axis(\'off\')\n            plt.show()\n    else:\n        print(f"Kích thước không khớp: CT thực {real_ct.shape} vs CT tổng hợp {synthetic_ct_hu.shape}")\nelse:\n    print(f"Không tìm thấy dữ liệu CT thực tại {real_ct_dir}")\n    print("Bỏ qua bước đánh giá chất lượng")',
    'execution_count': None,
    'outputs': []
})

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## Kết luận\n\nTrong notebook này, chúng ta đã thực hiện các bước sau:\n1. Tải mô hình CycleGAN đã được huấn luyện\n2. Chuẩn bị dữ liệu MRI đầu vào\n3. Chuyển đổi MRI sang CT tổng hợp\n4. Hiển thị và so sánh kết quả\n5. Phân đoạn các loại mô dựa trên giá trị HU\n6. Lưu kết quả dưới dạng tệp DICOM\n7. Đánh giá chất lượng chuyển đổi (nếu có dữ liệu CT thực)\n\nCT tổng hợp từ MRI có thể được sử dụng trong lập kế hoạch xạ trị, giúp giảm liều lượng bức xạ cho bệnh nhân và cải thiện quy trình lâm sàng.'
})

# Lưu notebook đã cập nhật
with open('MRI_to_CT_Tutorial.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 