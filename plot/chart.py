import re
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

# Danh sách các file log cần xử lý
# Bạn có thể thêm bớt tên file tại đây
LOG_FILES = {
    'Alpha = 1': 'alpha1.txt',
    'Alpha = 2': 'alpha2.txt',
    'Alpha = 3': 'alpha3.txt',
    'Alpha = 4': 'alpha4.txt',
    'Alpha = 5': 'alpha5.txt'
}

def parse_log_file(filename):
    """Đọc file log và trích xuất metrics."""
    data_list = []
    
    if not os.path.exists(filename):
        print(f"⚠️ Warning: File {filename} không tồn tại. Bỏ qua.")
        return pd.DataFrame()

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Regex để bắt dòng log chứa kết quả evaluate
    # Mẫu log: [ TIMESTAMP | node_X | INFO ] 📈 Evaluated. Results: {'test_loss': ...}
    log_pattern = re.compile(r'\[ (.*?) \| (.*?) \| INFO \] 📈 Evaluated. Results: (\{.*\})')
    
    for line in lines:
        match = log_pattern.search(line)
        if match:
            timestamp_str = match.group(1)
            node_id = match.group(2)
            json_str = match.group(3)
            
            try:
                # Chuyển chuỗi metrics (dạng dict python) thành dict thật
                metrics = ast.literal_eval(json_str)
                # Làm sạch key (xóa ký tự xuống dòng nếu có)
                clean_metrics = {k.strip(): v for k, v in metrics.items()}
                
                entry = {
                    'timestamp': pd.to_datetime(timestamp_str),
                    'node_id': node_id,
                    'accuracy': clean_metrics.get('test_acc'),
                    'loss': clean_metrics.get('test_loss')
                }
                data_list.append(entry)
            except Exception as e:
                continue
                
    df = pd.DataFrame(data_list)
    return df

def assign_rounds(df):
    """Gán số thứ tự vòng (Round) dựa trên thời gian cho từng node."""
    if df.empty:
        return df
    # Sắp xếp theo node và thời gian
    df = df.sort_values(by=['node_id', 'timestamp'])
    # Đánh số thứ tự lần xuất hiện log cho mỗi node -> đó chính là Round
    df['round'] = df.groupby('node_id').cumcount() + 1
    return df

def main():
    all_data_list = []

    print("🔄 Đang xử lý dữ liệu log...")
    
    for label, filename in LOG_FILES.items():
        print(f"   - Đọc file: {filename} ({label})")
        df = parse_log_file(filename)
        
        if not df.empty:
            df = assign_rounds(df)
            df['run_label'] = label # Gán nhãn để phân biệt (Alpha 1, 2...)
            all_data_list.append(df)

    if not all_data_list:
        print("❌ Không tìm thấy dữ liệu hợp lệ nào.")
        return

    # Gộp tất cả dữ liệu
    all_data = pd.concat(all_data_list)

    # Tính trung bình Accuracy và Loss của tất cả các node trong mỗi vòng
    summary = all_data.groupby(['run_label', 'round']).agg({
        'accuracy': 'mean',
        'loss': 'mean'
    }).reset_index()

    # --- VẼ BIỂU ĐỒ ---
    print("📊 Đang vẽ biểu đồ...")
    
    # Tạo hình vẽ với 2 biểu đồ con (subplot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Lấy danh sách các label để vẽ theo thứ tự
    labels = sorted(LOG_FILES.keys())

    # 1. Biểu đồ Accuracy
    for label in labels:
        subset = summary[summary['run_label'] == label]
        if not subset.empty:
            ax1.plot(subset['round'], subset['accuracy'], label=label, linewidth=2)
    
    ax1.set_title('Độ Chính Xác Trung Bình (Average Accuracy)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Vòng (Round)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 2. Biểu đồ Loss
    for label in labels:
        subset = summary[summary['run_label'] == label]
        if not subset.empty:
            ax2.plot(subset['round'], subset['loss'], label=label, linewidth=2)

    ax2.set_title('Hàm Mất Mát Trung Bình (Average Loss)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Vòng (Round)', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    
    # Lưu file ảnh
    output_file = 'comparison_chart_alpha.png'
    plt.savefig(output_file, dpi=300)
    print(f"✅ Đã lưu biểu đồ thành công: {output_file}")
    
    # Hiển thị (nếu chạy trên Jupyter hoặc môi trường có GUI)
    plt.show()

if __name__ == "__main__":
    main()