import re
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

# Danh sách các file log cần xử lý
# Bạn có thể thêm bớt tên file tại đây
LOG_FILES = {
    'Alpha = 1': ['alpha1.txt', 'alpha1.txt.1'],
    'Alpha = 2': ['alpha2.txt', 'alpha2.txt.1'],
    'Alpha = 3': ['alpha3.txt', 'alpha3.txt.1'],
    'Alpha = 4': ['alpha4.txt', 'alpha4.txt.1'],
    'Alpha = 5': ['alpha5.txt', 'alpha5.txt.1'],
}

def parse_log_file(filename):
    """Đọc file log và trích xuất metrics."""
    data_list = []
    
    if not os.path.exists(filename):
        print(f"⚠️ Warning: File '{filename}' không tồn tại. Bỏ qua.")
        return pd.DataFrame()

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    log_pattern = re.compile(r'\[ (.*?) \| (.*?) \| INFO \] 📈 Evaluated. Results: (\{.*\})')
    
    for line in lines:
        match = log_pattern.search(line)
        if match:
            timestamp_str = match.group(1)
            node_id = match.group(2)
            json_str = match.group(3)
            
            try:
                metrics = ast.literal_eval(json_str)
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
    # Sắp xếp theo node và thời gian (quan trọng khi gộp nhiều file)
    df = df.sort_values(by=['node_id', 'timestamp'])
    # Đánh số thứ tự lần xuất hiện log cho mỗi node -> đó chính là Round
    df['round'] = df.groupby('node_id').cumcount() + 1
    return df

def main():
    all_data_list = []

    print("🔄 Đang xử lý dữ liệu log...")
    
    # Duyệt qua từng Label và danh sách file tương ứng
    for label, files in LOG_FILES.items():
        # Đảm bảo 'files' luôn là một list (đề phòng người dùng nhập string đơn lẻ)
        if isinstance(files, str):
            files = [files]
            
        print(f" 📂 Đang xử lý nhóm: {label}")
        
        # List tạm chứa dữ liệu của các file thuộc cùng 1 label
        current_label_dfs = []
        
        for filename in files:
            print(f"    - Đọc file: {filename}")
            df_part = parse_log_file(filename)
            if not df_part.empty:
                current_label_dfs.append(df_part)
        
        # Nếu nhóm này có dữ liệu
        if current_label_dfs:
            # 1. Gộp tất cả các file của label này lại
            full_df_label = pd.concat(current_label_dfs)
            
            # 2. Tính toán Round (quan trọng: phải gộp xong mới tính round để liền mạch thời gian)
            full_df_label = assign_rounds(full_df_label)
            
            # 3. Gán nhãn
            full_df_label['run_label'] = label
            
            # 4. Thêm vào danh sách tổng
            all_data_list.append(full_df_label)
        else:
            print(f"    ⚠️ Không có dữ liệu hợp lệ trong nhóm {label}")

    if not all_data_list:
        print("❌ Không tìm thấy dữ liệu hợp lệ nào.")
        return

    # Gộp tất cả dữ liệu tổng
    all_data = pd.concat(all_data_list)

    # Tính trung bình Accuracy và Loss
    summary = all_data.groupby(['run_label', 'round']).agg({
        'accuracy': 'mean',
        'loss': 'mean'
    }).reset_index()

    # --- VẼ BIỂU ĐỒ ---
    print("📊 Đang vẽ biểu đồ...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    labels = sorted(LOG_FILES.keys())

    # 1. Biểu đồ Accuracy
    for label in labels:
        subset = summary[summary['run_label'] == label]
        if not subset.empty:
            ax1.plot(subset['round'], subset['accuracy'], label=label, linewidth=2, marker='o', markersize=3)
    
    ax1.set_title('Độ Chính Xác Trung Bình (Average Accuracy)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Vòng (Round)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 2. Biểu đồ Loss
    for label in labels:
        subset = summary[summary['run_label'] == label]
        if not subset.empty:
            ax2.plot(subset['round'], subset['loss'], label=label, linewidth=2, marker='s', markersize=3)

    ax2.set_title('Hàm Mất Mát Trung Bình (Average Loss)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Vòng (Round)', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    output_file = 'comparison_chart_alpha.png'
    plt.savefig(output_file, dpi=300)
    print(f"✅ Đã lưu biểu đồ thành công: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()