from data_process import load_dataset, compute_label_aggregations, select_data, preprocess_signals
import os

# 数据集路径
data_path = '/root/autodl-tmp/data/ptbxl'  # 确保路径正确
print(f"数据集路径: {data_path}")

# 检查路径是否存在
if not os.path.exists(data_path):
    raise FileNotFoundError(f"路径不存在: {data_path}")

try:
    # 加载数据集
    data, raw_labels = load_dataset(data_path, sampling_rate=100)
    print("数据集加载成功！")
    print(f"数据形状: {data.shape}, 标签形状: {raw_labels.shape}")

    # 预处理标签数据
    labels = compute_label_aggregations(raw_labels, data_path, 'all')
    print("标签预处理完成！")

    # 选择数据并转换为 one-hot 编码
    X, Y, y, mlb = select_data(data, labels, 'all', min_samples=0)
    print("数据选择完成！")

    # 预处理信号数据
    num_samples = len(X)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)

    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]

    X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)
    print("信号数据预处理完成！")
    print(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}, 测试集形状: {X_test.shape}")

except Exception as e:
    print(f"运行出错: {e}")
