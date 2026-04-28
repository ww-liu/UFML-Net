import sys
sys.path.append('/root/autodl-tmp/mrm')  # 确保路径正确

try:
    from data_load import load_dataset
    print("load_dataset function found!")
except ImportError as e:
    print(f"Error: {e}")
