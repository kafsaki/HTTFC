import os
import numpy as np

def read_and_check_txt(file_path):
    """
    读取 .txt 文件，返回数据及其形状。
    如果文件为空或格式不符，则返回空数组及其形状。
    """
    try:
        if os.path.getsize(file_path) == 0:  # 检查文件是否为空
            return False
        return True
    except Exception as e:  # 捕获异常和警告
        return False

def pad_and_save_txt(file_path):
    """
    填充文件为 42 行 "0,0" 并保存。
    """
    padded_data = np.zeros((42, 2), dtype=np.uint8)
    np.savetxt(file_path, padded_data, fmt="%d,%d", delimiter=",")

def process_directory(root_dir):
    """
    遍历文件结构，找到形状为 (1, 0) 的 .txt 文件，打印其路径并填充为 42 行 "0,0"。
    """
    filled_count = 0  # 初始化填充计数器
    for subdir, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith(".txt"):  # 只处理 .txt 文件
                file_path = os.path.join(subdir, file_name)
                flag = read_and_check_txt(file_path)
                if(not flag):
                    # print(f"Padding 0,0 in file: {file_path}")
                    pad_and_save_txt(file_path)  # 填充并保存
                    filled_count += 1  # 增加计数
    print(f"Total files padded: {filled_count}")

# 设置根目录路径
root_dir = "data/CE-CSL/mediapipe"

# 执行处理
process_directory(root_dir)
