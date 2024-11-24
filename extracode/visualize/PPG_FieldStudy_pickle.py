import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

DATA_PATH = "./data/PPG_FieldStudy/PPG_FieldStudy/S1/S1.pkl"
RESULT_PATH = "./extracode/visualize/PPG_FieldStudy_pickle"

# 确保结果目录存在
os.makedirs(RESULT_PATH, exist_ok=True)

# 加载数据
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f, encoding="latin1")  # 指定编码为 'latin1'


# 保存树结构到文件的函数
def save_tree(data, file, level=0, parent_key=""):
    prefix = " " * (level * 2)
    for key, value in data.items():
        full_key = f"{parent_key}/{key}" if parent_key else key
        if isinstance(value, dict):
            file.write(f"{prefix}{key}/ (dict)\n")
            save_tree(value, file, level + 1, full_key)
        elif isinstance(value, (list, tuple)):
            file.write(f"{prefix}{key}/ (list/tuple) - length: {len(value)}\n")
        elif isinstance(value, np.ndarray):
            file.write(f"{prefix}{key} (ndarray) - shape: {value.shape}\n")
        else:
            file.write(f"{prefix}{key} ({type(value).__name__})\n")


# 递归遍历并可视化数据，并将图片保存到文件
def visualize_data(data, parent_key=""):
    for key, value in data.items():
        full_key = f"{parent_key}/{key}" if parent_key else key

        if isinstance(value, dict):
            print(f"Exploring '{full_key}'")
            visualize_data(value, full_key)
        elif isinstance(value, (list, tuple)):
            print(f"Visualizing list/tuple '{full_key}' with length {len(value)}")
            plt.figure(figsize=(10, 4))
            plt.plot(value[:1000])  # 只展示前1000个数据点
            plt.title(f"Visualization of {full_key}")
            plt.xlabel("Sample")
            plt.ylabel("Value")
            save_path = os.path.join(RESULT_PATH, f"{full_key.replace('/', '_')}.png")
            plt.savefig(save_path)
            plt.close()
        elif isinstance(value, np.ndarray):
            print(f"Visualizing ndarray '{full_key}' with shape {value.shape}")
            if value.ndim == 1:
                plt.figure(figsize=(10, 4))
                plt.plot(value[:1000])  # 只展示前1000个数据点
                plt.title(f"Visualization of {full_key}")
                plt.xlabel("Sample")
                plt.ylabel("Value")
                save_path = os.path.join(
                    RESULT_PATH, f"{full_key.replace('/', '_')}.png"
                )
                plt.savefig(save_path)
                plt.close()
            elif value.ndim == 2 and value.shape[1] <= 3:
                # 分别绘制每个通道的折线图
                plt.figure(figsize=(10, 4))
                for i in range(value.shape[1]):
                    plt.plot(value[:1000, i], label=f"Channel {i+1}")
                plt.title(f"Line Plot of {full_key}")
                plt.xlabel("Sample")
                plt.ylabel("Value")
                plt.legend()
                save_path = os.path.join(
                    RESULT_PATH, f"{full_key.replace('/', '_')}.png"
                )
                plt.savefig(save_path)
                plt.close()
        elif isinstance(value, (int, float)):
            print(f"Scalar value '{full_key}': {value}")
        elif isinstance(value, str):
            print(f"String value '{full_key}': {value}")
        else:
            print(f"Unsupported data type for '{full_key}': {type(value)}")


# 打印数据结构到文件
tree_path = os.path.join(RESULT_PATH, "data_structure.txt")
with open(tree_path, "w") as file:
    save_tree(data, file)

# 开始可视化
visualize_data(data)
