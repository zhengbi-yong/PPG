import kagglehub
import os
import shutil

# 下载数据集到默认的缓存路径
path = kagglehub.dataset_download("dishantvishwakarma/ppg-dataset-shared")
print("下载完成，数据集路径为:", path)

# 设定目标文件夹路径，将数据集移至该脚本父目录的父目录下的 data 文件夹
target_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

# 如果目标文件夹不存在，创建该文件夹
os.makedirs(target_path, exist_ok=True)

# 将数据集文件逐个复制到目标文件夹
for item in os.listdir(path):
    source_item = os.path.join(path, item)
    target_item = os.path.join(target_path, item)

    # 如果目标文件已存在，跳过复制
    if os.path.exists(target_item):
        print(f"文件已存在，跳过复制: {target_item}")
    else:
        if os.path.isdir(source_item):
            shutil.copytree(source_item, target_item)
        else:
            shutil.copy2(source_item, target_item)

print("数据集已移动到目标路径:", target_path)
