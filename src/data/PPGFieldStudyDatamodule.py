import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pickle
import os


class PPGDataset(Dataset):
    def __init__(self, data_dir, window_size=8, step_size=2):
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.data = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        # 遍历所有受试者的文件夹
        for subject_folder in os.listdir(self.data_dir):
            subject_path = os.path.join(self.data_dir, subject_folder)
            if os.path.isdir(subject_path):
                pkl_file = os.path.join(subject_path, f"{subject_folder}.pkl")
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)

                # 获取PPG信号和标签
                ppg_signal = data["signal"]["wrist"]["BVP"]
                labels = data["label"]

                # 滑动窗口分割信号和标签
                num_segments = (len(ppg_signal) - self.window_size * 64) // (
                    self.step_size * 64
                ) + 1
                for i in range(num_segments):
                    start_idx = i * self.step_size * 64
                    end_idx = start_idx + self.window_size * 64
                    self.data.append(ppg_signal[start_idx:end_idx])
                    self.labels.append(labels[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class PPGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, window_size=8, step_size=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.window_size = window_size
        self.step_size = step_size

    def setup(self, stage=None):
        # 分别为训练、验证和测试集创建数据集
        self.train_dataset = PPGDataset(
            os.path.join(self.data_dir, "train"), self.window_size, self.step_size
        )
        self.val_dataset = PPGDataset(
            os.path.join(self.data_dir, "val"), self.window_size, self.step_size
        )
        self.test_dataset = PPGDataset(
            os.path.join(self.data_dir, "test"), self.window_size, self.step_size
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
