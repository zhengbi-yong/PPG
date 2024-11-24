import os
import pickle
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule


class TorchStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor):
        """
        计算数据的均值和标准差。

        :param data: 输入数据张量，形状为 [时间步, 特征]
        """
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        使用计算得到的均值和标准差对数据进行标准化。

        :param data: 输入数据张量，形状为 [时间步, 特征]
        :return: 标准化后的数据张量
        """
        return (data - self.mean) / self.std

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        计算均值和标准差，并对数据进行标准化。

        :param data: 输入数据张量，形状为 [时间步, 特征]
        :return: 标准化后的数据张量
        """
        self.fit(data)
        return self.transform(data)


class PPGDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, Any],
        labels: torch.Tensor,
        window_size: int = 256,
        stride: int = 128,
        transform: Optional[Any] = None,
    ):
        """
        自定义 PPG 数据集。

        :param data: 包含所有受试者拼接信号数据的字典。
        :param labels: 拼接后的标签张量。
        :param window_size: 每个窗口的时间步数。
        :param stride: 窗口之间的步幅。
        :param transform: 可选的变换操作。
        """
        self.window_size = window_size
        self.stride = stride
        self.transform = transform

        # 假设所有信号已被截断到相同长度
        self.length = data["chest"]["ECG"].shape[0]
        self.num_windows = (self.length - window_size) // stride + 1

        # 验证所有信号长度是否相同
        for location in ["chest", "wrist"]:
            for sensor in data[location]:
                assert data[location][sensor].shape[0] == self.length, (
                    f"信号长度不匹配在 {location}/{sensor}: "
                    f"期望 {self.length}, 但得到 {data[location][sensor].shape[0]}"
                )

        # 验证标签长度
        assert (
            len(labels) >= self.num_windows
        ), f"标签数量 ({len(labels)}) 少于窗口数量 ({self.num_windows})。"

        # 使用 TorchStandardScaler 对信号进行标准化
        self.scalers = {}
        for location in ["chest", "wrist"]:
            self.scalers[location] = {}
            for sensor, signal in data[location].items():
                scaler = TorchStandardScaler()
                if signal.ndimension() > 1:
                    scaler.fit(signal)
                    self.scalers[location][sensor] = scaler
                else:
                    scaler.fit(signal.unsqueeze(1))
                    self.scalers[location][sensor] = scaler

                # 标准化信号
                if signal.ndimension() > 1:
                    data[location][sensor] = scaler.transform(signal)
                else:
                    data[location][sensor] = scaler.transform(
                        signal.unsqueeze(1)
                    ).squeeze()

        self.data = data
        self.labels = labels[
            : self.num_windows
        ]  # Ensure labels match the number of windows

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        """
        获取指定索引的窗口数据和对应标签。

        :param idx: 窗口的索引。
        :return: (特征张量, 标签张量) 的元组
        """
        start = idx * self.stride
        end = start + self.window_size

        features = {}
        for location in ["chest", "wrist"]:
            features[location] = {}
            for sensor, signal in self.data[location].items():
                window = signal[start:end]
                features[location][sensor] = window

        # 将所有传感器的数据沿特征维度拼接
        chest_features = torch.cat(
            [features["chest"][sensor] for sensor in features["chest"]], dim=1
        )  # 形状: [window_size, chest_features]
        wrist_features = torch.cat(
            [features["wrist"][sensor] for sensor in features["wrist"]], dim=1
        )  # 形状: [window_size, wrist_features]
        combined_features = torch.cat(
            [chest_features, wrist_features], dim=1
        )  # 形状: [window_size, total_features]

        if self.transform:
            combined_features = self.transform(combined_features)

        # 保留整个窗口的特征，不进行汇聚
        aggregated_features = combined_features  # 形状: [window_size, total_features]

        # 获取对应的标签
        label = self.labels[idx]

        return aggregated_features, label


class PPGDataModule(LightningDataModule):
    """
    PPGDataModule is a LightningDataModule for handling PPG dataset.
    Attributes:
        data_dir (str): The root directory containing subject folders.
        window_size (int): Number of time steps per window.
        stride (int): Stride between windows.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        train_split (float): Proportion of data for training.
        val_split (float): Proportion of data for validation.
        test_split (float): Proportion of data for testing.
        train_dataset (Optional[Dataset]): Training dataset.
        val_dataset (Optional[Dataset]): Validation dataset.
        test_dataset (Optional[Dataset]): Testing dataset.
    Methods:
        setup(stage: Optional[str] = None):
            Args:
                stage (Optional[str]): Optional stage to setup (fit, validate, test, predict).
        train_dataloader():
            Returns DataLoader for training dataset.
        val_dataloader():
            Returns DataLoader for validation dataset.
        test_dataloader():
            Returns DataLoader for testing dataset.
    """

    def __init__(
        self,
        data_dir: str = "./data/PPG_FieldStudy",
        window_size: int = 256,
        stride: int = 128,
        batch_size: int = 64,
        num_workers: int = 0,  # 调试时设置为 0
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
    ):
        """
        PPG 数据集的 DataModule。

        :param data_dir: 包含受试者文件夹的根目录。
        :param window_size: 每个窗口的时间步数。
        :param stride: 窗口之间的步幅。
        :param batch_size: 批大小。
        :param num_workers: DataLoader 的工作线程数量。
        :param train_split: 训练集的比例。
        :param val_split: 验证集的比例。
        :param test_split: 测试集的比例。
        """
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        加载并预处理所有受试者的数据。

        :param stage: 可选的阶段 (fit, validate, test, predict)。
        """
        # 初始化用于拼接数据的字典
        concatenated_data = {
            "chest": {
                "ACC": [],
                "ECG": [],
                "EMG": [],
                "EDA": [],
                "Temp": [],
                "Resp": [],
            },
            "wrist": {
                "ACC": [],
                "BVP": [],
                "EDA": [],
                "TEMP": [],
            },
        }
        concatenated_labels = []

        # 遍历每个受试者目录 (S1 到 S15)
        for subject_num in range(1, 16):
            subject_dir = os.path.join(self.data_dir, f"S{subject_num}")
            pkl_file = os.path.join(subject_dir, f"S{subject_num}.pkl")

            if not os.path.isfile(pkl_file):
                print(f"Pickle 文件未找到: {pkl_file}. 跳过。")
                continue

            try:
                with open(pkl_file, "rb") as f:
                    subject_data = pickle.load(f, encoding="latin1")
                print(f"成功加载 {pkl_file}.")
            except Exception as e:
                print(f"加载 {pkl_file} 时出错: {e}. 跳过。")
                continue

            # 拼接每个传感器的数据
            for location in ["chest", "wrist"]:
                for sensor, signal in subject_data["signal"][location].items():
                    concatenated_data[location][sensor].append(
                        torch.tensor(signal, dtype=torch.float32)
                    )

            # 拼接标签
            concatenated_labels.append(
                torch.tensor(subject_data["label"], dtype=torch.float32)
            )

        if not concatenated_labels:
            raise ValueError("未加载到任何有效的 Pickle 文件。请检查你的数据。")

        # 拼接所有受试者的信号数据
        for location in ["chest", "wrist"]:
            for sensor in concatenated_data[location]:
                # 沿时间轴拼接
                concatenated_data[location][sensor] = torch.cat(
                    concatenated_data[location][sensor], dim=0
                )
                print(
                    f"传感器 '{location}/{sensor}' 拼接后的形状: {concatenated_data[location][sensor].shape}"
                )

        # 计算所有传感器的最小长度
        lengths = [
            concatenated_data[location][sensor].shape[0]
            for location in ["chest", "wrist"]
            for sensor in concatenated_data[location]
        ]
        min_length = min(lengths)
        print(f"所有传感器的最小信号长度: {min_length}")

        # 将所有信号截断到最小长度
        for location in ["chest", "wrist"]:
            for sensor in concatenated_data[location]:
                original_length = concatenated_data[location][sensor].shape[0]
                concatenated_data[location][sensor] = concatenated_data[location][
                    sensor
                ][:min_length]
                print(
                    f"截断 '{location}/{sensor}' 从 {original_length} 到 {min_length}"
                )

        # 拼接所有标签
        concatenated_labels = torch.cat(concatenated_labels, dim=0)
        print(f"总拼接标签的形状: {concatenated_labels.shape}")

        # 根据最小长度计算窗口数量
        num_windows = (min_length - self.window_size) // self.stride + 1
        print(f"基于最小长度的窗口数量: {num_windows}")

        # 确保有足够的标签
        if len(concatenated_labels) < num_windows:
            raise ValueError(
                f"标签数量 ({len(concatenated_labels)}) 少于窗口数量 ({num_windows})。"
            )
        concatenated_labels = concatenated_labels[:num_windows]
        print(f"截断标签以匹配窗口数量: {concatenated_labels.shape}")

        # 初始化 PPGDataset
        dataset = PPGDataset(
            data=concatenated_data,
            labels=concatenated_labels,
            window_size=self.window_size,
            stride=self.stride,
        )

        # 将数据集分割为训练集、验证集和测试集
        total_length = len(dataset)
        train_length = int(self.train_split * total_length)
        val_length = int(self.val_split * total_length)
        test_length = total_length - train_length - val_length

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            [train_length, val_length, test_length],
            generator=torch.Generator().manual_seed(42),
        )

        print(
            f"数据集已分割为 训练集: {train_length}, 验证集: {val_length}, 测试集: {test_length}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # 调试时设置为 0
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # 调试时设置为 0
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # 调试时设置为 0
            pin_memory=True,
        )
