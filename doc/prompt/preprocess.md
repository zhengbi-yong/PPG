```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanMetric


class PPGBaseModule(LightningModule):
    def __init__(
        self,
        input_dim: int = 14,  # 每个时间步的特征数
        window_size: int = 256,  # 时间步数
        hidden_dim: int = 128,
        lr: float = 1e-3,
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1,
    ):
        """
        LightningModule for PPG-based physiological indicator prediction.

        :param input_dim: Number of input features per time step.
        :param window_size: Number of time steps per window.
        :param hidden_dim: Number of hidden units.
        :param lr: Learning rate.
        :param scheduler_step_size: Step size for learning rate scheduler.
        :param scheduler_gamma: Gamma for learning rate scheduler.
        """
        super().__init__()
        self.save_hyperparameters()

        # 定义1D卷积网络
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 计算池化后的时间步数
        pooled_size = window_size // 2  # 因为进行了一个池化层，时间步数减半

        self.fc1 = nn.Linear((hidden_dim*2) * pooled_size, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Loss function
        self.criterion = nn.MSELoss()

        # Metrics
        self.train_mse = MeanMetric()
        self.val_mse = MeanMetric()
        self.test_mse = MeanMetric()

    def forward(self, x):
        """
        Forward pass.

        :param x: Input tensor of shape [batch_size, window_size, 14]
        :return: Predicted tensor of shape [batch_size]
        """
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, 14, window_size] 以适应 Conv1d
        x = F.relu(self.bn1(self.conv1(x)))  # [batch_size, hidden_dim, window_size]
        x = F.relu(self.bn2(self.conv2(x)))  # [batch_size, hidden_dim*2, window_size]
        x = self.pool(x)  # [batch_size, hidden_dim*2, window_size//2]
        x = x.view(x.size(0), -1)  # 展平成 [batch_size, hidden_dim*2 * (window_size//2)]
        x = F.relu(self.fc1(x))  # [batch_size, hidden_dim]
        x = self.dropout(x)
        x = self.fc2(x).squeeze(1)  # [batch_size]
        return x

    def training_step(self, batch, batch_idx):
        """
        Training step.

        :param batch: Batch of data.
        :param batch_idx: Batch index.
        :return: Loss value.
        """
        x, y = batch  # x: [batch_size, window_size, 14], y: [batch_size]
        y_pred = self.forward(x)  # [batch_size]
        loss = self.criterion(y_pred, y)

        # Log loss and metric
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_mse(y_pred, y)
        self.log(
            "train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        :param batch: Batch of data.
        :param batch_idx: Batch index.
        """
        x, y = batch  # x: [batch_size, window_size, 14], y: [batch_size]
        y_pred = self.forward(x)  # [batch_size]
        loss = self.criterion(y_pred, y)

        # Log loss and metric
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_mse(y_pred, y)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step.

        :param batch: Batch of data.
        :param batch_idx: Batch index.
        """
        x, y = batch  # x: [batch_size, window_size, 14], y: [batch_size]
        y_pred = self.forward(x)  # [batch_size]
        loss = self.criterion(y_pred, y)

        # Log loss and metric
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_mse(y_pred, y)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        :return: Optimizer and scheduler configuration.
        """
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        return [optimizer], [scheduler]
```
这是我的一个能够运行的例子，我的项目是基于lightning的项目，所以我希望你把MvCFT的代码写成上述格式，以便我们能够快速地进行实验。
```python
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
                    data[location][sensor] = scaler.transform(signal.unsqueeze(1)).squeeze()

        self.data = data
        self.labels = labels[:self.num_windows]  # 确保标签与窗口数量匹配

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
        combined_features = torch.cat([chest_features, wrist_features], dim=1)  # 形状: [window_size, total_features]

        if self.transform:
            combined_features = self.transform(combined_features)

        # 保留整个窗口的特征，不进行汇聚
        aggregated_features = combined_features  # 形状: [window_size, total_features]

        # 获取对应的标签
        label = self.labels[idx]

        return aggregated_features, label


class PPGDataModule(LightningDataModule):
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
                    concatenated_data[location][sensor].append(torch.tensor(signal, dtype=torch.float32))

            # 拼接标签
            concatenated_labels.append(torch.tensor(subject_data["label"], dtype=torch.float32))

        if not concatenated_labels:
            raise ValueError(
                "未加载到任何有效的 Pickle 文件。请检查你的数据。"
            )

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
                concatenated_data[location][sensor] = concatenated_data[location][sensor][:min_length]
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
        print(
            f"截断标签以匹配窗口数量: {concatenated_labels.shape}"
        )

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
```
上面的代码是我目前加载数据集所使用的代码，请你根据论文中提到的方法对其进行修改。