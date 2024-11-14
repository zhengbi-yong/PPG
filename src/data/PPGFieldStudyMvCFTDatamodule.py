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
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) / self.std

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)


def mswf_filter(signal: torch.Tensor, short_window: int = 5, long_window: int = 15) -> torch.Tensor:
    """
    Multi-Size Weighted Fitting (MSWF) function to smooth and filter PPG signal.
    """
    # Placeholder: Implement the actual MSWF filtering logic as described in the paper
    return signal  # This should be the filtered signal


def generate_kfr_features(signal: torch.Tensor, m: int = 320) -> torch.Tensor:
    """
    Generate Kinetic Feature Reconstruction (KFR) features: SPE, GASF, and RP.
    """
    # Placeholder for PAA, SPE, GASF, RP computations
    spe = signal.mean(dim=0).unsqueeze(0)  # Placeholder for SPE
    gasf = signal.mean(dim=0).unsqueeze(0)  # Placeholder for GASF
    rp = signal.mean(dim=0).unsqueeze(0)    # Placeholder for RP
    return torch.cat([spe, gasf, rp], dim=0)


class PPGDataset(Dataset):
    def __init__(self, data: Dict[str, Any], labels: torch.Tensor, window_size: int = 256, stride: int = 128):
        self.window_size = window_size
        self.stride = stride

        # Apply MSWF and KFR to preprocess data
        for location in ["chest", "wrist"]:
            for sensor in data[location]:
                data[location][sensor] = mswf_filter(data[location][sensor])
                data[location][sensor] = generate_kfr_features(data[location][sensor])

        self.data = data
        self.labels = labels
        self.length = data["chest"]["ECG"].shape[0]
        self.num_windows = (self.length - window_size) // stride + 1

        assert len(labels) >= self.num_windows, f"标签数量 ({len(labels)}) 少于窗口数量 ({self.num_windows})。"

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size

        features = {}
        for location in ["chest", "wrist"]:
            features[location] = {}
            for sensor, signal in self.data[location].items():
                window = signal[start:end]
                features[location][sensor] = window

        chest_features = torch.cat([features["chest"][sensor] for sensor in features["chest"]], dim=1)
        wrist_features = torch.cat([features["wrist"][sensor] for sensor in features["wrist"]], dim=1)
        combined_features = torch.cat([chest_features, wrist_features], dim=1)

        label = self.labels[idx]
        return combined_features, label


class PPGDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./data/PPG_FieldStudy", window_size: int = 256, stride: int = 128, 
                 batch_size: int = 64, num_workers: int = 0, train_split: float = 0.7, val_split: float = 0.15, 
                 test_split: float = 0.15):
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

            for location in ["chest", "wrist"]:
                for sensor, signal in subject_data["signal"][location].items():
                    concatenated_data[location][sensor].append(torch.tensor(signal, dtype=torch.float32))

            concatenated_labels.append(torch.tensor(subject_data["label"], dtype=torch.float32))

        if not concatenated_labels:
            raise ValueError("未加载到任何有效的 Pickle 文件。请检查你的数据。")

        for location in ["chest", "wrist"]:
            for sensor in concatenated_data[location]:
                concatenated_data[location][sensor] = torch.cat(concatenated_data[location][sensor], dim=0)

        lengths = [concatenated_data[location][sensor].shape[0] for location in ["chest", "wrist"] for sensor in concatenated_data[location]]
        min_length = min(lengths)

        for location in ["chest", "wrist"]:
            for sensor in concatenated_data[location]:
                concatenated_data[location][sensor] = concatenated_data[location][sensor][:min_length]

        concatenated_labels = torch.cat(concatenated_labels, dim=0)[:min_length]
        num_windows = (min_length - self.window_size) // self.stride + 1

        concatenated_labels = concatenated_labels[:num_windows]

        dataset = PPGDataset(data=concatenated_data, labels=concatenated_labels, window_size=self.window_size, stride=self.stride)
        total_length = len(dataset)
        train_length = int(self.train_split * total_length)
        val_length = int(self.val_split * total_length)
        test_length = total_length - train_length - val_length

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_length, val_length, test_length], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
