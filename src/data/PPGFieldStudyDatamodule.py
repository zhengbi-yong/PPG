import os
import pickle
import numpy as np
from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from lightning import LightningDataModule


class PPGDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, Any],
        labels: np.ndarray,
        window_size: int = 256,
        stride: int = 128,
        transform: Optional[Any] = None,
    ):
        """
        Custom Dataset for PPG data.

        :param data: Dictionary containing concatenated signal data from all subjects.
        :param labels: Numpy array of concatenated labels from all subjects.
        :param window_size: Number of time steps per window.
        :param stride: Stride between windows.
        :param transform: Optional transform to be applied on a sample.
        """
        self.window_size = window_size
        self.stride = stride
        self.transform = transform

        # Assuming all signals have been truncated to the same length
        self.length = data["chest"]["ECG"].shape[0]
        self.num_windows = (self.length - window_size) // stride + 1

        # Validate that all signals have the same length
        for location in ["chest", "wrist"]:
            for sensor in data[location]:
                assert data[location][sensor].shape[0] == self.length, (
                    f"Signal length mismatch in {location}/{sensor}: "
                    f"expected {self.length}, got {data[location][sensor].shape[0]}"
                )

        # Validate labels length
        assert (
            len(labels) >= self.num_windows
        ), f"Number of labels ({len(labels)}) is less than number of windows ({self.num_windows})."

        # Normalize signals using StandardScaler
        self.scalers = {}
        for location in ["chest", "wrist"]:
            self.scalers[location] = {}
            for sensor, signal in data[location].items():
                scaler = StandardScaler()
                if signal.ndim > 1:
                    scaler.fit(signal)
                    self.scalers[location][sensor] = scaler
                else:
                    scaler.fit(signal.reshape(-1, 1))
                    self.scalers[location][sensor] = scaler

        self.data = data
        self.labels = labels[:self.num_windows]  # Ensure labels match the number of windows

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        """
        Retrieves a window of data and its corresponding label.

        :param idx: Index of the window.
        :return: Tuple of (features, label)
        """
        start = idx * self.stride
        end = start + self.window_size

        features = {}
        for location in ["chest", "wrist"]:
            features[location] = {}
            for sensor, signal in self.data[location].items():
                window = signal[start:end]
                scaler = self.scalers[location][sensor]
                window = (
                    scaler.transform(window)
                    if signal.ndim > 1
                    else scaler.transform(window.reshape(-1, 1)).squeeze()
                )
                features[location][sensor] = window

        # Combine all features into a single array
        # Concatenate all sensor data along the feature dimension
        chest_features = np.concatenate(
            [features["chest"][sensor] for sensor in features["chest"]], axis=1
        )  # Shape: [window_size, 8]
        wrist_features = np.concatenate(
            [features["wrist"][sensor] for sensor in features["wrist"]], axis=1
        )  # Shape: [window_size, 6]
        combined_features = np.concatenate(
            [chest_features, wrist_features], axis=1
        )  # Shape: [window_size, 14]

        if self.transform:
            combined_features = self.transform(combined_features)

        # 保留整个窗口的特征，不进行汇聚
        aggregated_features = combined_features  # Shape: [window_size, 14]

        # 获取对应的标签
        label = self.labels[idx]

        return torch.tensor(aggregated_features, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


class PPGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/PPG_FieldStudy",
        window_size: int = 256,
        stride: int = 128,
        batch_size: int = 64,
        num_workers: int = 0,  # Set to 0 for debugging
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
    ):
        """
        DataModule for PPG dataset.

        :param data_dir: The root directory containing subject folders.
        :param window_size: Number of time steps per window.
        :param stride: Stride between windows.
        :param batch_size: Batch size.
        :param num_workers: Number of workers for DataLoader.
        :param train_split: Proportion of data for training.
        :param val_split: Proportion of data for validation.
        :param test_split: Proportion of data for testing.
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
        Load and preprocess data from all subjects.

        :param stage: Optional stage to setup (fit, validate, test, predict).
        """
        # Initialize empty dictionaries for concatenated data
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

        # Iterate through each subject directory (S1 to S15)
        for subject_num in range(1, 16):
            subject_dir = os.path.join(self.data_dir, f"S{subject_num}")
            pkl_file = os.path.join(subject_dir, f"S{subject_num}.pkl")

            if not os.path.isfile(pkl_file):
                print(f"Pickle file not found: {pkl_file}. Skipping.")
                continue

            try:
                with open(pkl_file, "rb") as f:
                    subject_data = pickle.load(f, encoding="latin1")
                print(f"Loaded {pkl_file} successfully.")
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}. Skipping.")
                continue

            # Concatenate each sensor's data
            for location in ["chest", "wrist"]:
                for sensor, signal in subject_data["signal"][location].items():
                    concatenated_data[location][sensor].append(signal)

            # Concatenate labels
            concatenated_labels.append(subject_data["label"])

        if not concatenated_labels:
            raise ValueError(
                "No valid pickle files were loaded. Please check your data."
            )

        # After loading all subjects, concatenate the signals along the time axis
        for location in ["chest", "wrist"]:
            for sensor in concatenated_data[location]:
                # Stack along the first dimension (time)
                concatenated_data[location][sensor] = np.concatenate(
                    concatenated_data[location][sensor], axis=0
                )
                print(
                    f"Sensor '{location}/{sensor}' concatenated shape: {concatenated_data[location][sensor].shape}"
                )

        # Compute the minimum length across all sensors
        lengths = [
            concatenated_data[location][sensor].shape[0]
            for location in ["chest", "wrist"]
            for sensor in concatenated_data[location]
        ]
        min_length = min(lengths)
        print(f"Minimum signal length across all sensors: {min_length}")

        # Truncate all signals to min_length
        for location in ["chest", "wrist"]:
            for sensor in concatenated_data[location]:
                original_length = concatenated_data[location][sensor].shape[0]
                concatenated_data[location][sensor] = concatenated_data[location][
                    sensor
                ][:min_length]
                print(
                    f"Truncated '{location}/{sensor}' from {original_length} to {min_length}"
                )

        # Concatenate all labels
        concatenated_labels = np.concatenate(concatenated_labels, axis=0)
        print(f"Total concatenated labels shape: {concatenated_labels.shape}")

        # Determine the number of windows based on min_length
        num_windows = (min_length - self.window_size) // self.stride + 1
        print(f"Number of windows based on min_length: {num_windows}")

        # Ensure there are enough labels
        if len(concatenated_labels) < num_windows:
            raise ValueError(
                f"Not enough labels ({len(concatenated_labels)}) for the number of windows ({num_windows})."
            )
        concatenated_labels = concatenated_labels[:num_windows]
        print(
            f"Truncated labels to match number of windows: {concatenated_labels.shape}"
        )

        # Initialize the PPGDataset with truncated data and labels
        dataset = PPGDataset(
            data=concatenated_data,
            labels=concatenated_labels,
            window_size=self.window_size,
            stride=self.stride,
        )

        # Split the dataset into train, val, and test
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
            f"Dataset split into Train: {train_length}, Val: {val_length}, Test: {test_length}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # Set to 0 for debugging
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # Set to 0 for debugging
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # Set to 0 for debugging
            pin_memory=True,
        )
