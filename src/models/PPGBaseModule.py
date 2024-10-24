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
