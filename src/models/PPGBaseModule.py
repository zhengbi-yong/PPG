import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanSquaredError
from torchmetrics import MeanMetric


class PPGBaseModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1,
    ):
        """
        LightningModule for PPG-based physiological indicator prediction.

        :param input_dim: Number of input features.
        :param hidden_dim: Number of hidden units.
        :param lr: Learning rate.
        :param scheduler_step_size: Step size for learning rate scheduler.
        :param scheduler_gamma: Gamma for learning rate scheduler.
        """
        super().__init__()
        self.save_hyperparameters()

        # Define the network architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 1),  # Single output
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Metrics
        self.train_mse = MeanMetric()
        self.val_mse = MeanMetric()
        self.test_mse = MeanMetric()

    def forward(self, x):
        """
        Forward pass.

        :param x: Input tensor of shape [batch_size, 14]
        :return: Predicted tensor of shape [batch_size]
        """
        encoded = self.encoder(x)  # [batch_size, hidden_dim//2]
        output = self.regressor(encoded)  # [batch_size, 1]
        return output.squeeze(1)  # [batch_size]

    def training_step(self, batch, batch_idx):
        """
        Training step.

        :param batch: Batch of data.
        :param batch_idx: Batch index.
        :return: Loss value.
        """
        x, y = batch  # x: [batch_size, 14], y: [batch_size]
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
        x, y = batch  # x: [batch_size, 14], y: [batch_size]
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
        x, y = batch  # x: [batch_size, 14], y: [batch_size]
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
