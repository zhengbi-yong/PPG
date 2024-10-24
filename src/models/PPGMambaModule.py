import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanMetric
from mamba_ssm import Mamba2  # 确保已安装 mamba_ssm 并正确导入

class PPGMambaModule(LightningModule):
    def __init__(
        self,
        input_dim: int = 14,          # 每个时间步的特征数
        d_model: int = 128,           # Mamba2 模型的特征维度
        d_state: int = 64,            # SSM 状态扩展因子
        d_conv: int = 4,              # 局部卷积宽度
        expand: int = 2,              # 块扩展因子
        dropout: float = 0.3,         # Dropout 比例
        lr: float = 1e-3,             # 学习率
        scheduler_step_size: int = 10,# 学习率调度器的步长
        scheduler_gamma: float = 0.1, # 学习率调度器的 gamma 值
    ):
        """
        LightningModule 使用 Mamba2 进行 PPG 数据预测。

        :param input_dim: 每个时间步的输入特征数。
        :param d_model: Mamba2 模型的特征维度。
        :param d_state: SSM 状态扩展因子，通常为 64 或 128。
        :param d_conv: 局部卷积宽度。
        :param expand: 块扩展因子。
        :param dropout: Dropout 比例。
        :param lr: 学习率。
        :param scheduler_step_size: 学习率调度器的步长。
        :param scheduler_gamma: 学习率调度器的 gamma 值。
        """
        super().__init__()
        self.save_hyperparameters()

        # 输入投影层：将输入特征从 input_dim 映射到 d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Mamba2 模块
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 池化层：对 Mamba2 的输出进行平均池化，得到固定长度的特征表示
        self.pool = nn.AdaptiveAvgPool1d(1)  # 输出形状 [batch_size, d_model, 1]

        # 回归头：将池化后的特征映射到目标值
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1)  # 单输出

        # 损失函数
        self.criterion = nn.MSELoss()

        # 评价指标
        self.train_mse = MeanMetric()
        self.val_mse = MeanMetric()
        self.test_mse = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        :param x: 输入张量，形状为 [batch_size, seq_len, input_dim]
        :return: 预测张量，形状为 [batch_size]
        """
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]

        # Mamba2 模块
        x = self.mamba(x)  # [batch_size, seq_len, d_model]

        # 转置为 [batch_size, d_model, seq_len] 以适应 AdaptiveAvgPool1d
        x = x.permute(0, 2, 1)  # [batch_size, d_model, seq_len]

        # 池化
        x = self.pool(x)  # [batch_size, d_model, 1]
        x = x.squeeze(-1)  # [batch_size, d_model]

        # 回归头
        x = F.relu(self.fc1(x))  # [batch_size, d_model // 2]
        x = self.dropout(x)
        x = self.fc2(x).squeeze(1)  # [batch_size]

        return x

    def training_step(self, batch, batch_idx):
        """
        训练步骤。

        :param batch: 数据批次。
        :param batch_idx: 批次索引。
        :return: 损失值。
        """
        x, y = batch  # x: [batch_size, window_size, 14], y: [batch_size]
        y_pred = self.forward(x)  # [batch_size]
        loss = self.criterion(y_pred, y)

        # 记录损失和指标
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_mse(y_pred, y)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        验证步骤。

        :param batch: 数据批次。
        :param batch_idx: 批次索引。
        """
        x, y = batch  # x: [batch_size, window_size, 14], y: [batch_size]
        y_pred = self.forward(x)  # [batch_size]
        loss = self.criterion(y_pred, y)

        # 记录损失和指标
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_mse(y_pred, y)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        测试步骤。

        :param batch: 数据批次。
        :param batch_idx: 批次索引。
        """
        x, y = batch  # x: [batch_size, window_size, 14], y: [batch_size]
        y_pred = self.forward(x)  # [batch_size]
        loss = self.criterion(y_pred, y)

        # 记录损失和指标
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_mse(y_pred, y)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。

        :return: 优化器和调度器配置。
        """
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        return [optimizer], [scheduler]
