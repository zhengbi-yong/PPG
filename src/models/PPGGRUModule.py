import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanMetric

class PPGGRUModule(LightningModule):
    def __init__(
        self,
        input_dim: int = 14,          # 每个时间步的特征数
        hidden_dim: int = 128,        # GRU隐藏状态的维度
        num_layers: int = 2,          # GRU层数
        bidirectional: bool = True,   # 是否使用双向GRU
        dropout: float = 0.3,         # GRU和全连接层之间的Dropout比例
        lr: float = 1e-3,             # 学习率
        scheduler_step_size: int = 10,# 学习率调度器的步长
        scheduler_gamma: float = 0.1, # 学习率调度器的gamma值
    ):
        """
        LightningModule 使用 GRU 进行 PPG 数据预测。

        :param input_dim: 每个时间步的输入特征数。
        :param hidden_dim: GRU隐藏状态的维度。
        :param num_layers: GRU的层数。
        :param bidirectional: 是否使用双向GRU。
        :param dropout: Dropout比例。
        :param lr: 学习率。
        :param scheduler_step_size: 学习率调度器的步长。
        :param scheduler_gamma: 学习率调度器的gamma值。
        """
        super().__init__()
        self.save_hyperparameters()

        # 输入投影层：将输入特征从 input_dim 映射到 hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # GRU层
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0  # 只有多层时才使用Dropout
        )

        # 计算GRU输出的特征维度
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 回归头：全连接层
        self.fc1 = nn.Linear(gru_output_dim, gru_output_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(gru_output_dim // 2, 1)  # 单输出

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
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]

        # GRU层
        # h_0的形状为 [num_layers * num_directions, batch_size, hidden_dim]
        h_0 = torch.zeros(
            self.hparams.num_layers * (2 if self.hparams.bidirectional else 1),
            x.size(0),
            self.hparams.hidden_dim,
            device=x.device
        )
        gru_out, h_n = self.gru(x, h_0)  # gru_out: [batch_size, seq_len, hidden_dim * num_directions]

        # 处理双向GRU的隐藏状态
        if self.hparams.bidirectional:
            # 取最后一层的正向和反向隐藏状态
            h_forward = h_n[-2, :, :]  # [batch_size, hidden_dim]
            h_backward = h_n[-1, :, :] # [batch_size, hidden_dim]
            h = torch.cat((h_forward, h_backward), dim=1)  # [batch_size, hidden_dim * 2]
        else:
            h = h_n[-1, :, :]  # [batch_size, hidden_dim]

        # 全连接层
        x = F.relu(self.fc1(h))      # [batch_size, hidden_dim]
        x = self.dropout(x)
        x = self.fc2(x).squeeze(1)   # [batch_size]

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
