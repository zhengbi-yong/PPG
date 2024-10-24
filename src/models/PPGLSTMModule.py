import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanMetric

class PPGLSTMModule(LightningModule):
    def __init__(
        self,
        input_dim: int = 14,          # 每个时间步的特征数
        hidden_dim: int = 128,        # LSTM隐藏状态的维度
        num_layers: int = 2,          # LSTM层数
        bidirectional: bool = True,   # 是否使用双向LSTM
        dropout: float = 0.3,         # LSTM和全连接层之间的Dropout比例
        lr: float = 1e-3,             # 学习率
        scheduler_step_size: int = 10, # 学习率调度器的步长
        scheduler_gamma: float = 0.1,  # 学习率调度器的gamma值
    ):
        """
        LightningModule 使用 LSTM 进行 PPG 数据预测。

        :param input_dim: 每个时间步的输入特征数。
        :param hidden_dim: LSTM隐藏状态的维度。
        :param num_layers: LSTM的层数。
        :param bidirectional: 是否使用双向LSTM。
        :param dropout: Dropout的比例。
        :param lr: 学习率。
        :param scheduler_step_size: 学习率调度器的步长。
        :param scheduler_gamma: 学习率调度器的gamma值。
        """
        super().__init__()
        self.save_hyperparameters()

        # 定义LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0  # 只有多层时才使用Dropout
        )

        # 计算LSTM输出的特征维度
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 定义全连接层
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 单输出

        # 损失函数
        self.criterion = nn.MSELoss()

        # 评价指标
        self.train_mse = MeanMetric()
        self.val_mse = MeanMetric()
        self.test_mse = MeanMetric()

    def forward(self, x):
        """
        前向传播。

        :param x: 输入张量，形状为 [batch_size, window_size, 14]
        :return: 预测张量，形状为 [batch_size]
        """
        # LSTM的输入形状为 [batch_size, window_size, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)  
        # h_n的形状为 [num_layers * num_directions, batch_size, hidden_dim]

        # 如果使用双向LSTM，连接最后一层的正向和反向隐藏状态
        if self.hparams.bidirectional:
            # 取最后一层的正向和反向隐藏状态
            h_forward = h_n[-2, :, :]  # [batch_size, hidden_dim]
            h_backward = h_n[-1, :, :] # [batch_size, hidden_dim]
            h = torch.cat((h_forward, h_backward), dim=1)  # [batch_size, hidden_dim * 2]
        else:
            h = h_n[-1, :, :]  # [batch_size, hidden_dim]

        # 通过全连接层
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
