import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanMetric

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        位置编码模块。

        :param d_model: 输入特征的维度。
        :param max_len: 支持的最大序列长度。
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建一个 [max_len, d_model] 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        为输入添加位置编码。

        :param x: 输入张量，形状为 [batch_size, seq_len, d_model]
        :return: 添加位置编码后的张量，形状相同
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class PPGTransformerModule(LightningModule):
    def __init__(
        self,
        input_dim: int = 14,            # 每个时间步的特征数
        d_model: int = 128,             # Transformer模型的特征维度
        nhead: int = 8,                 # 多头注意力机制的头数
        num_encoder_layers: int = 4,    # Transformer编码器层数
        dim_feedforward: int = 256,     # 前馈网络的维度
        dropout: float = 0.1,           # Dropout比例
        lr: float = 1e-3,               # 学习率
        scheduler_step_size: int = 10,  # 学习率调度器的步长
        scheduler_gamma: float = 0.1,   # 学习率调度器的gamma值
    ):
        """
        LightningModule 使用 Transformer 进行 PPG 数据预测。

        :param input_dim: 每个时间步的输入特征数。
        :param d_model: Transformer模型的特征维度。
        :param nhead: 多头注意力机制的头数。
        :param num_encoder_layers: Transformer编码器层数。
        :param dim_feedforward: 前馈网络的维度。
        :param dropout: Dropout比例。
        :param lr: 学习率。
        :param scheduler_step_size: 学习率调度器的步长。
        :param scheduler_gamma: 学习率调度器的gamma值。
        """
        super().__init__()
        self.save_hyperparameters()

        # 线性层将输入特征映射到d_model维度
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # 使用batch_first=True以匹配输入形状 [batch_size, seq_len, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # 全连接层用于回归
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
        # 线性映射
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]

        # 添加位置编码
        x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]

        # Transformer编码器
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]

        # 池化操作：可以使用平均池化或取最后一个时间步
        # 这里使用平均池化
        x = x.mean(dim=1)  # [batch_size, d_model]

        # 全连接层
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
