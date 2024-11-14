import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class CrossViewFeatureFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.q_linear = nn.Linear(input_dim, hidden_dim)
        self.k_linear = nn.Linear(input_dim, hidden_dim)
        self.v_linear = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, view1, view2):
        Q = self.q_linear(view1)
        K = self.k_linear(view2)
        V = self.v_linear(view2)
        attn_weights = self.softmax(Q @ K.transpose(-2, -1) / np.sqrt(view1.size(-1)))
        return attn_weights @ V

class MultiScaleTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(), nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        return self.ff(attn_out + x)

class PPGMvCFTModule(LightningModule):
    def __init__(self, input_dim, d_model, lr=1e-3):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.cvff = CrossViewFeatureFusion(d_model, d_model)
        self.mst = MultiScaleTransformer(d_model)
        self.fc = nn.Linear(d_model, 1)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x_view1, x_view2):
        x_view1 = self.conv1d(x_view1)
        x_view2 = self.conv1d(x_view2)
        fused = self.cvff(x_view1, x_view2)
        output = self.mst(fused)
        return self.fc(output.mean(dim=1))

    def training_step(self, batch, batch_idx):
        x_view1, x_view2, y = batch
        y_hat = self.forward(x_view1, x_view2)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
