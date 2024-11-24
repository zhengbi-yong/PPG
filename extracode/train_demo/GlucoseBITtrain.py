# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging

# 配置日志（可选）
logging.basicConfig(
    filename='GlucoseBITProcess.log',
    filemode='w',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 设置随机种子，确保结果可重复
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据
csv_path = '../../data/GlucoseBIT/glucose_data.csv'  # 请确保CSV文件路径正确
df = pd.read_csv(csv_path)

# 查看数据基本信息
print("数据集概览:")
print(df.head())
print(df['label'].value_counts())

# 按5分钟划分数据
def group_by_five_minutes(df):
    """
    将数据按照每5分钟进行分组，并计算每组的特征均值。
    """
    # 确保数据按时间排序
    df = df.sort_values(['date', 'time_min']).reset_index(drop=True)
    
    # 创建一个新的列 'minute_group' 表示每5分钟的组
    df['minute_group'] = (df['time_min'] // 5).astype(int)
    
    # 按照 'date' 和 'minute_group' 进行分组，并计算每组的特征均值
    grouped = df.groupby(['date', 'minute_group']).agg({
        'frequency_hz': 'mean',
        'amplitude': 'mean',
        'phase': 'mean'
    }).reset_index()
    
    # 计算累计分钟数
    # 假设每个 'minute_group' 代表一个连续的5分钟段
    grouped = grouped.sort_values(['date', 'minute_group']).reset_index(drop=True)
    grouped['cumulative_time_min'] = grouped.groupby('date')['minute_group'].transform(lambda x: x * 5)
    
    # 分配标签
    def assign_label(total_minutes):
        if (0 <= total_minutes < 10) or (total_minutes >= 65):
            return 0  # low
        elif (10 <= total_minutes < 25) or (45 <= total_minutes < 65):
            return 1  # medium
        elif 25 <= total_minutes < 45:
            return 2  # high
        else:
            return 0  # 默认归为低
    
    grouped['label'] = grouped['cumulative_time_min'].apply(assign_label)
    
    return grouped

# 按5分钟划分
df_grouped = group_by_five_minutes(df)

print("按5分钟划分后的数据集概览:")
print(df_grouped.head())
print(df_grouped['label'].value_counts())

# 特征和标签
feature_columns = ['frequency_hz', 'amplitude', 'phase']
X = df_grouped[feature_columns].values
y = df_grouped['label'].values

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"验证集大小: {X_val.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 定义自定义 Dataset 类
class GlucoseDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建 Dataset 实例
train_dataset = GlucoseDataset(X_train, y_train)
val_dataset = GlucoseDataset(X_val, y_val)
test_dataset = GlucoseDataset(X_test, y_test)

# 创建 DataLoader
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义神经网络模型
class GlucoseClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_classes=3):
        super(GlucoseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 初始化模型、损失函数和优化器
model = GlucoseClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义训练函数
def train_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        print(inputs.shape)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 定义验证函数
def validate_epoch(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 训练和验证过程
num_epochs = 10000000  # 您可以根据需要调整轮数
best_val_acc = 0.0
best_model_path = 'best_glucose_model.pth'

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
    val_loss, val_acc = validate_epoch(model, device, val_loader, criterion)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
    
    # 每100轮打印一次
    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{num_epochs}]")
        print(f"训练 Loss: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        print(f"验证 Loss: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        print("-" * 30)

print(f"最佳验证准确率: {best_val_acc:.4f}")

# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))

# 定义测试函数
def test_model(model, device, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_acc = correct / total
    return test_acc

# 测试模型
test_accuracy = test_model(model, device, test_loader)
print(f"测试集准确率: {test_accuracy:.4f}")
