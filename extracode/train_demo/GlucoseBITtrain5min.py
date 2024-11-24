# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import os

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

# 定义标签函数
def assign_label(total_minutes):
    if (0 <= total_minutes < 10) or (total_minutes >= 65):
        return 0  # low
    elif (10 <= total_minutes < 25) or (45 <= total_minutes < 65):
        return 1  # medium
    elif 25 <= total_minutes < 45:
        return 2  # high
    else:
        return 0  # 默认归为低

# 为每个数据点分配标签
df['data_label'] = df['time_min'].apply(assign_label)

# 移除原有的 'label' 列，以避免混淆（如果 'label' 列不再需要）
if 'label' in df.columns:
    df = df.drop(columns=['label'])

# 按滑动窗口划分数据，确保窗口内数据属于同一类别
def sliding_window_group(df, window_size=5, step_size=1, samples_per_minute=100):
    """
    将数据按照滑动窗口进行分组，并将每组的数据展平成一个向量。
    
    参数：
    - df: 原始数据的DataFrame
    - window_size: 窗口大小（分钟）
    - step_size: 窗口步长（分钟）
    - samples_per_minute: 每分钟的采样数
    
    返回：
    - grouped_df: 按滑动窗口划分并展平后的DataFrame
    """
    # 确保数据按时间排序
    df = df.sort_values(['date', 'segment', 'time_min', 'time_sec']).reset_index(drop=True)
    
    # 初始化列表来存储处理后的数据
    processed_data = []
    
    # 计算每个窗口的样本数
    samples_per_window = window_size * samples_per_minute  # 5分钟 * 100采样/分钟 = 500
    feature_length = samples_per_window * 3  # 每个数据点有3个特征
    
    # 遍历每个日期和段
    for (date, segment), group in df.groupby(['date', 'segment']):
        group = group.reset_index(drop=True)
        total_minutes = group['time_min'].max()
        # 计算窗口的起始分钟
        window_starts = np.arange(0, total_minutes - window_size + step_size, step_size)
        for start_min in window_starts:
            end_min = start_min + window_size
            # 过滤出窗口内的数据
            window_data = group[(group['time_min'] >= start_min) & (group['time_min'] < end_min)]
            actual_samples = window_data.shape[0]
            # 检查样本数是否足够
            if actual_samples < samples_per_window:
                logging.warning(f"日期 {date} 的段 {segment} 的窗口 [{start_min}, {end_min}) 样本数为 {actual_samples}，不足 {samples_per_window}。该窗口将被忽略。")
                continue  # 忽略样本数不足的窗口
            # 提取前samples_per_window个数据点
            window_data = window_data.iloc[:samples_per_window]
            # 提取特征并展平
            features = window_data[['frequency_hz', 'amplitude', 'phase']].values.flatten()
            # 分配标签，使用窗口的结束时间
            window_label = assign_label(end_min)
            # 检查窗口内所有数据点的标签是否一致
            if not (window_data['data_label'] == window_label).all():
                logging.warning(f"日期 {date} 的段 {segment} 的窗口 [{start_min}, {end_min}) 中存在跨标签数据。该窗口将被忽略。")
                continue  # 忽略跨标签的窗口
            # 添加到列表
            processed_data.append({
                'date': date,
                'segment': segment,
                'window_start_min': start_min,
                'window_end_min': end_min,
                'features': features,
                'label': window_label
            })
    
    # 创建处理后的DataFrame
    grouped_df = pd.DataFrame(processed_data)
    
    return grouped_df

# 按滑动窗口划分并展平数据
df_grouped = sliding_window_group(df, window_size=5, step_size=1, samples_per_minute=100)

# 检查是否有数据被处理
if df_grouped.empty:
    raise ValueError("分组后的数据集为空。请检查数据是否满足每5分钟100个样本的要求，或者调整 `window_size`、`step_size` 和 `samples_per_minute` 参数。")

print("按滑动窗口划分后的数据集概览:")
print(df_grouped.head())
print("列名:", df_grouped.columns)
print(df_grouped['label'].value_counts())
print(f"总分组数: {len(df_grouped)}")

# 特征和标签
X = np.vstack(df_grouped['features'].values)  # 形状为 [样本数, 1500]
y = df_grouped['label'].values

print(f"特征形状: {X.shape}")  # 应为 [样本数, 1500]
print(f"标签形状: {y.shape}")  # 应为 [样本数]

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据集划分
if len(X) < 2:
    raise ValueError("样本数不足，无法进行训练和测试。请确保有足够的滑动窗口组被保留。")

# 首先划分训练集和临时集（训练集70%，临时集30%）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 打印临时集的标签分布
print("y_temp 分布:", pd.Series(y_temp).value_counts())

# 尝试使用 stratify 进行第二次划分
try:
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
except ValueError as e:
    print("Stratified split failed, trying without stratification.")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=None
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
    def __init__(self, input_size=1500, hidden_size=512, num_classes=3):
        super(GlucoseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 设置输入尺寸为 1500
input_size = X.shape[1]  # 1500
model = GlucoseClassifier(input_size=input_size, hidden_size=512, num_classes=3).to(device)

# 检查是否存在旧的模型文件，避免加载不兼容的 state_dict
best_model_path = 'best_glucose_model.pth'
if os.path.exists(best_model_path):
    print(f"检测到旧的模型文件 '{best_model_path}'，将其删除以避免加载不兼容的参数。")
    os.remove(best_model_path)

# 初始化损失函数和优化器
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
num_epochs = 1000  # 您可以根据需要调整轮数
best_val_acc = 0.0
best_model_path = 'best_glucose_model.pth'  # 确保这是一个新的文件名或已删除旧文件

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
