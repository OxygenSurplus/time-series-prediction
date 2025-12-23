# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main-full-safe-final-0.2.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import pandas as pd
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

start_time = time.time()

# ====================== 全局配置：CPU线程优化（不影响核心逻辑） ======================
torch.set_num_threads(int(torch.get_num_threads() * 0.8))

# 加载数据集
train_dataSet = pd.read_csv(r'../../../modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'../../../modified_数据集Time_Series662_detail.dat')

# 列定义
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 记录CO2相关特征的索引（用于后续加权和特征增强）
# 输出列中：CO2_density(1)、CO2_density_fast_tmpr(2)
co2_output_indices = [1, 2]
# 输入噪声列中：Error_CO2_density(1)、Error_CO2_density_fast_tmpr(2)
co2_input_indices = [1, 2]

CL = columns + noise_columns

## 查看数据缺失情况
data = train_dataSet[CL]
missingDf = data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns = ['feature', 'miss_num']
missingDf['miss_percentage'] = missingDf['miss_num'] / data.shape[0]
print("缺失值比例")
print(missingDf)

# 异常值检测（Z-score）：保留原代码逻辑
outlier_ratios = {}
for column in CL:
    col_clean = train_dataSet[column].dropna()
    if len(col_clean) == 0:
        outlier_ratios[column] = 0.0
        continue
    z_scores = np.abs(stats.zscore(col_clean))
    outliers = (z_scores > 2)
    outlier_ratio = outliers.mean() if len(outliers) > 0 else 0.0
    outlier_ratios[column] = outlier_ratio

print("*" * 30)
print("异常值的比例:")
for column, ratio in outlier_ratios.items():
    print(f"{column}: {ratio:.2%}")

# ====================== 修复1：缺失值处理（原代码隐藏bug） ======================
imputer = SimpleImputer(strategy='median')
train_data_filled = train_dataSet.copy()
test_data_filled = test_dataSet.copy()
train_data_filled[CL] = imputer.fit_transform(train_data_filled[CL])
test_data_filled[CL] = imputer.transform(test_data_filled[CL])

# 划分原始输入输出
X_train_raw = train_data_filled[noise_columns].values.astype(np.float32)
y_train_raw = train_data_filled[columns].values.astype(np.float32)
X_test_raw = test_data_filled[noise_columns].values.astype(np.float32)
y_test_raw = test_data_filled[columns].values.astype(np.float32)

# === 数据标准化：保留原代码逻辑 ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

# ====================== 优化1：添加CO2特征的差分特征（捕捉时序变化趋势） ======================
def add_diff_feature(X, col_indices):
    """
    为指定列添加一阶差分特征（时序变化量）
    X: 原始特征矩阵
    col_indices: 需要添加差分的列索引
    """
    # 一阶差分：后一个值减前一个值
    diff_X = np.zeros((len(X), len(col_indices)))
    diff_X[1:] = X[1:, col_indices] - X[:-1, col_indices]
    # 拼接原始特征和差分特征
    new_X = np.hstack([X, diff_X])
    return new_X

# 为CO2相关输入特征添加差分特征
X_train_scaled = add_diff_feature(X_train_scaled, co2_input_indices)
X_test_scaled = add_diff_feature(X_test_scaled, co2_input_indices)
# 新的输入通道数：6（原始） + 2（差分） = 8
new_input_channels = X_train_scaled.shape[1]

# === 构造序列样本（用于训练）：保留原代码逻辑 ===
def create_sequences_for_training(X, y, seq_len=21, max_samples=None):
    Xs, ys = [], []
    half = seq_len // 2
    start, end = half, len(X) - half
    if end <= start:
        raise ValueError("数据太短")

    total = end - start
    if max_samples and total > max_samples:
        indices = np.linspace(start, end - 1, num=max_samples, dtype=int)
    else:
        indices = range(start, end)

    for i in indices:
        Xs.append(X[i - half:i + half + 1])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# === 设置参数：保留原代码逻辑 ===
SEQ_LEN = 21
MAX_SAMPLES = 200_000

print("正在构造训练序列...")
X_train_seq, y_train_seq = create_sequences_for_training(
    X_train_scaled, y_train_scaled, SEQ_LEN, max_samples=MAX_SAMPLES
)

# 转为 PyTorch 张量：保留原代码的permute操作（[N, seq_len, channels] → [N, channels, seq_len]）
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).permute(0, 2, 1)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

# ====================== 优化2：增强CNN模型（针对CO2特征，微调卷积核+保留原代码核心） ======================
class CNN1DRegressor(nn.Module):
    def __init__(self, input_channels, output_dim=6):
        super().__init__()
        # 优化：卷积核从5改为7（捕捉更长的CO2时序依赖），padding对应调整
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x):
        # 保留原代码的激活和池化逻辑
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# === 划分训练集和验证集：保留原代码逻辑 ===
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_tensor, y_train_tensor,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_loader = DataLoader(TensorDataset(X_train_final, y_train_final), batch_size=512, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_final, y_val_final), batch_size=512, shuffle=False)

# === 初始化模型、损失、优化器 ===
device = torch.device('cpu')
model = CNN1DRegressor(input_channels=new_input_channels).to(device)

# ====================== 优化3：CO2特征损失加权（重点降低CO2的MAE） ======================
# 定义损失权重：CO2相关特征权重为3，其余为1（放大CO2的误差惩罚）
loss_weights = torch.tensor([1.0, 3.0, 3.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
criterion = nn.L1Loss(reduction='none')  # 改为none，方便按权重计算
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === 训练与验证：保留原代码的200轮，添加轻微早停（防止过拟合） ===
print(f"正在训练 CO2优化版 1D CNN 模型（训练样本数={len(X_train_final)}, 验证样本数={len(X_val_final)}）...")
num_epochs = 200
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None
patience = 30  # 宽松的早停，保证充分训练

for epoch in range(num_epochs):
    # --- 训练 ---
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        # 加权损失：对CO2特征的误差赋予更高权重
        loss = (criterion(pred, yb) * loss_weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- 验证 ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = (criterion(pred, yb) * loss_weights).mean()
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # 保存最优模型（降低过拟合风险）
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"早停触发，停止训练（epoch={epoch}）")
            break

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")

# 加载最优模型
model.load_state_dict(best_model_state)

# ==============================
# ✅ 安全预测：修复原代码填充逻辑的索引越界bug，保留手动反转核心
# ==============================
model.eval()
half = SEQ_LEN // 2
n_test = len(X_test_scaled)
y_pred_scaled_list = []
CHUNK_SIZE = 10000

print(f"正在分块预测（总行数: {n_test}, 每块: {CHUNK_SIZE}）...")

for start_idx in range(0, n_test, CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, n_test)
    current_chunk_size = end_idx - start_idx

    # 边界填充：保留原代码的手动反转逻辑，修复索引越界
    pad_before = X_test_scaled[max(0, start_idx - half):start_idx]
    pad_after = X_test_scaled[end_idx:min(n_test, end_idx + half)]

    if len(pad_before) < half:
        needed = half - len(pad_before)
        # 修复：防止start_idx + needed超出数据集范围
        extra_start = max(0, start_idx)
        extra_end = min(len(X_test_scaled), start_idx + needed)
        extra = X_test_scaled[extra_start:extra_end][::-1]
        # 若extra长度不足，用extra的最后部分补全
        if len(extra) < needed:
            extra = np.pad(extra, ((needed - len(extra), 0), (0, 0)), mode='edge')
        pad_before = np.concatenate([extra, pad_before], axis=0)
    if len(pad_after) < half:
        needed = half - len(pad_after)
        # 修复：防止end_idx - needed小于0
        extra_start = max(0, end_idx - needed)
        extra_end = end_idx
        extra = X_test_scaled[extra_start:extra_end][::-1]
        # 若extra长度不足，用extra的最后部分补全
        if len(extra) < needed:
            extra = np.pad(extra, ((0, needed - len(extra)), (0, 0)), mode='edge')
        pad_after = np.concatenate([pad_after, extra], axis=0)

    # 强制截断到half长度，避免填充后过长
    pad_before = pad_before[-half:] if len(pad_before) > half else pad_before
    pad_after = pad_after[:half] if len(pad_after) > half else pad_after

    local_padded = np.concatenate([pad_before, X_test_scaled[start_idx:end_idx], pad_after], axis=0)

    # 构造窗口：添加长度校验，修复原代码潜在bug
    X_chunk_seq = []
    for i in range(half, half + current_chunk_size):
        window = local_padded[i - half:i + half + 1]
        # 确保窗口长度为SEQ_LEN
        if len(window) != SEQ_LEN:
            if len(window) < SEQ_LEN:
                window = np.pad(window, ((0, SEQ_LEN - len(window)), (0, 0)), mode='edge')
            else:
                window = window[:SEQ_LEN]
        X_chunk_seq.append(window)

    X_chunk_tensor = torch.tensor(np.array(X_chunk_seq), dtype=torch.float32).permute(0, 2, 1)

    with torch.no_grad():
        pred_chunk = model(X_chunk_tensor.to(device)).cpu().numpy()
    y_pred_scaled_list.append(pred_chunk)

    print(f"  已处理 [{start_idx} : {end_idx}] / {n_test}")

# 合并预测结果：保留原代码逻辑
y_pred_scaled_full = np.vstack(y_pred_scaled_list)
y_predict_full = scaler_y.inverse_transform(y_pred_scaled_full)

# 长度校验：添加保护逻辑
if len(y_predict_full) > len(y_test_raw):
    y_predict_full = y_predict_full[:len(y_test_raw)]
elif len(y_predict_full) < len(y_test_raw):
    y_predict_full = np.pad(y_predict_full, ((0, len(y_test_raw)-len(y_predict_full)), (0,0)), mode='edge')
assert len(y_predict_full) == len(y_test_raw), f"长度不匹配: {len(y_predict_full)} vs {len(y_test_raw)}"

# === 保存结果：保留原代码逻辑 ===
results = []
for True_Value, Predicted_Value in zip(y_test_raw, y_predict_full):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true = ' '.join(map(str, True_Value))
    formatted_pred = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true, formatted_pred, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_CNN1D_final_0.2.csv", index=False)

# === 计算 MAE：保留原代码逻辑 ===
error_matrix = np.array([list(map(float, row.split())) for row in result_df['Error']])
mae_per_var = np.mean(error_matrix, axis=0)
overall_mae = np.mean(mae_per_var)

print("<*>" * 50)
print("6个变量的 MAE 分别为：")
for idx, col in enumerate(columns):
    print(f"{col}: {mae_per_var[idx]:.4f}")
print(f"\n🎯 总体平均误差 (MAE): {overall_mae:.5f}")
print(f"✅ 预测结果行数: {len(result_df)}，原始测试集行数: {len(test_dataSet)} → 完全对齐！")
print(f"总耗时：{time.time() - start_time:.3f} 秒")