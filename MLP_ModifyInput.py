# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy import stats

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'../../modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'../../modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
                 'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

CL = columns + noise_columns

## 查看数据缺失情况
data = train_dataSet[CL]
missingDf = data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns = ['feature', 'miss_num']
missingDf['miss_percentage'] = missingDf['miss_num'] / data.shape[0]
print("缺失值比例")
print(missingDf)

# 初始化一个字典来存储每一列的异常值比例
outlier_ratios = {}
for column in CL:
    z_scores = np.abs(stats.zscore(train_dataSet[column]))
    outliers = (z_scores > 2)
    outlier_ratio = outliers.mean()
    outlier_ratios[column] = outlier_ratio

print("*" * 30)
print("异常值的比例:")
for column, ratio in outlier_ratios.items():
    print(f"{column}: {ratio:.2%}")

# # === 关键修改：构造带噪观测 Noisy_* = True + Error ===
# noisy_cols = []
# for col in columns:
#     err_col = f'Error_{col}'
#     noisy_col = f'Noisy_{col}'
#     train_dataSet[noisy_col] = train_dataSet[col] + train_dataSet[err_col]
#     test_dataSet[noisy_col] = test_dataSet[col] + test_dataSet[err_col]
#     noisy_cols.append(noisy_col)

# 划分训练集和测试集（使用带噪观测作为输入）
# X_train = train_dataSet[noisy_cols].values.astype(np.float32)
# y_train = train_dataSet[columns].values.astype(np.float32)
#
# X_test = test_dataSet[noisy_cols].values.astype(np.float32)
# y_test = test_dataSet[columns].values.astype(np.float32)

X_train = train_dataSet[noise_columns].values.astype(np.float32)
y_train = train_dataSet[columns].values.astype(np.float32)

X_test = test_dataSet[noise_columns].values.astype(np.float32)
y_test = test_dataSet[columns].values.astype(np.float32)
# 特征标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# === 定义轻量 MLP 模型（快速 + 高精度）===
print("开始训练 MLP 模型...")
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=False
)

# 训练
mlp.fit(X_train_scaled, y_train)
print(f"✅ MLP 训练完成，耗时: {time.time() - start_time:.2f} 秒")

# 预测
y_predict = mlp.predict(X_test_scaled)

# 保存结果（保持你原来的格式）
results = []
for true_val, pred_val in zip(y_test, y_predict):
    error = np.abs(true_val - pred_val)
    formatted_true = ' '.join(map(str, true_val))
    formatted_pred = ' '.join(map(str, pred_val))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true, formatted_pred, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_MLP1.csv", index=False)

print("<*>" * 50)

# 从 CSV 读取并计算平均 MAE（兼容你原有逻辑）
data = pd.read_csv("result_MLP1.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()
overall_mae = means.mean()

print("6个变量的 MAE 分别为：\n", means)
print(f"\n🎯 总体 MAE: {overall_mae:.5f}")

end_time = time.time()
print(f"总耗时：{end_time - start_time:.3f} 秒")