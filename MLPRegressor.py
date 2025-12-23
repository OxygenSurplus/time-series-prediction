# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy import stats

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'../../modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'../../modified_数据集Time_Series662_detail.dat')

# columns表示原始列，noise_columns表示添加噪声的列
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

CL = columns + noise_columns

## 查看数据缺失情况
data = train_dataSet[CL]
missingDf = data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns = ['feature', 'miss_num']
missingDf['miss_percentage'] = missingDf['miss_num'] / data.shape[0]  # 缺失值比例
print("缺失值比例")
print(missingDf)

# 初始化一个字典来存储每一列的异常值比例
outlier_ratios = {}

# 遍历每一列
for column in CL:
    col_data = train_dataSet[column]
    if col_data.isnull().all():
        outlier_ratios[column] = 0.0
        continue
    clean_data = col_data.dropna()
    z_scores = np.abs(stats.zscore(clean_data))
    outliers_mask = np.zeros(len(col_data), dtype=bool)
    outliers_mask[col_data.notna()] = (z_scores > 2)
    outlier_ratio = outliers_mask.mean()
    outlier_ratios[column] = outlier_ratio

print("*" * 30)
# 打印结果
print("异常值的比例:")
for column, ratio in outlier_ratios.items():
    print(f"{column}: {ratio:.2%}")

# 划分训练集和测试集
X_train = train_dataSet[noise_columns].values
y_train = train_dataSet[columns].values

X_test = test_dataSet[noise_columns].values
y_test = test_dataSet[columns].values

# ========== 必须标准化输入 X 和输出 y（对神经网络强烈推荐）==========
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
# 注意：预测后需要反变换回原始尺度

# ========== 使用 MLPRegressor ==========
# 网络结构示例：(128, 64) 表示两层隐藏层，分别有128和64个神经元
# 你可以根据数据量调整；若数据量小，建议用更小网络防止过拟合
model_adj = MLPRegressor(
    hidden_layer_sizes=(128, 64),   # 隐藏层结构
    activation='relu',              # 激活函数
    solver='adam',                  # 优化器
    alpha=0.001,                    # L2 正则化强度
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=500,                   # 最大迭代次数（注意不是epoch数）
    random_state=217,
    early_stopping=True,            # 启用早停
    validation_fraction=0.1,        # 验证集比例
    n_iter_no_change=10,            # 早停耐心轮数
    verbose=False                   # 若想看训练过程可设为 True
)

# 模型训练（使用标准化后的 y）
model_adj.fit(X_train_scaled, y_train_scaled)

# 预测（先预测标准化后的 y，再反变换）
y_pred_scaled = model_adj.predict(X_test_scaled)
y_predict = scaler_y.inverse_transform(y_pred_scaled)

# 保存结果
results = []
for True_Value, Predicted_Value in zip(y_test, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_MLP.csv", index=False)

print("<*>" * 50)

# 从CSV文件读取并计算平均绝对误差（MAE）
data = pd.read_csv("result_MLP.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric, errors='coerce')
means = numbers.mean()
print("6个输出变量的平均绝对误差（MAE）为：\n", means)
print("总体平均 MAE:", means.mean())

end_time = time.time()
print(f"总耗时：{end_time - start_time:.3f}秒")