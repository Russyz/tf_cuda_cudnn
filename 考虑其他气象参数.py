import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 数据加载与预处理      
#fold_dir = r'C:\Users\39736\OneDrive\专利\专利代码'
fold_dir = r'C:\Users\zhangcsh\OneDrive\专利\专利代码'
file_name = "/Mast_2023-06-06_00-00-00_2024-01-26_00-00-00.csv"
data_mast = pd.read_csv(fold_dir+'/'+file_name,sep=';')
#WS1_100_Mean = data_mast['WS1_100_Mean'].values.reshape(-1, 1)
WS1_035_Mean = data_mast['WS1_035_Mean'].values.reshape(-1, 1)
Baro_035_Mean = data_mast['Baro_035_Mean'].values.reshape(-1, 1)
Hum_035_Mean = data_mast['Hum_035_Mean'].values.reshape(-1, 1)
Rain_Mean = data_mast['Rain_Mean'].values.reshape(-1, 1)
Temp_035_Mean = data_mast['Temp_035_Mean'].values.reshape(-1, 1)
# 多参数
multi_para = np.concatenate((WS1_035_Mean, Baro_035_Mean, Hum_035_Mean, Rain_Mean, Temp_035_Mean), axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
# 归一化
data_scaled = scaler.fit_transform(multi_para)
time = pd.to_datetime(data_mast['date']+" "+data_mast['time'])

# 构建滑动窗口数据集
N = 144*10  # 历史窗口长度
M = 6*1  # 预测长度
X, Y, T = [], [], []
for i in range(len(data_scaled)-N-M+1):
    X.append(data_scaled[i:i+N])
    Y.append(data_scaled[i+N:i+N+M,0])
    T.append(time[i+N:i+N+M])
X = np.array(X)
Y = np.array(Y)
T = np.array(T)
# 划分数据集
train_size = int(0.95 * len(X))
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]
time_test = T[train_size:]

# 定义LSTM模型
model = Sequential()
model.add(LSTM(64, input_shape=(N, 5)))
model.add(Dense(M))
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train, Y_train, epochs=3, batch_size=64, validation_split=0.1)

# 预测与反归一化
predictions = model.predict(X_test)
# 扩展预测结果以匹配归一化器维度
predictions_extended = np.zeros((predictions.shape[0], M, 5))
predictions_extended[:, :, 0] = predictions
predictions = scaler.inverse_transform(predictions_extended.reshape(-1, 5))[:, 0].reshape(-1, M)

# 处理Y_test
Y_test_extended = np.zeros((Y_test.shape[0], M, 5))
Y_test_extended[:, :, 0] = Y_test
Y_test = scaler.inverse_transform(Y_test_extended.reshape(-1, 5))[:, 0].reshape(-1, M)

# 计算RMSE
mse = mean_squared_error(Y_test, predictions)
mae = mean_absolute_error(Y_test, predictions)
print(f"RMSE: {np.sqrt(mse)}")
print(f"MAE: {mae}")

plt.figure(figsize=(12, 6))
plt.plot(time_test.flatten(), Y_test.flatten(), '+', label='True Wind Speed', color='blue')
plt.plot(time_test.flatten(), predictions.flatten(), '.',label='Predicted Wind Speed', color='red')
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Prediction')
plt.legend()
plt.show()
