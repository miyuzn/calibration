import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
import re

# 定义特征函数模型 R = kF^α
def model(F, k, alpha):
    return k * F**alpha

# 数据存放目录（本地）
exp = 'insole_25r'
data_dir = f'./recur/{exp}/intermediate_results/'
output_file = f'./recur/{exp}/true_recur.csv'

# 获取目录下的所有CSV文件路径
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

# 初始化一个列表，用于存储结果
results = []

# 处理每个CSV文件
for file in csv_files:
    # 从文件名中提取传感器编号（假设编号为文件名中的数字）
    sensor_number = int(re.findall(r'\d+', os.path.basename(file))[0])
    
    # 读取数据
    data = pd.read_csv(file)
    
    # 过滤出 Pressure > 1N 的数据
    filtered_data = data[data['Pressure'] > 2]
    
    # 检查过滤后的数据是否为空
    if len(filtered_data) == 0:
        continue  # 如果没有数据满足条件，跳过这个文件
    
    # 获取 Pressure 和 Resistance 的值
    Pressure_values = filtered_data['Pressure'].values
    Resistance_values = filtered_data['Resistance'].values
    
    # 进行非线性回归以拟合模型
    try:
        popt, _ = curve_fit(model, Pressure_values, Resistance_values, maxfev=5000)
        # 提取拟合结果中的 k 和 α
        k, alpha = popt
        # 存储结果
        results.append([sensor_number, k, alpha])
    except RuntimeError:
        print(f"无法拟合传感器编号 {sensor_number} 的数据")

# 检查是否有结果
if results:
    # 将结果转换为 DataFrame
    results_df = pd.DataFrame(results, columns=['Sensor', 'k', 'alpha'])
    # 按传感器编号排序
    results_df = results_df.sort_values(by='Sensor')
    # 保存为 CSV 文件
    results_df.to_csv(output_file, index=False)
else:
    print("结果为空，没有符合条件的数据进行拟合。")
