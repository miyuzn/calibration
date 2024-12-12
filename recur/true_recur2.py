import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
import re
import param

# 定义特征函数模型 R = kF^α
def model(F, k, alpha):
    return k * F**alpha

# 根据波动过滤数据
def filter_by_fluctuation(Pressure_values, Resistance_values, Fluctuation_values):
    min_std = np.min(Fluctuation_values)
    # 修改过滤条件：标准差小于5倍最小值或标准差大于三倍最小值但小于100
    filtered_indices = np.where((Fluctuation_values < 5 * min_std) | ((Fluctuation_values > 5 * min_std) & (Fluctuation_values < 100)))
    return Pressure_values[filtered_indices], Resistance_values[filtered_indices], Fluctuation_values[filtered_indices]

def split_data(Pressure_values, Resistance_values, Fluctuation_values):
    return Pressure_values, Resistance_values, Fluctuation_values

# 使用过滤后的点进行回归（使用所有大于2N的点）
def regression_all_points(Pressure_values, Resistance_values, Fluctuation_values):
    # 过滤数据
    Pressure_filtered, Resistance_filtered, Fluctuation_filtered = filter_by_fluctuation(Pressure_values, Resistance_values, Fluctuation_values)
    
    # 进一步过滤出 Pressure > 2N 的数据
    filtered_indices = np.where(Pressure_filtered > 2)
    Pressure_filtered = Pressure_filtered[filtered_indices]
    Resistance_filtered = Resistance_filtered[filtered_indices]
    Fluctuation_filtered = Fluctuation_filtered[filtered_indices]

    # 计算标准差的平均值
    mean_fluctuation = np.mean(Fluctuation_filtered)

    # 进行非线性回归以拟合模型
    try:
        popt, _ = curve_fit(model, Pressure_filtered, Resistance_filtered, maxfev=5000)
        # 提取拟合结果中的 k 和 α
        k, alpha = popt
        return k, alpha, mean_fluctuation
    except RuntimeError:
        print("无法拟合数据")
        return None, None, None

# 使用过滤后的点进行回归（使用2N和10N两个点）
def regression_two_points(Pressure_values, Resistance_values, Fluctuation_values):
    # 过滤数据
    # Pressure_filtered, Resistance_filtered, Fluctuation_filtered = filter_by_fluctuation(Pressure_values, Resistance_values, Fluctuation_values)
    Pressure_filtered, Resistance_filtered, Fluctuation_filtered = split_data(Pressure_values, Resistance_values, Fluctuation_values)
    # 对Pressure和Resistance排序，以便于插值
    sorted_indices = np.argsort(Pressure_filtered)
    Pressure_sorted = Pressure_filtered[sorted_indices]
    Resistance_sorted = Resistance_filtered[sorted_indices]
    Fluctuation_sorted = Fluctuation_filtered[sorted_indices]

    # 定义插值函数
    def interpolate(x, xp, yp):
        return np.interp(x, xp, yp)

    # 插值获得5N和9N处的Resistance值
    Pressure_target = [5, 9]
    Resistance_target = []
    Fluctuation_target = []

    for p in Pressure_target:
        if p < Pressure_sorted[0] or p > Pressure_sorted[-1]:
            print(f"压力 {p}N 超出数据范围，无法插值。")
            return None, None, None
        Resistance_target.append(interpolate(p, Pressure_sorted, Resistance_sorted))
        Fluctuation_target.append(interpolate(p, Pressure_sorted, Fluctuation_sorted))

    Pressure_two_points = np.array(Pressure_target)
    Resistance_two_points = np.array(Resistance_target)
    Fluctuation_two_points = np.array(Fluctuation_target)

    # 计算标准差的平均值
    mean_fluctuation = np.mean(Fluctuation_two_points)

    # 使用初始参数估计进行非线性回归
    try:
        popt, _ = curve_fit(model, Pressure_two_points, Resistance_two_points, p0=[1, 1], maxfev=5000)
        # 提取拟合结果中的 k 和 α
        k, alpha = popt
        return k, alpha, mean_fluctuation
    except RuntimeError:
        print("无法拟合5N和9N的点")
        return None, None, None

# 数据存放目录（本地）
exp = f'./recur/{param.exp}'
data_dir = f'./recur/{param.exp}/intermediate_results/'
output_file_all = f'./recur/{param.exp}/true_recur_all_points.csv'
output_file_two = f'./recur/{param.exp}/true_recur_two_points.csv'

# 获取目录下的所有CSV文件路径
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

# 初始化两个列表，用于分别存储使用所有点和仅使用两点的回归结果
results_all_points = []
results_two_points = []

# 处理每个CSV文件
for file in csv_files:
    # 从文件名中提取传感器编号（假设编号为文件名中的数字）
    sensor_number = int(re.findall(r'\d+', os.path.basename(file))[0])
    
    # 读取数据
    data = pd.read_csv(file)
    
    # 获取 Pressure, Resistance 和 Fluctuation 的值
    Pressure_values = data['Pressure'].values
    Resistance_values = data['Resistance'].values
    Fluctuation_values = data['Fluctuation'].values
    
    # 使用过滤后的点进行回归（所有大于2N的点）
    k_all, alpha_all, mean_fluctuation_all = regression_all_points(Pressure_values, Resistance_values, Fluctuation_values)
    if k_all is not None and alpha_all is not None:
        results_all_points.append([sensor_number, k_all, alpha_all, mean_fluctuation_all])
    
    # 使用过滤后的点进行回归（插值后的2N和10N两个点）
    k_two, alpha_two, mean_fluctuation_two = regression_two_points(Pressure_values, Resistance_values, Fluctuation_values)
    if k_two is not None and alpha_two is not None:
        results_two_points.append([sensor_number, k_two, alpha_two, mean_fluctuation_two])

# 检查是否有结果并输出所有点的回归结果
if results_all_points:
    # 将结果转换为 DataFrame
    results_df_all = pd.DataFrame(results_all_points, columns=['Sensor', 'k', 'alpha', 'Mean Fluctuation'])
    # 按传感器编号排序
    results_df_all = results_df_all.sort_values(by='Sensor')
    # 保存为 CSV 文件
    results_df_all.to_csv(output_file_all, index=False)
else:
    print("使用所有点的回归结果为空，没有符合条件的数据进行拟合。")

# 检查是否有结果并输出两点的回归结果
if results_two_points:
    # 将结果转换为 DataFrame
    results_df_two = pd.DataFrame(results_two_points, columns=['Sensor', 'k', 'alpha', 'Mean Fluctuation'])
    # 按传感器编号排序
    results_df_two = results_df_two.sort_values(by='Sensor')
    # 保存为 CSV 文件
    results_df_two.to_csv(output_file_two, index=False)
else:
    print("使用两点的回归结果为空，没有符合条件的数据进行拟合。")
