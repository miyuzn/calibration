import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.stats import linregress

# 解析时间字符串为秒
def parse_time_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

def align_rf(r_path, f_path):
    data1 = pd.read_csv(r_path)
    data2 = pd.read_csv(f_path)
    prefix = os.path.splitext(os.path.basename(r_path))[0]

    data1['Time'] = data1['Time'].apply(parse_time_to_seconds)
    data2['Time'] = pd.to_numeric(data2['Time'])

    # 对齐时间范围
    start_time = max(data1['Time'].min(), data2['Time'].min())
    end_time = min(data1['Time'].max(), data2['Time'].max())

    # 生成对齐后的时间序列
    aligned_time = np.arange(start_time, end_time, 0.5)

    # 进行线性插值
    interp_func1 = interp1d(data1['Time'], data1['Resistance'], kind='linear', fill_value='extrapolate')
    interp_func2 = interp1d(data2['Time'], data2['Force'], kind='linear', fill_value='extrapolate')

    # 对齐后的数据
    aligned_resistance = interp_func1(aligned_time)
    aligned_force = interp_func2(aligned_time)

    # 创建对齐后的数据框
    aligned_data = pd.DataFrame({'Time': aligned_time, 'Force': aligned_force, 'Resistance': aligned_resistance})

    # 保存到新的CSV文件
    output_path = f'./P11_test/{prefix}_aligned.csv'
    aligned_data.to_csv(output_path, index=False)
    return output_path

def recur_power(aligned_csv_path):
    # 读取对齐后的数据
    aligned_data = pd.read_csv(aligned_csv_path)

    # 提取力和电阻数据
    force = aligned_data['Force']
    resistance = aligned_data['Resistance']

    # 取对数
    log_force = np.log(force)
    log_resistance = np.log(resistance)

    # 进行线性回归
    slope, intercept, r_value, p_value, std_err = linregress(log_force, log_resistance)

    # 计算拟合参数
    alpha = slope
    log_k = intercept
    k = np.exp(log_k)
    return k, alpha
    
if __name__ == "__main__":
    sensor_num = 'P3'
    # 读取R/F数据，生成对齐后的csv文件
    r_path = f'./{sensor_num}/glove_test.csv'
    f_path = f'./{sensor_num}/glove_test_N.csv'
    aligned_path = align_rf(r_path, f_path)

    # 根据对齐后的csv文件，回归出幂函数k和α的值
    k, alpha = recur_power(aligned_path)
    print(f"k:{k}, alpha:{alpha}")
