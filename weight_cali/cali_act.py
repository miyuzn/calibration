import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import param

# cali_pre: 读入测量csv计算激活电压值和阻值

# SMA滤波器
def sma_filter(input, window_size=10):
    result = []

    if len(input) < window_size:
        raise ValueError("SMA滤波器错误: 数据长度小于窗口长度.")

    # 开始部分，窗口从1增长到window_size
    for i in range(0, window_size):
        window = input[:i + 1]
        sma = sum(window) / (i + 1)
        result.append(sma)

    # 中间部分，窗口为window_size
    for i in range(window_size, len(input) - window_size):
        window = input[i:i + window_size]
        sma = sum(window) / window_size
        result.append(sma)

    # 结尾部分，窗口从window_size减小到1
    for i in range(len(input) - window_size, len(input)):
        window = input[i:]
        sma = sum(window) / (len(input) - i)
        result.append(sma)

    return result

# 读取压力数据函数
# CSV -> 数据列表
def read_pressure_data_from_csv(filepath, p_num=25):
    pressure_data_list = []

    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # 获取列名并检查是否有 Timestamp 列
        fieldnames = reader.fieldnames
        if fieldnames[0] != 'Timestamp':
            # 如果第一列不是 Timestamp，则跳过一列重新读取
            #next(reader)
            reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract timestamp and its millisecond part
            timestamp = float(row['Timestamp'])
            # Extract sensor values
            pressure_sensors = [int(row[f'P{i}']) for i in range(1, p_num + 1)]

            # Add it to the list
            pressure_data_list.append(pressure_sensors)

    return pressure_data_list

def draw_pressure(pressure_data, title='Pressure Value', xlabel='Package Count', ylabel='Voltage Value'):
    for i in range(len(pressure_data)):
        plt.plot(pressure_data[i], label=f'P{i+1}')
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# 计算初始稳定状态的值
def stable_avg(pressure_data, stable_window=1000):
    stable_list = []
    for i in pressure_data:
        stable_list.append(np.average(i[0:stable_window]))
    return np.array(stable_list)  

# 找到激活状态的均值
def find_activation_mean(data, percent_threshold=0.75, min_duration=150):
    num_sensors, num_samples = data.shape
    activation_means = []
    activation_durations = []
    activation_diffs = []

    for sensor in range(num_sensors):
        sensor_data = data[sensor, :]
        min_value = np.average(sensor_data[:500])
        max_value = np.max(sensor_data)
        threshold = (max_value - min_value) * percent_threshold + min_value

        activation_start = -1
        activation_end = -1

        # 找到激活状态的起始和结束位置
        for i in range(num_samples):
            if sensor_data[i] > threshold:
                if activation_start == -1:
                    activation_start = i
            else:
                if activation_start != -1:
                    if i - activation_start >= min_duration:
                        activation_end = i
                        break
                    else:
                        activation_start = -1

        # 如果找到激活状态，计算平均读数
        if activation_start != -1 and activation_end != -1:
            activation_mean = np.mean(sensor_data[activation_start:activation_end])
            activation_duration = activation_end - activation_start
            activation_diff = max_value - activation_mean 
        else:
            activation_mean = np.nan  # 如果没有找到有效的激活状态，返回 NaN
            activation_duration = 0
            activation_diff = 0

        activation_means.append(activation_mean)
        activation_durations.append(activation_duration)
        activation_diffs.append(activation_diff)

    return np.array(activation_means), np.array(activation_durations), np.array(activation_diffs)

# 计算激活电压值（单位:mV）
def calculate_activation_values(pressure_list):
    # 通过sma滤波器
    pressure_sma = []
    for i in pressure_list:
        pressure_sma.append(sma_filter(i, 100))
    # 计算初始稳定值
    # stable_list = np.array(stable_avg(pressure_sma))
    # 修正后的读数
    # pressure_fixed = pressure_sma - stable_list[:, np.newaxis]
    # 计算激活时读数
    activation_values = find_activation_mean(np.array(pressure_sma),0.75,50)
    # 加回修正值，得到激活时平均压力读数
    activation_real, activation_durations, activation_diffs = activation_values

    # print(activation_real)
    # print(activation_durations)
    print(activation_diffs)
    #draw_pressure(pressure_list, title="Non-SMA Pressure")
    draw_pressure(pressure_sma, title="SMA Pressure")
    #draw_pressure(pressure_fixed, title="Fixed Pressure")
    
    return activation_real, activation_durations

def v_list_to_r(v_list):
    # 设定参考电压V_ref和分压电阻R1的值
    v_ref = 0.312
    R1 = 5000
    for i in range(len(v_list)):
        # 将电压值更新为电阻值
        # current_v = v_list[i] / 1000
        if v_list[i] > v_ref:
            v_list[i] = R1 * v_ref / (v_list[i] - v_ref)
        else:
            v_list[i] = float('inf')

    return v_list

# 保存激活阻值到csv文件中
def save_activation_resistance_to_csv(activation_voltage_list, output_name):
    # 转换激活电压值到激活阻值
    activation_resistance = v_list_to_r(activation_voltage_list)

    # 输出激活读数和持续时间到文件中
    new_data = pd.DataFrame([activation_resistance])

    # 检查文件是否存在
    if os.path.exists(output_name):
        # 如果文件存在，追加新数据并不写入列名
        new_data.to_csv(output_name, mode='a', header=False, index=False)
    else:
        # 如果文件不存在，创建新文件并写入列名
        new_data.to_csv(output_name, mode='w', header=True, index=False)

# 保存激活电压数值到csv文件中
def save_activation_value_to_csv(activation_real, activation_durations, output_name):
    # 单位转换
    activation_real /= 1000 # mV -> V

    # 输出激活读数和持续时间到文件中
    new_data = pd.DataFrame([activation_real])

    # 检查文件是否存在
    if os.path.exists(output_name):
        # 如果文件存在，追加新数据并不写入列名
        new_data.to_csv(output_name, mode='a', header=False, index=False)
    else:
        # 如果文件不存在，创建新文件并写入列名
        new_data.to_csv(output_name, mode='w', header=True, index=False)

# 根据回归参数将电压转换到压力
# 输入单位: mV
def v_to_f(v, sensor_id, df):
    v1 = df.V1[sensor_id]
    f1 = df.F1[sensor_id]
    alpha = df.alpha[sensor_id]

    # 计算F值
    F = f1 * ((v * 0.001 / v1) ** alpha)
    return F

# 将电压列表转换为压力列表
def pressure_list_to_f(pressure_list, recur_results_path):
    # 读取结果CSV文件
    results_df = pd.read_csv(recur_results_path)

    voltage_list = []

    for i in pressure_list:
        tmp_list = []
        for j in range(len(i)):
            f = v_to_f(i[j], j, results_df)
            tmp_list.append(f)
        voltage_list.append(tmp_list)
    return voltage_list

# 从csv文件提取平均激活值函数
def extract_activation(input_csv, output_csv, sensor_num):
    # 从csv读入压力列表数据
    pressure_time_list = np.array(read_pressure_data_from_csv(input_csv, sensor_num))
    pressure_list = pressure_time_list.T

    # 计算激活读数
    activation_real, activation_durations = calculate_activation_values(pressure_list)
    # print(activation_real)
    # print(activation_durations)

    # 将数据写入文件
    save_activation_value_to_csv(activation_real, activation_durations, output_csv)
    save_activation_resistance_to_csv(activation_real, output_csv)
    


def process_specific_csv_files_in_directory(directory, output_csv, sensor_num):
    # 定义文件名匹配模式，匹配-1.csv、-2.csv等
    pattern = re.compile(r"-\d+\.csv$")
    
    # 获取目录中所有符合条件的CSV文件
    for filename in os.listdir(directory):
        if pattern.search(filename):
            input_csv = os.path.join(directory, filename)
            print(f"Processing file: {input_csv}")
            
            # 使用extract_activation处理每个符合条件的文件
            extract_activation(input_csv, output_csv, sensor_num)

if __name__ == "__main__":
    # 定义参数
    sensor_num = 35
    activation_output_path = param.act_csv

    # 调用处理函数
    process_specific_csv_files_in_directory(param.data_dir, activation_output_path, sensor_num)