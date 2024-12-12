import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# 假设param.py中有如下参数定义
# 请根据实际情况修改
class param:
    data_dir = './weight_cali/exp/1203/23l'          # 原始数据目录，存放500g-1.csv, 500g-2.csv, 1kg-1.csv, 1kg-2.csv等
    output_csv = f'{data_dir}/output.csv'  # 最终输出回归结果文件

############################
# 以下为原cali_act.py中的函数 #
############################

def sma_filter(input, window_size=10):
    result = []

    if len(input) < window_size:
        raise ValueError("SMA滤波器错误: 数据长度小于窗口长度.")

    for i in range(0, window_size):
        window = input[:i + 1]
        sma = sum(window) / (i + 1)
        result.append(sma)

    for i in range(window_size, len(input) - window_size):
        window = input[i:i + window_size]
        sma = sum(window) / window_size
        result.append(sma)

    for i in range(len(input) - window_size, len(input)):
        window = input[i:]
        sma = sum(window) / (len(input) - i)
        result.append(sma)

    return result

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

        if activation_start != -1 and activation_end != -1:
            activation_mean = np.mean(sensor_data[activation_start:activation_end])
            activation_duration = activation_end - activation_start
            activation_diff = max_value - activation_mean 
        else:
            activation_mean = np.nan
            activation_duration = 0
            activation_diff = 0

        activation_means.append(activation_mean)
        activation_durations.append(activation_duration)
        activation_diffs.append(activation_diff)

    return np.array(activation_means), np.array(activation_durations), np.array(activation_diffs)

def calculate_activation_values(pressure_list):
    pressure_sma = []
    for i in pressure_list:
        pressure_sma.append(sma_filter(i, 100))

    activation_values = find_activation_mean(np.array(pressure_sma),0.75,50)
    activation_real, activation_durations, activation_diffs = activation_values

    #draw_pressure(pressure_sma, title="SMA Pressure")
    return activation_real, activation_durations

def v_list_to_r(v_list):
    v_ref = 0.312
    R1 = 5000
    for i in range(len(v_list)):
        if v_list[i] > v_ref:
            v_list[i] = R1 * v_ref / (v_list[i] - v_ref)
        else:
            v_list[i] = float('inf')
    return v_list

##############################
# 以下为原cali_recur.py中的函数 #
##############################

def r_f_recur_calculate(data_df, output_csv):
    # data_df格式：第一行R，第二行R, ... 最后两行为F对应的值
    # 假设数据格式与原input.csv类似：第一行'R'只是标记行，后面是具体载荷对应的R数据行，再下面是F行
    # 我们需要从data_df中区分出传感器列并进行拟合
    # data_df的列格式为[0,1,2...,34]：传感器ID
    # 行格式：
    #   R
    #   507g行(为该负载下R的值)
    #   1009g行(为该负载下R的值)
    #   F
    #   507g行(对应F)
    #   1009g行(对应F)

    # 假设我们最终的数据结构为：
    # index: [ 'R', '507g', '1009g', 'F', '507g', '1009g' ]   或简化成两组R和F数据
    # 实际上，我们只需要两组R和F来进行回归：
    # R行有两行数据：一行为低载荷对应R，一行为高载荷对应R
    # F行有两行数据：一行为低载荷F，一行为高载荷F
    # 列为各传感器
    # 那么对于每个传感器，我们有(R1,F1)和(R2,F2)，拟合幂函数R = k * (F^alpha)
    # 由于源代码是对于V和F的关系，我们这里是R和F的关系，需要按照r_f_recur_calculate中的逻辑拟合

    # 提取数据
    # 假设data_df顺序是：
    # 行： ['R'] (标记), '507g', '1009g', ['F'](标记), '507g', '1009g'
    # 我们跳过标记行，直接抓载荷行与力行。
    R_values_507g = data_df.loc['507g(R)'].values.astype(float)
    R_values_1009g = data_df.loc['1009g(R)'].values.astype(float)
    F_values_507g = data_df.loc['507g(F)'].values.astype(float)
    F_values_1009g = data_df.loc['1009g(F)'].values.astype(float)

    # 开始拟合
    results = []
    for sensor_id in range(len(R_values_507g)):
        R = np.array([R_values_507g[sensor_id], R_values_1009g[sensor_id]])
        F = np.array([F_values_507g[sensor_id], F_values_1009g[sensor_id]])

        if len(R) == 2 and len(F) == 2:
            # 取对数拟合直线 log(R) = alpha * log(F) + log(k)
            log_R = np.log(R)
            log_F = np.log(F)

            A = np.vstack([log_F, np.ones(len(log_F))]).T
            alpha, log_k = np.linalg.lstsq(A, log_R, rcond=None)[0]
            k = np.exp(log_k)
            # sensor_id从0开始计数，传感器编号为sensor_id+1
            results.append({'Sensor': sensor_id+1, 'k': k, 'alpha': alpha})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

########################################
# 以下为整合后的主流程，实现全自动化步骤 #
########################################

def process_files_for_load(directory, load_pattern, sensor_num=35):
    """
    对给定负载（如'500g'或'1kg'）对应的多个文件（如500g-1.csv, 500g-2.csv）进行处理。
    计算其激活电阻值R，并对同一负载下的多个文件求平均，最终返回平均R数组。
    """
    # 匹配类似 "500g-1.csv", "500g-2.csv" 等文件
    pattern = re.compile(fr"{load_pattern}-\d+\.csv$")
    R_list_per_file = []

    for filename in os.listdir(directory):
        if pattern.search(filename):
            input_csv = os.path.join(directory, filename)
            pressure_time_list = np.array(read_pressure_data_from_csv(input_csv, sensor_num))
            pressure_list = pressure_time_list.T
            activation_real, _ = calculate_activation_values(pressure_list)
            # 转换为电阻
            activation_resistance = v_list_to_r(activation_real / 1000.0) # activation_real为mV,除1000得V
            R_list_per_file.append(activation_resistance)

    if len(R_list_per_file) == 0:
        raise ValueError(f"没有找到匹配{load_pattern}的文件。")

    # 对同负载下多组数据求平均
    R_list_per_file = np.array(R_list_per_file)
    avg_R = np.nanmean(R_list_per_file, axis=0)
    return avg_R


def main():
    # 假设我们有两个载荷：约500g和1kg，对应力值需要用户根据实验数据提供
    # 此处使用题中给出的范例:
    # 507g对应4.967N (接近500g)
    # 1009g对应9.888N (接近1kg)
    # 请根据实际实验数据修改
    sensor_num = 35

    # 尝试对500g与1kg文件进行处理（根据用户给出的例子修改）
    # 在原始数据文件中可能是500g-1.csv, 500g-2.csv ... 1kg-1.csv, 1kg-2.csv ...
    # 用户提供的act.csv中示例表明使用了507g与1009g，这里保持一致
    
    load1_pattern = "500g"  # 原文示例文件名，如"500g-1.csv"
    load2_pattern = "1kg"   # 类似"1kg-1.csv"
    # 实际上用户最终希望使用507g和1009g数据与力值，这里假设文件命名为500g-*.csv,1kg-*.csv
    # 如果实际命名不一致，需要相应调整load_pattern或命名文件

    R_500g = process_files_for_load(param.data_dir, load1_pattern, sensor_num=sensor_num)
    R_1kg  = process_files_for_load(param.data_dir, load2_pattern, sensor_num=sensor_num)

    # 对应的F值（用户给定）
    F_500g = 4.967
    F_1kg = 9.888

    # 组织数据成DataFrame，格式类似原input.csv
    # 我们需要两行表示R（500g和1kg对应的R），两行表示F（500g和1kg对应的F）
    # 行索引我们用 '507g(R)', '1009g(R)', '507g(F)', '1009g(F)'
    # 列为传感器索引0到34
    sensors = list(range(sensor_num))
    data = {
        # R行
        '507g(R)': R_500g,
        '1009g(R)': R_1kg,
        # F行
        '507g(F)': np.array([F_500g]*sensor_num),
        '1009g(F)': np.array([F_1kg]*sensor_num)
    }

    # 转置使列为传感器，行为上述标签
    df_for_recur = pd.DataFrame(data, index=sensors).T

    # df_for_recur现在行是'507g(R)'等，列是传感器ID
    # 为了与回归函数统一处理，将DataFrame行索引设置如上，并传入回归函数
    # 回归函数需要从中提取R和F数据
    # 当前df_for_recur的结构：
    # 行： 507g(R), 1009g(R), 507g(F), 1009g(F)
    # 列：0,1,2,...,34
    # 正好可以直接使用拟合函数进行处理

    # 执行回归计算并输出结果
    r_f_recur_calculate(df_for_recur, param.output_csv)

    print("回归计算完成，结果已输出到：", param.output_csv)


if __name__ == "__main__":
    main()
