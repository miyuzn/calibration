import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

# 定义函数来读取和处理力与电阻数据，并保存中间结果
def load_data_and_save_intermediate(ori_data_dir, output_dir, sensor_num):
    sensors_data = {}
    
    for sensor_id in range(1, sensor_num + 1):
        # 读取力数据
        force_files = [f"{sensor_id}.{i}f.csv" for i in range(1, 4)]
        resistance_files = [f"{sensor_id}.{i}r.csv" for i in range(1, 4)]
        
        pressure_data_list = [
            pd.read_csv(os.path.join(ori_data_dir, f), skiprows=13, encoding='latin1') for f in force_files
        ]
        resistance_data_list = [
            pd.read_csv(os.path.join(ori_data_dir, f), skiprows=31, encoding='latin1') for f in resistance_files
        ]
        
        aligned_datasets = []
        for i in range(3):
            pressure_data = pressure_data_list[i]
            pressure_values = pressure_data.iloc[:, 0].values
            pressure_time = 0.5 * np.arange(len(pressure_values))

            resistance_data = resistance_data_list[i].iloc[:, [1, 3]]
            resistance_data.columns = ['Time', 'Resistance']
            time_parts = resistance_data['Time'].str.split(':', expand=True).astype(float)
            resistance_time = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
            resistance_values = resistance_data['Resistance'].values

            # 对电阻数据进行插值，以匹配压力时间
            interp_func = interp1d(resistance_time, resistance_values, kind='linear', fill_value="extrapolate")
            aligned_resistance_values = interp_func(pressure_time)

            aligned_data = pd.DataFrame({
                'Time (s)': pressure_time,
                'Pressure': pressure_values,
                'Resistance': aligned_resistance_values
            })
            aligned_datasets.append(aligned_data)
        
        # 计算数据集的平均值
        min_length = min([len(aligned_datasets[i]) for i in range(3)])
        for i in range(3):
            aligned_datasets[i] = aligned_datasets[i].iloc[:min_length]

        average_pressure = np.mean([aligned_datasets[i]['Pressure'].values for i in range(3)], axis=0)
        average_resistance = np.mean([aligned_datasets[i]['Resistance'].values for i in range(3)], axis=0)

        # 存储平均数据
        sensors_data[f'P{sensor_id}'] = {
            'Pressure': average_pressure,
            'Resistance': average_resistance
        }
        
        # 保存中间结果到CSV文件
        intermediate_df = pd.DataFrame({
            'Pressure': average_pressure,
            'Resistance': average_resistance
        })
        intermediate_df.to_csv(os.path.join(output_dir, f'P{sensor_id}_pressure_resistance.csv'), index=False)
    
    return sensors_data

def calculate_percentage_errors(sensor_data, recur_result_df, result_path):
    pressures_to_evaluate = np.linspace(2, 10, 17)  # 在2-10N范围内取16个点
    
    # 定义输出文件路径
    output_path = os.path.join(current_dir, result_path)
    
    with open(output_path, mode='w', newline='') as file:
        # 动态获取方法名称
        methods = [col for col in recur_result_df.columns if 'Unnamed' not in col and col != 'Sensor' and col != 'Unnamed: 0']
        
        for method in methods:
            # 输出方法名
            file.write(f"Method: {method}\n")
            
            method_results = []
            for idx, row in recur_result_df.iterrows():
                if row['Unnamed: 0'] == 'Sensor':
                    continue  # 跳过标题行

                sensor_id = str(row['Unnamed: 0'])
                true_data = sensor_data[sensor_id]
                interp_func = interp1d(true_data['Pressure'], true_data['Resistance'], kind='linear', fill_value="extrapolate")
                true_resistances = interp_func(pressures_to_evaluate)

                percentage_errors = []
                for i, pressure in enumerate(pressures_to_evaluate):
                    true_value = true_resistances[i]
                    k = float(row[method])
                    alpha_column = f'Unnamed: {list(recur_result_df.columns).index(method) + 1}'
                    alpha = float(row[alpha_column])
                    predicted_value = k * pressure ** alpha
                    percentage_error = 100 * abs(predicted_value - true_value) / true_value
                    percentage_errors.append(percentage_error)
                
                # 计算标准差
                err_avg = np.average(percentage_errors)
                
                # 每个传感器形成一行，包含10个力值下的百分比误差和标准差
                method_results.append([sensor_id] + percentage_errors + [err_avg])
            
            # 将结果转换为DataFrame，并将10个力值和标准差作为列
            columns = ['Sensor'] + [f'{p}N' for p in pressures_to_evaluate] + ['Avg Err']
            method_df = pd.DataFrame(method_results, columns=columns)
            
            # 写入CSV文件
            method_df.to_csv(file, index=False)
            file.write("\n")  # 添加空行以区分不同方法的结果

# 实验类型
exp = "insole_25r"

# 定义相对路径，假设脚本与Force和Resistance文件夹在同一级目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
ori_data_dir = os.path.join(current_dir, f"{exp}/data")

# 创建一个目录来存储中间结果
output_dir = os.path.join(current_dir, f"{exp}/intermediate_results")
os.makedirs(output_dir, exist_ok=True)

# 定义输入输出文件目录
input_path = f'./{exp}/recur_input.csv'
result_path = f'./{exp}/recur_output.csv'
sensor_num = 35

# 加载数据并保存中间结果
sensor_data = load_data_and_save_intermediate(ori_data_dir, output_dir, sensor_num)

# 加载recur_result.csv文件
recur_result_path = os.path.join(current_dir, input_path)
recur_result_df = pd.read_csv(recur_result_path)

# 执行计算并输出结果
calculate_percentage_errors(sensor_data, recur_result_df, result_path)