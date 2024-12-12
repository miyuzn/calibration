import sensor
import os
import pandas as pd
import param

# 读取传感器的拟合参数
def load_fitting_parameters(true_recur):
    df = pd.read_csv(true_recur)
    params = {row['Sensor']: (row['k'], row['alpha']) for _, row in df.iterrows()}
    return params

# 处理目录下所有csv文件，包括子目录
def v_csv_to_r(input_dir, output_dir, sensor_num, param_file):
    params = load_fitting_parameters(param_file)
    
    for dirpath, _, files in os.walk(input_dir):  # 遍历输入目录及其子目录
        for file in files:
            if file.endswith('.csv') and not file.startswith('F_'):
                # 生成输入文件的完整路径
                input_csv = os.path.join(dirpath, file)
                
                # 生成相应输出目录的路径
                relative_path = os.path.relpath(dirpath, input_dir)
                output_dir_with_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_dir_with_subdir, exist_ok=True)  # 确保输出目录存在
                
                # 生成输出文件的完整路径
                output_csv = os.path.join(output_dir_with_subdir, f'F_{file}')
                
                # 读取并处理数据
                s = sensor.read_sensor_data_from_csv(input_csv, sensor_num)
                for i in s:
                    i.sensor_v_to_r()
                    i.sensor_r_to_f(params)  # 将电阻转换为压力
                
                # 保存处理后的数据
                sensor.save_sensor_data_to_csv(s, output_csv)

if __name__ == "__main__":
    input_dir = ""
    output_dir = ""
    true_recur = param.true_all_points_path  # 传感器拟合参数的CSV文件
    v_csv_to_r(input_dir, output_dir, 35, true_recur)
