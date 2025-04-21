import os
import pandas as pd
from sensor import read_sensor_data_from_csv, save_sensor_data_to_csv

exp = './weight_cali/exp/0416/25l'  # 数据存放目录（本地）

# 加载传感器的拟合参数
def load_fitting_parameters(true_recur):
    df = pd.read_csv(true_recur)
    params = {row['Sensor']: (row['k'], row['alpha']) for _, row in df.iterrows()}
    return params

# 主函数：批量处理目录中的CSV文件
def batch_process_csv_files(exp):
    exps = [exp]
    for exp in exps:
        recur_results_csv = f"{exp}/output.csv"
        output_dir = f"{exp}/force_output/"
        os.makedirs(output_dir, exist_ok=True)

        # 遍历当前目录和子目录中的CSV文件
        for root, _, files in os.walk(f"{exp}/input/"):
            # 如果当前目录是输出文件夹，跳过
            if root.startswith(output_dir):
                continue

            for file in files:
                if file.endswith(".csv") and file not in ["output.csv"]:
                    
                    # 加载拟合参数
                    params = load_fitting_parameters(recur_results_csv)
                    
                    # 文件路径处理
                    sensor_input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(sensor_input_path, ".")
                    relative_path = os.path.relpath(relative_path, start=os.path.commonpath([output_dir, relative_path]))
                    output_path = os.path.join(output_dir, relative_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # 读取传感器数据
                    sensor_data_list = read_sensor_data_from_csv(sensor_input_path,10)
                    
                    # 使用sensor模块的v_to_r和r_to_f方法进行转换
                    for sensor_data in sensor_data_list:
                        sensor_data.sensor_v_to_r()  # 电压转电阻
                        sensor_data.sensor_r_to_f(params)  # 电阻转压力

                    # 保存转换后的数据
                    save_sensor_data_to_csv(sensor_data_list, output_path)
                    print(f"Processed {file} and saved to {output_path}")

# 执行批量处理
if __name__ == "__main__":
    batch_process_csv_files(exp=exp)
