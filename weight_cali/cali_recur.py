import pandas as pd
import numpy as np
import param

# cali_recur: 读入激活压力值和测量参考值，输出回归幂函数参数列表
def recur_calculate(recur_input_path, recur_results_path):
    data = pd.read_csv(recur_input_path)
    # 存储结果的列表
    results = []

    # 遍历传感器数据
    for sensor_id in data.columns[1:]:  # 跳过第一列，因为它是索引
        # 读取数据
        sensor_data = data[sensor_id].dropna().values
        
        # 分割V和F
        V = sensor_data[0:2]  # 读入V
        F = sensor_data[2:4]  # 读入F

        if len(V) == len(F) and len(V) > 0:
            # 拟合幂函数
            alpha = np.log(F[1]/F[0]) / np.log(V[1]/V[0])
            # 存储结果
            results.append({'V1': V[0], 'F1': F[0], 'alpha': alpha})

    # 检查结果
    # print(results)

    # 将结果写入新的CSV文件
    results_df = pd.DataFrame(results)
    results_df.to_csv(recur_results_path, index=False)

# cali_recur: 读入激活电阻值和测量参考值，输出回归幂函数参数列表
def r_f_recur_calculate(recur_input_path, recur_results_path):
    data = pd.read_csv(recur_input_path)
    # 存储结果的列表
    results = []

    # 遍历传感器数据
    for sensor_id in data.columns[1:]:  # 跳过第一列，因为它是索引
        # 读取数据
        sensor_data = data[sensor_id].dropna().values
        
        # 分割R和F
        R = sensor_data[0:2]  # 读入R
        F = sensor_data[2:4]  # 读入F

        if len(R) == len(F) and len(R) > 0:
            # 取对数
            log_R = np.log(R)
            log_F = np.log(F)

            # 拟合直线
            A = np.vstack([log_F, np.ones(len(log_F))]).T
            alpha, log_k = np.linalg.lstsq(A, log_R, rcond=None)[0]

            # 存储结果
            k = np.exp(log_k)
            results.append({'Sensor': int(sensor_id) + 1, 'k': k, 'alpha': alpha})

    # 将结果写入新的CSV文件
    results_df = pd.DataFrame(results)
    results_df.to_csv(recur_results_path, index=False)

if __name__ == "__main__":
    # 定义输入输出文件路径
    # sensor_ori_path = "./cali_data/bundou_heavy.csv"
    # activation_output_path = "./cali_data/bundou_heavy_activation.csv"
    # recur_input_path = "./cali_data/bundou_recur_input2.csv"
    # recur_results_path = "./cali_data/bundou_recur_results2.csv"
    recur_input_path = param.input_csv
    recur_results_path = param.output_csv

    # 校准csv -> 激活压力值
    # sensor_ori.csv -> activation_output.csv
    # cali_pre.extract_activation(sensor_ori_path, activation_output_path, 25)

    # 激活压力值+参考压力值 -> 回归幂函数参数列表
    # recur_input.csv(手动编写) -> recur_results.csv
    # recur_calculate(recur_input_path, recur_results_path)
    r_f_recur_calculate(recur_input_path, recur_results_path)