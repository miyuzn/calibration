import os
import pandas as pd
from sensor import read_sensor_data_from_csv, save_sensor_data_to_csv
import param
import re

# Load the fitting parameters for the sensors
def load_fitting_parameters(recur_results_csv):
    df = pd.read_csv(recur_results_csv)
    params = {row['Sensor']: (row['k'], row['alpha']) for _, row in df.iterrows()}
    return params

# Main function: batch process CSV files in a directory
def batch_process_csv_files():
    exp = './weight_cali/0123-glove'
    recur_results_csv = param.output_csv  # Specify your fitting parameters CSV file here

    # Load fitting parameters once
    params = load_fitting_parameters(recur_results_csv)

    # Walk through the directory and process CSV files
    for root, _, files in os.walk(f"./{exp}/"):
        for file in files:
            if file.endswith(".csv") and file != os.path.basename(recur_results_csv) and not re.search(r"_([rR]|[fF])\.csv$", file):
                # Handle file paths
                sensor_input_path = os.path.join(root, file)
                output_path = os.path.splitext(sensor_input_path)[0] + "_f.csv"  # Modify output file path with '_output.csv' suffix

                # Read sensor data from CSV
                sensor_data_list = read_sensor_data_from_csv(sensor_input_path)
                
                # Convert voltage to resistance and resistance to pressure
                for sensor_data in sensor_data_list:
                    sensor_data.sensor_v_to_r()  # Voltage to resistance
                    sensor_data.sensor_r_to_f(params)  # Resistance to force/pressure

                # Save the converted data
                save_sensor_data_to_csv(sensor_data_list, output_path)
                print(f"Processed {file} and saved to {output_path}")


# Execute the batch processing
if __name__ == "__main__":
    batch_process_csv_files()
