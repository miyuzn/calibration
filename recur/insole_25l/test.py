# Complete code for the analysis with bar chart visualization and CSV output
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Modify the function to handle cases with one empty dataset
def analyze_fixed_range_keep_valid(sensor_file, dir1, dir2, log_min=0.0, log_max=1.0, resistance_threshold=50000):
    # Load data for both dates
    data1 = pd.read_csv(os.path.join(dir1, sensor_file))
    data2 = pd.read_csv(os.path.join(dir2, sensor_file))
    
    # Filter out points where Resistance > threshold
    data1 = data1[data1['Resistance'] <= resistance_threshold]
    data2 = data2[data2['Resistance'] <= resistance_threshold]
    
    # Convert to log scale and filter by range
    log_pressure1 = np.log10(data1['Pressure'])
    log_resistance1 = np.log10(data1['Resistance'])
    log_pressure2 = np.log10(data2['Pressure'])
    log_resistance2 = np.log10(data2['Resistance'])
    
    mask1 = (log_pressure1 >= log_min) & (log_pressure1 <= log_max)
    mask2 = (log_pressure2 >= log_min) & (log_pressure2 <= log_max)
    
    log_pressure1_filtered = log_pressure1[mask1]
    log_resistance1_filtered = log_resistance1[mask1]
    log_pressure2_filtered = log_pressure2[mask2]
    log_resistance2_filtered = log_resistance2[mask2]
    
    # Initialize results
    result = {
        "sensor": sensor_file,
        "date_1": {"r_squared": None, "std_error": None},
        "date_2": {"r_squared": None, "std_error": None},
    }
    
    # Perform regression only if valid points exist
    if len(log_pressure1_filtered) >= 2:
        slope1, intercept1, r_value1, _, std_err1 = linregress(
            log_pressure1_filtered, log_resistance1_filtered
        )
        result["date_1"] = {"r_squared": r_value1**2, "std_error": std_err1}
    
    if len(log_pressure2_filtered) >= 2:
        slope2, intercept2, r_value2, _, std_err2 = linregress(
            log_pressure2_filtered, log_resistance2_filtered
        )
        result["date_2"] = {"r_squared": r_value2**2, "std_error": std_err2}
    
    return result

# Define directories
dir_0926 = './recur/insole_25l/intermediate_results'
dir_0912 = './recur/insole_25l/intermediate_results_0912'
files_0926 = os.listdir(dir_0926)

# Analyze all sensors excluding outliers safely
filtered_results_safe = []
for sensor_file in files_0926:
    result = analyze_fixed_range_keep_valid(sensor_file, dir_0912, dir_0926)
    filtered_results_safe.append(result)

# Fix the issue with removing the 'P' prefix and converting to integers
filtered_results_safe_df = pd.DataFrame([
    {
        "Sensor": int(res["sensor"].replace('P', '').replace('_pressure_resistance.csv', '')),
        "R² (0912)": res["date_1"]["r_squared"],
        "Std Error (0912)": res["date_1"]["std_error"],
        "R² (0926)": res["date_2"]["r_squared"],
        "Std Error (0926)": res["date_2"]["std_error"],
    }
    for res in filtered_results_safe
])

# Sort DataFrame by Sensor number
filtered_results_safe_df = filtered_results_safe_df.sort_values(by='Sensor')

# Save the results to a CSV file
output_csv_path = './ppt_resource/test.csv'
filtered_results_safe_df.to_csv(output_csv_path, index=False)

# Plot results as bar charts
plt.figure(figsize=(14, 8))
x_labels = filtered_results_safe_df['Sensor'].astype(str)

# Plot R² values for both dates
bar_width = 0.4
index = np.arange(len(x_labels))

plt.bar(index - bar_width / 2, filtered_results_safe_df['R² (0912)'], bar_width, label='R² (0912)', alpha=0.8)
plt.bar(index + bar_width / 2, filtered_results_safe_df['R² (0926)'], bar_width, label='R² (0926)', alpha=0.8)

# Set titles and labels
plt.title('Comparison of R² Values (Sorted by Sensor Number, Excluding Outliers, Log Pressure Range: 0.0-1.0)')
plt.xlabel('Sensor Number')
plt.ylabel('R² Value')
plt.xticks(index, x_labels, rotation=90)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))

# Plot standard errors for both dates
plt.bar(index - bar_width / 2, filtered_results_safe_df['Std Error (0912)'], bar_width, label='Std Error (0912)', alpha=0.8)
plt.bar(index + bar_width / 2, filtered_results_safe_df['Std Error (0926)'], bar_width, label='Std Error (0926)', alpha=0.8)

# Set titles and labels
plt.title('Comparison of Standard Errors (Sorted by Sensor Number, Excluding Outliers, Log Pressure Range: 0.0-1.0)')
plt.xlabel('Sensor Number')
plt.ylabel('Standard Error')
plt.xticks(index, x_labels, rotation=90)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# Output the path to the saved CSV
print(output_csv_path)
