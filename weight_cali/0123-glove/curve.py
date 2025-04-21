import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the uploaded CSV file to inspect its structure
file_path = './weight_cali/exp/0123-glove/output.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand the format of the data
data.head()

# Generate force values from 0 to 10 N
force_range = np.linspace(1, 10, 500)  # Start at 0.1 to avoid division by zero

# Plot characteristic curves for each sensor
plt.figure(figsize=(12, 8))

for _, row in data.iterrows():
    sensor_id = int(row['Sensor'])
    k = row['k']
    alpha = row['alpha']
    resistance = k * (force_range ** alpha)
    plt.plot(force_range, resistance, label=f"Sensor {sensor_id}")

# Customize the plot
plt.title("Sensor Characteristic Curves (R = k * F^alpha)", fontsize=16)
plt.xlabel("Force (N)", fontsize=14)
plt.ylabel("Resistance (R)", fontsize=14)
plt.legend(title="Sensors", fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
