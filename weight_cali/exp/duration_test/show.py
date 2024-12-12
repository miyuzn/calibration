import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file and parse the sheet, using the first two rows as headers
file_path = './weight_cali/exp/duration_test/exp.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=[0, 1])

# Initialize an empty dictionary to collect sensor data
sensor_data = {}

# Get the list of dates (experiments)
dates = df.columns.levels[0]

for date in dates:
    # Extract the data for that date
    date_df = df[date]
    # Drop rows where 'Sensor' is NaN
    date_df = date_df[date_df['Sensor'].notna()]
    # Iterate over the rows
    for idx, row in date_df.iterrows():
        sensor = row['Sensor']
        k = row['k']
        alpha = row['alpha']

        # Ensure sensor is an integer (or appropriate type)
        sensor = int(sensor)

        if sensor not in sensor_data:
            sensor_data[sensor] = {}

        sensor_data[sensor][date] = {'k': k, 'alpha': alpha}

# Define the range for F (Force values)
F_values = np.linspace(1, 10, 500)

# Plot characteristic curves for each sensor
num_sensors = len(sensor_data)
ncols = 2
nrows = (num_sensors + ncols - 1) // ncols
plt.figure(figsize=(15, 5 * nrows))

sensor_list = sorted(sensor_data.keys())

for i, sensor in enumerate(sensor_list):
    plt.subplot(nrows, ncols, i + 1)

    for date in sensor_data[sensor]:
        k = float(sensor_data[sensor][date]['k'])
        alpha = float(sensor_data[sensor][date]['alpha'])

        R_values = k * (F_values ** alpha)

        plt.plot(F_values, R_values, label=date)

    plt.xlabel('Force (F)')
    plt.ylabel('Resistance (R)')
    plt.title(f'Characteristic Curve for Sensor {sensor}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
