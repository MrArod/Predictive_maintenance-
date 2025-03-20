import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

# --- Load Data Section --- #
train_file_path = "train_FD001.txt"  # Update if needed

# Define column names (5 metadata + 21 sensors)
column_names = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + \
               [f"sensor_{i}" for i in range(1, 22)]

# Read dataset using the engine python to utilize the space delimiter \s+
# The dataset has no header, so we use None to avoid using the first row as column names
train_df = pd.read_csv(train_file_path, sep="\s+", header=None, names=column_names, engine="python")

# Convert time_in_cycles to integer to avoid widening conversion
train_df["time_in_cycles"] = train_df["time_in_cycles"].astype(int)

# Select only sensor columns for z-score calculation
sensor_columns = [col for col in train_df.columns if "sensor" in col]

# --- Preform Z Scores calculation for Anomaly Flags --- #
z_scores = train_df[sensor_columns].apply(zscore)

# Set the anomaly threshold low because of the low variability in the data
threshold = 3
# Function to flag anomalies based on z scores over a the threshold
anomaly_flags = (z_scores.abs() > threshold)

# Utilize a list comprehension to iterate over the sensor columns
# and add the anomaly flags to the df if anomaly is detected
# The anomaly_flags defined earlier contains boolean values indicating anomalies
train_df[[f"Anomaly_{col}" for col in sensor_columns]] = anomaly_flags

# Save flagged dataset
# index set to False to avoid saving a line number column
train_df.to_csv("flagged_FD001_data.csv", index=False)

# --- Preforming Anomaly Statistics per Unit --- #
# Count the number of cycles with anomalies per engine
# Create a new df with the grouped unit number then sums across the columns boolean values from the anomaly flags df
anomaly_counts_per_unit = anomaly_flags.groupby(train_df["unit_number"]).sum().sum(axis=1)

# Get total cycles per engine
total_cycles_per_unit = train_df.groupby("unit_number")["time_in_cycles"].max()

# Compute anomaly percentage per engine
anomaly_percentage = (anomaly_counts_per_unit / total_cycles_per_unit) * 100

# Sort for better visualization
anomaly_percentage = anomaly_percentage.sort_values(ascending=False)

# --- Bar Plot: Anomaly Percentage Per Engine --- #
plt.figure(figsize=(12, 6))
plt.bar(anomaly_percentage.index, anomaly_percentage, color='red')
plt.xlabel("Engine Unit Number")
plt.ylabel("Percentage of Cycles with Anomalies (%)")
plt.title("Anomalies Present Per Engine Unit")
plt.xticks(rotation=45)

# Save the plot file
plt.savefig("anomaly_percentage_per_unit.png")
plt.show()

# --- Print Summary for Presentation --- #
print("Predictive Maintenance Summary")
print(f"Total Engine Units Analyzed: {train_df['unit_number'].nunique()}")
print(f"Total Sensor Columns: {len(sensor_columns)}")
print(f"Z Score Threshold for Anomalies: {threshold}")
print("Top 5 Engines with the Highest Anomaly Percentage:")
print(anomaly_percentage.head(5))