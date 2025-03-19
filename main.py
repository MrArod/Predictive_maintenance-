import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

# --- --- --- Load Data Section --- --- --- #
# Load in the sensor training data - FD001 
train_file_path = "/workspaces/Predictive_maintenance-/train_FD001.txt"

# Read in the first training data set
# Removing header to avoid first row of data as auto headder
# Using the \\s+ delimiter to account for space " " typos 
# Setting the engine parameter to python for the \s+ delimiter
# Dropping any columns that empty
temp_df = pd.read_csv(train_file_path, sep = "\s+", header = None, engine = "python").dropna(axis = 1, how = "all")
num_columns = temp_df.shape[1]

# Enter the first 5 columns 
# Using a list comprehension for the sensor column names iterating over the f string for the range of columns
# Using the range function to account for the non sensor columns in the dataset (first 5) and indexing from 1 to avoid "sensor 0"
column_names = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"]
column_names += [f"sensor_{i}" for i in range(1, num_columns - 5 + 1)] # add 1 to account for the indexing starting at 1

# reread in the training data with the new column names
train_df = pd.read_csv(train_file_path, sep="\s+", header=None, names=column_names, engine="python").dropna(axis=1, how="all")

# Conversion of "time_in_cycles" column to an int to avoid widening conversions
train_df["time_in_cycles"] = train_df["time_in_cycles"].astype(int)


# --- --- --- Calculating Z Scores Section --- --- --- #
# Using a list comprehension to iterate over the columns and find the sensor columns then assigning to "sensor_columns"
sensor_columns = [col for col in train_df.columns if "sensor" in col]

# Create a function to calculate the z-scores by taking in a dataframe and columns as the arguments
def calculate_z_scores(df, columns):
    return df[columns].apply(zscore)

# Function to calculate the z scores for the sensor columns and assign to "z_scores" variable
# Note: the argument for the function is now specified as the sensor columns only not all columns
z_scores = calculate_z_scores(train_df, sensor_columns)


# --- --- --- Flagging Anomalies section --- --- --- #
# Function to flag anomalies by taking in the z-scores and threshold as arguments
threshold = 3  # Z score 
def flag_anomalies(z_scores, threshold=3):
    return (z_scores.abs() > threshold)

# Call the flag anomalies function against the z scores and assign to "anomaly_flags" variable
anomaly_flags = flag_anomalies(z_scores)
# Iterate over the sensor columns and add the anomaly flags to the dataframe 
# The applied anomalies function returns a boolean value which is then added to the dataframe as a new column
train_df[[f"Anomaly_{col}" for col in sensor_columns]] = anomaly_flags


# --- --- --- Save Flagged Data to CSV section --- --- --- #
# Save the flagged data to a CSV file
train_df.to_csv("flagged_FD001_data.csv", index=False)


# --- --- --- Display Sample Anomalies section --- --- --- #
# Preview the first 10 rows of the anomalies
# Using the "any" method to check if any of the columns have been added to anomaly flag
anomalies = train_df[anomaly_flags.any(axis=1)]
print("Sensor Anomalies in FD001:")
print(anomalies[["time_in_cycles"] + sensor_columns[:3] + [f"Anomaly_{sensor_columns[0]}"]].head(10))

# --- --- --- Scatter Plots for All Sensors section --- --- --- #
available_units = train_df["unit_number"].unique()
unit_id = available_units[0] if len(available_units) > 0 else None

if unit_id is not None:
    unit_df = train_df[train_df["unit_number"] == unit_id].sort_values("time_in_cycles")
    
    if not unit_df.empty:
        for sensor in sensor_columns:
            plt.figure(figsize=(12, 6))
            plt.scatter(unit_df["time_in_cycles"], unit_df[sensor], label="Sensor Data", color='blue', alpha=0.5)
            anomaly_points = unit_df[unit_df[f"Anomaly_{sensor}"]]
            plt.scatter(anomaly_points["time_in_cycles"], anomaly_points[sensor], color='red', label="Anomalies", marker='x', s=80)
            
            plt.xlabel("Time in Cycles")
            plt.ylabel(sensor)
            plt.title(f"Scatter Plot of Sensor Data - {sensor} (Anomalies Highlighted) for Engine Unit {unit_id}")
            plt.legend()
            
            # Save the figure
            plot_filename = f"sensor_{sensor}_anomalies_scatter.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"Plot saved as {plot_filename}. You can open it to view the results.")
    else:
        print("No data available for the selected engine unit.")
else:
    print("No valid engine units found in the dataset.")
