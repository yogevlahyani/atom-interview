import pandas as pd
from utils import create_plot_trajectory, extract_max_value_window, haversine_distance

# Load GPS data
df = pd.read_json("gnss_data_jabotinsky_7_8.jsonl", lines=True)

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'])

print(f"Loaded {len(df)} GPS readings")
time_duration = (df['time'].max() - df['time'].min()).total_seconds() / 60
print(f"Time duration: {time_duration:.2f} minutes")

# Calculate distances between consecutive points
df['distance'] = haversine_distance(
    df['lat'].shift(1),
    df['lon'].shift(1),
    df['lat'],
    df['lon']
)

# Calculate time deltas in seconds
df['time_delta'] = df['time'].diff().dt.total_seconds()

# Calculate speed in m/s
df['speed'] = df['distance'] / df['time_delta']

# Remove the first row (NaN values due to shift/diff)
df_clean = df.dropna(subset=['speed'])

print(f"\nSpeed statistics:")
print(f"Average speed: {df_clean['speed'].mean():.2f} m/s ({df_clean['speed'].mean() * 3.6:.2f} km/h)")
print(f"Max speed: {df_clean['speed'].max():.2f} m/s ({df_clean['speed'].max() * 3.6:.2f} km/h)")
print(f"Min speed: {df_clean['speed'].min():.2f} m/s ({df_clean['speed'].min() * 3.6:.2f} km/h)")

# Task 1: Create and show the trajectory plot
create_plot_trajectory(df_clean)

# Task 2: Extract the most valuable S-second window
best_window, metadata = extract_max_value_window(df_clean, duration_seconds=120)
print(f"\nMost valuable window metadata for 120 seconds:")
for key, value in metadata.items():
    print(f"{key}: {value}")
best_window.to_json(
    "best_window_120s.jsonl",
    orient="records",
    lines=True,
    date_format="iso"
)
    
best_window, metadata = extract_max_value_window(df_clean, duration_seconds=300)
print(f"\nMost valuable window metadata for 300 seconds:")
for key, value in metadata.items():
    print(f"{key}: {value}")
best_window.to_json(
    "best_window_300s.jsonl",
    orient="records",
    lines=True,
    date_format="iso"
)