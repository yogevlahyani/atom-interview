import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import haversine_distance
import math

def analyze_pattern(df: pd.DataFrame):
    """
    Analyze and classify the movement pattern in the GPS data.

    Returns:
        classification: str - The detected pattern type
        metrics: dict - Metrics used for classification
        description: str - Detailed description of the pattern
    """
    # Convert time to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df['time'] = pd.to_datetime(df['time'])

    df = df.sort_values('time').reset_index(drop=True)

    # Calculate movement metrics
    metrics = {}

    # 1. Speed variation analysis
    if 'speed' not in df.columns:
        df['distance'] = haversine_distance(
            df['lat'].shift(1),
            df['lon'].shift(1),
            df['lat'],
            df['lon']
        )
        df['time_delta'] = df['time'].diff().dt.total_seconds()
        df['speed'] = df['distance'] / df['time_delta']
        df = df.dropna(subset=['speed'])

    metrics['avg_speed'] = df['speed'].mean()
    metrics['std_speed'] = df['speed'].std()
    metrics['cv_speed'] = metrics['std_speed'] / metrics['avg_speed'] if metrics['avg_speed'] > 0 else 0

    # 2. Directional analysis - calculate heading changes
    lat0 = df['lat'].mean()
    lon0 = df['lon'].mean()
    R = 6371000.0

    x = (np.pi / 180.0) * (df['lon'].values - lon0) * R * np.cos((np.pi / 180.0) * lat0)
    y = (np.pi / 180.0) * (df['lat'].values - lat0) * R

    headings = []
    turn_angles = []

    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        dist = math.hypot(dx, dy)

        if dist > 0.5:  # Only calculate heading for significant movements
            heading = math.atan2(dy, dx)
            headings.append(heading)

            if len(headings) > 1:
                dh = headings[-1] - headings[-2]
                # Normalize to [-pi, pi]
                dh = (dh + math.pi) % (2.0 * math.pi) - math.pi
                turn_angles.append(abs(dh))

    metrics['avg_turn_angle'] = np.mean(turn_angles) if turn_angles else 0
    metrics['std_turn_angle'] = np.std(turn_angles) if turn_angles else 0
    metrics['max_turn_angle'] = max(turn_angles) if turn_angles else 0
    metrics['num_significant_turns'] = sum(1 for t in turn_angles if t > np.pi/4)  # > 45 degrees

    # 3. Movement linearity
    start_to_end_dist = haversine_distance(
        df.iloc[0]['lat'], df.iloc[0]['lon'],
        df.iloc[-1]['lat'], df.iloc[-1]['lon']
    )
    total_path_length = df['distance'].sum()
    metrics['linearity'] = start_to_end_dist / total_path_length if total_path_length > 0 else 0

    # 4. Stop detection (speed < 1 m/s)
    stop_threshold = 1.0  # m/s
    stops = df['speed'] < stop_threshold
    metrics['stop_percentage'] = (stops.sum() / len(df)) * 100

    # Count stop-and-go events
    stop_go_transitions = 0
    for i in range(1, len(stops)):
        if stops.iloc[i] != stops.iloc[i-1]:
            stop_go_transitions += 1
    metrics['stop_go_transitions'] = stop_go_transitions

    # 5. Spatial coverage (bounding box ratio)
    lat_range = df['lat'].max() - df['lat'].min()
    lon_range = df['lon'].max() - df['lon'].min()
    metrics['lat_range'] = lat_range
    metrics['lon_range'] = lon_range
    metrics['aspect_ratio'] = lat_range / lon_range if lon_range > 0 else 0

    # Classification logic
    classification, description = classify_movement(metrics, df)

    return classification, metrics, description

def classify_movement(metrics, df):
    """
    Classify the movement pattern based on metrics.

    Pattern types:
    - straight line: High linearity, low turn angles
    - zigzag: Multiple direction changes, moderate linearity
    - loop: Low linearity, returns near start
    - stop-and-go: High speed variation, many stops
    - random walk: Low linearity, no clear pattern
    """

    # Decision tree for classification
    if metrics['stop_percentage'] > 30 and metrics['stop_go_transitions'] > 10:
        classification = "stop-and-go"
        description = (
            f"The vehicle exhibits stop-and-go behavior with {metrics['stop_percentage']:.1f}% "
            f"of the time spent at very low speeds and {metrics['stop_go_transitions']} transitions "
            f"between moving and stopped states. This pattern is typical of urban driving with "
            f"traffic lights, intersections, or congestion."
        )

    elif metrics['linearity'] > 0.7 and metrics['avg_turn_angle'] < 0.3:
        classification = "straight line"
        description = (
            f"The vehicle follows a relatively straight path with linearity of {metrics['linearity']:.2f} "
            f"(1.0 = perfectly straight). The average turn angle is {np.degrees(metrics['avg_turn_angle']):.1f}° "
            f"with only {metrics['num_significant_turns']} significant turns (>45°). This suggests "
            f"highway or arterial road driving."
        )

    elif metrics['num_significant_turns'] > 5 and metrics['linearity'] < 0.5:
        # Check if it returns to start (loop)
        start_lat, start_lon = df.iloc[0]['lat'], df.iloc[0]['lon']
        end_lat, end_lon = df.iloc[-1]['lat'], df.iloc[-1]['lon']
        end_to_start = haversine_distance(end_lat, end_lon, start_lat, start_lon)

        if end_to_start < 50:  # Within 50 meters of start
            classification = "loop"
            description = (
                f"The vehicle follows a looping pattern, with {metrics['num_significant_turns']} "
                f"significant turns and ending {end_to_start:.1f}m from the starting position. "
                f"The path has low linearity ({metrics['linearity']:.2f}) and returns near its "
                f"origin, suggesting a circular route or return trip."
            )
        else:
            classification = "zigzag"
            description = (
                f"The vehicle exhibits a zigzag pattern with {metrics['num_significant_turns']} "
                f"significant turns (>45°) and linearity of {metrics['linearity']:.2f}. "
                f"The path involves multiple direction changes, suggesting navigation through "
                f"city streets, turns at intersections, or route corrections."
            )

    elif metrics['cv_speed'] > 0.5 and metrics['std_turn_angle'] > 0.5:
        classification = "random walk"
        description = (
            f"The vehicle follows an irregular pattern with high speed variation "
            f"(coefficient of variation: {metrics['cv_speed']:.2f}) and inconsistent "
            f"turning behavior (std turn angle: {np.degrees(metrics['std_turn_angle']):.1f}°). "
            f"This suggests exploratory movement, searching behavior, or navigation in "
            f"unfamiliar areas."
        )

    else:
        classification = "mixed movement"
        description = (
            f"The vehicle exhibits mixed movement patterns that don't clearly fit a single "
            f"category. Linearity: {metrics['linearity']:.2f}, {metrics['num_significant_turns']} "
            f"significant turns, {metrics['stop_percentage']:.1f}% time stopped. This could be "
            f"typical urban navigation with a combination of straight segments and turns."
        )

    return classification, description

def plot_pattern(df, classification, metrics):
    """
    Create visualizations to show the movement pattern.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Trajectory plot colored by speed
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['lon'], df['lat'],
                         c=df['speed'],
                         cmap='RdYlGn_r',
                         s=30,
                         alpha=0.7)
    ax1.plot(df['lon'], df['lat'], 'k-', alpha=0.3, linewidth=1)
    ax1.scatter(df.iloc[0]['lon'], df.iloc[0]['lat'],
               c='green', s=200, marker='o',
               edgecolors='black', linewidths=2,
               label='Start', zorder=5)
    ax1.scatter(df.iloc[-1]['lon'], df.iloc[-1]['lat'],
               c='red', s=200, marker='s',
               edgecolors='black', linewidths=2,
               label='End', zorder=5)
    plt.colorbar(scatter, ax=ax1, label='Speed (m/s)')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'GPS Trajectory - Pattern: {classification.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Speed over time
    ax2 = axes[0, 1]
    time_minutes = (df['time'] - df['time'].min()).dt.total_seconds() / 60
    ax2.plot(time_minutes, df['speed'], 'b-', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Stop threshold (1 m/s)')
    ax2.axhline(y=metrics['avg_speed'], color='g', linestyle='--', alpha=0.5,
                label=f'Average: {metrics["avg_speed"]:.2f} m/s')
    ax2.fill_between(time_minutes, 0, df['speed'], alpha=0.3)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_title('Speed Profile Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Turn angle distribution
    ax3 = axes[1, 0]
    lat0 = df['lat'].mean()
    lon0 = df['lon'].mean()
    R = 6371000.0
    x = (np.pi / 180.0) * (df['lon'].values - lon0) * R * np.cos((np.pi / 180.0) * lat0)
    y = (np.pi / 180.0) * (df['lat'].values - lat0) * R

    turn_angles = []
    for i in range(2, len(x)):
        dx1, dy1 = x[i-1] - x[i-2], y[i-1] - y[i-2]
        dx2, dy2 = x[i] - x[i-1], y[i] - y[i-1]
        if math.hypot(dx1, dy1) > 0.5 and math.hypot(dx2, dy2) > 0.5:
            h1 = math.atan2(dy1, dx1)
            h2 = math.atan2(dy2, dx2)
            dh = (h2 - h1 + math.pi) % (2.0 * math.pi) - math.pi
            turn_angles.append(np.degrees(abs(dh)))

    if turn_angles:
        ax3.hist(turn_angles, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(x=45, color='r', linestyle='--', alpha=0.5, label='Significant turn (45°)')
        ax3.set_xlabel('Absolute Turn Angle (degrees)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Turn Angle Distribution (n={metrics["num_significant_turns"]} significant turns)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    metrics_text = f"""
Pattern Classification: {classification.upper()}

Movement Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Speed Statistics:
  • Average: {metrics['avg_speed']:.2f} m/s ({metrics['avg_speed']*3.6:.2f} km/h)
  • Std Dev: {metrics['std_speed']:.2f} m/s
  • Coefficient of Variation: {metrics['cv_speed']:.2f}

Directional Characteristics:
  • Linearity: {metrics['linearity']:.2f} (1.0 = straight)
  • Significant turns (>45°): {metrics['num_significant_turns']}
  • Avg turn angle: {np.degrees(metrics['avg_turn_angle']):.1f}°
  • Max turn angle: {np.degrees(metrics['max_turn_angle']):.1f}°

Stop-and-Go Behavior:
  • Time stopped (<1 m/s): {metrics['stop_percentage']:.1f}%
  • Stop/go transitions: {metrics['stop_go_transitions']}

Spatial Coverage:
  • Lat range: {metrics['lat_range']*111000:.1f} m
  • Lon range: {metrics['lon_range']*111000*np.cos(np.radians(df['lat'].mean())):.1f} m
    """

    ax4.text(0.1, 0.9, metrics_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('pattern_analysis.png', dpi=150, bbox_inches='tight')
    print("Pattern visualization saved to pattern_analysis.png")
    plt.show()

if __name__ == "__main__":
    # Load the best window from Part 2 (120 seconds)
    df = pd.read_json("best_window_120s.jsonl", lines=True)

    print("=" * 70)
    print("PATTERN RECOGNITION ANALYSIS - 120 Second Window")
    print("=" * 70)

    # Analyze pattern
    classification, metrics, description = analyze_pattern(df)

    # Print results
    print(f"\nPattern Classification: {classification.upper()}\n")
    print("Description:")
    print(f"{description}\n")

    print("\nDetailed Metrics:")
    print("-" * 70)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:10.4f}")
        else:
            print(f"  {key:25s}: {value:10}")

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_pattern(df, classification, metrics)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
