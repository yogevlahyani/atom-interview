import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
from collections import Counter
from typing import Tuple, Dict, Any

# Haversine formula to calculate distance between two GPS coordinates
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance in meters between two lat/lon points
    Using Haversine formula
    """
    R = 6371000  # Earth radius in meters

    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c

    return distance

def create_plot_trajectory(df):
    """
    Create a plot of the GPS trajectory
    """
    # Plot trajectory colored by speed
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['lon'], df['lat'],
                        c=df['speed'],
                        cmap='RdYlGn_r',  # Green (slow) to Red (fast)
                        s=10,
                        alpha=0.7)
    plt.colorbar(scatter, label='Speed (m/s)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Trajectory Colored by Speed')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def extract_max_value_window(df: pd.DataFrame, duration_seconds: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Find the S-second window most valuable for building a street map.

    Inputs:
      df: pandas DataFrame with at least:
          - 'time' (string like "YYYY-MM-DD HH:MM:SS.ffffff" or datetime-like)
          - 'lat', 'lon' (floats)
          Optional:
          - 'h_acc_m' (horizontal accuracy in meters; smaller is better)

    Returns:
      best_window: DataFrame slice (copy) containing only rows in the best window
      metadata: dict describing the metric, window boundaries, and breakdown

    Metric (single scalar):
      "Uncertainty-weighted map information gain" =
        avg_quality * (unique_cells + 0.02*path_length_m + 3.0*turn_score_pi)

      - unique_cells: grid coverage novelty (new ground covered)
      - path_length_m: small bonus for sustained movement
      - turn_score_pi: geometry richness proxy (turns/intersections)
      - avg_quality: downweights noisy GNSS (1 / (1 + h_acc_m))
    """
    if df is None or len(df) == 0:
        raise ValueError("df is empty")

    # ---- parameters (tunable, kept inside for easy reuse) ----
    cell_size_m = 3.0            # grid resolution for "novel coverage"
    min_speed_mps = 0.8          # ignore turn noise when nearly stationary
    distance_weight = 0.02
    turn_weight = 3.0

    # ---- normalize & sort time ----
    d = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(d["time"]):
        d["time"] = pd.to_datetime(d["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    d = d.dropna(subset=["time", "lat", "lon"]).sort_values("time").reset_index(drop=True)

    if len(d) < 2:
        raise ValueError("Need at least 2 rows with valid time/lat/lon")

    t0 = d.loc[0, "time"]
    tsec = (d["time"] - t0).dt.total_seconds().to_numpy()

    # ---- local projection lat/lon -> x/y meters (equirectangular) ----
    lat0 = float(d["lat"].mean())
    lon0 = float(d["lon"].mean())
    R = 6371000.0

    lat = d["lat"].to_numpy(dtype=float)
    lon = d["lon"].to_numpy(dtype=float)

    x = (math.pi / 180.0) * (lon - lon0) * R * math.cos((math.pi / 180.0) * lat0)
    y = (math.pi / 180.0) * (lat - lat0) * R

    # ---- quality weight (0,1], higher is better ----
    if "h_acc_m" in d.columns:
        h_acc = pd.to_numeric(d["h_acc_m"], errors="coerce").fillna(5.0).to_numpy(dtype=float)
    else:
        h_acc = (pd.Series([5.0] * len(d))).to_numpy(dtype=float)

    q = 1.0 / (1.0 + h_acc)

    # ---- grid cells for novelty ----
    cx = (x / cell_size_m).astype(int)
    cy = (y / cell_size_m).astype(int)
    cells = list(zip(cx, cy))

    # ---- per-step distance, speed, heading, and turning magnitude ----
    n = len(d)
    dist = [0.0] * n
    speed = [0.0] * n
    heading = [0.0] * n
    turn_mag = [0.0] * n

    for i in range(1, n):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        di = math.hypot(dx, dy)
        dist[i] = di

        dti = tsec[i] - tsec[i - 1]
        if dti > 0:
            speed[i] = di / dti

        heading[i] = math.atan2(dy, dx) if di > 0 else heading[i - 1]

        dh = heading[i] - heading[i - 1]
        dh = (dh + math.pi) % (2.0 * math.pi) - math.pi  # normalize to [-pi, pi]
        turn_mag[i] = abs(dh)

    # ---- sliding window maximization ----
    counts = Counter()
    sum_dist = 0.0
    sum_turn = 0.0
    sum_q = 0.0

    best_value = None
    best_i = 0
    best_j = 0
    best_breakdown = {}

    i = 0
    j = 0
    while i < n:
        while j < n and (tsec[j] - tsec[i]) <= duration_seconds:
            counts[cells[j]] += 1
            sum_dist += dist[j]
            if speed[j] >= min_speed_mps:
                sum_turn += turn_mag[j]
            sum_q += q[j]
            j += 1

        window_len = j - i
        if window_len >= 2:
            unique_cells = len(counts)
            avg_q = sum_q / window_len
            turn_score_pi = sum_turn / math.pi

            value = avg_q * (unique_cells + distance_weight * sum_dist + turn_weight * turn_score_pi)

            if best_value is None or value > best_value:
                best_value = value
                best_i = i
                best_j = j  # exclusive
                best_breakdown = {
                    "unique_cells": unique_cells,
                    "path_length_m": float(sum_dist),
                    "turn_score_pi": float(turn_score_pi),
                    "avg_quality": float(avg_q),
                    "points": int(window_len),
                }

        # slide forward: remove i
        if i < j:
            c = cells[i]
            counts[c] -= 1
            if counts[c] == 0:
                del counts[c]
            sum_dist -= dist[i]
            if speed[i] >= min_speed_mps:
                sum_turn -= turn_mag[i]
            sum_q -= q[i]

        i += 1
        if i > j:
            j = i

    best_window = d.iloc[best_i:best_j].copy()

    metadata = {
        "metric_name": "Uncertainty-weighted map information gain",
        "plain_english_metric": (
            "A window is valuable if it covers lots of new ground (unique grid cells) and contains "
            "useful road shape (turns/intersections), while down-weighting windows with worse GNSS accuracy."
        ),
        "duration_seconds": int(duration_seconds),
        "window_start_time": best_window.iloc[0]["time"].isoformat(),
        "window_end_time": best_window.iloc[-1]["time"].isoformat(),
        "value": float(best_value) if best_value is not None else None,
        "breakdown": best_breakdown,
        "parameters": {
            "cell_size_m": cell_size_m,
            "min_speed_mps_for_turns": min_speed_mps,
            "distance_weight": distance_weight,
            "turn_weight": turn_weight,
        },
        "justification_2_3_sentences": (
            "Street maps improve most when you capture new coverage and geometry such as turns and intersections; "
            "repeated straight segments add little new information. This metric rewards unique spatial coverage and "
            "turning complexity (a strong proxy for intersections) and down-weights noisy GNSS so the chosen window is reliable."
        ),
    }

    return best_window, metadata