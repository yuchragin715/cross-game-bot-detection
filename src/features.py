import numpy as np

feature_cols = [
    "total_movement", "avg_speed", "n_events", "mean_dt",
    "idle_ratio", "avg_turn_angle", "speed_std", "speed_max", "dist_std",
]

cross_game_feature_cols = [
    "avg_speed", "mean_dt", "idle_ratio", "avg_turn_angle",
    "speed_std", "speed_max", "dist_std",
]

def extract_features(mouse_df):
    df = mouse_df.sort_values("time")
    if len(df) < 2:
        return None

    distance = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    dt = df["time"].diff()
    valid = dt > 0
    speed = distance[valid] / dt[valid]
    
    raw_turn = np.arctan2(df["dy"], df["dx"]).diff()
    angles = ((raw_turn + np.pi) % (2 * np.pi) - np.pi).abs()[valid]

    if speed.empty:
        return None

    return {
        "total_movement": distance.sum(),
        "avg_speed": speed.mean(),
        "n_events": len(df),
        "mean_dt": dt[valid].mean(),
        "idle_ratio": (distance < 1).mean(),
        "avg_turn_angle": angles.mean(),
        "speed_std": speed.std(),
        "speed_max": speed.max(),
        "dist_std": distance.std(),
    }
