import numpy as np
import pandas as pd

from src.config import WINDOW_MS, WINDOW_MIN_EVENTS

feature_cols = [
    "total_movement", "avg_speed", "n_events", "mean_dt",
    "idle_ratio", "avg_turn_angle", "speed_std", "speed_max", "dist_std",
]

cross_game_feature_cols = [
    "avg_speed", "mean_dt", "idle_ratio", "avg_turn_angle",
    "speed_std", "speed_max", "dist_std",
]

SCALE_INVARIANT_COLS = ["speed_cv", "speed_peak", "dist_cv", "turn_angle"]
_EPS = 1e-9


def to_scale_invariant(feat_df):
    out = pd.DataFrame(index=feat_df.index)
    out["speed_cv"] = feat_df["speed_std"] / (feat_df["avg_speed"] + _EPS)
    out["speed_peak"] = feat_df["speed_max"] / (feat_df["avg_speed"] + _EPS)
    mean_dist = feat_df["total_movement"] / (feat_df["n_events"] + _EPS)
    out["dist_cv"] = feat_df["dist_std"] / (mean_dist + _EPS)
    out["turn_angle"] = feat_df["avg_turn_angle"]
    return out


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


def segment_trace(mouse_df, window_ms=None, min_events=None):
    if window_ms is None:
        window_ms = WINDOW_MS
    if min_events is None:
        min_events = WINDOW_MIN_EVENTS

    if mouse_df is None or len(mouse_df) < 2:
        return []

    df = mouse_df.sort_values("time").reset_index(drop=True)
    t0 = float(df["time"].iloc[0])
    t_end = float(df["time"].iloc[-1])
    if not np.isfinite(t0) or not np.isfinite(t_end) or t_end <= t0:
        return []

    windows = []
    start = t0
    window_ms = float(window_ms)
    while start < t_end:
        end = start + window_ms
        w = df[(df["time"] >= start) & (df["time"] < end)].copy()
        if len(w) >= int(min_events):
            w["time"] = w["time"] - w["time"].iloc[0]
            windows.append(w.reset_index(drop=True))
        start = end
    return windows


def build_window_feature_table(
    mouse_dfs,
    groups,
    session_ids,
    is_bot,
    window_ms=None,
    min_events=None,
    bot_type=None,
):
    mouse_dfs = list(mouse_dfs)
    groups = list(np.asarray(groups))
    session_ids = list(session_ids)
    if not (len(mouse_dfs) == len(groups) == len(session_ids)):
        raise ValueError(
            "build_window_feature_table: mouse_dfs / groups / session_ids length mismatch"
        )

    rows = []
    for mouse, gid, sid in zip(mouse_dfs, groups, session_ids):
        for wi, w in enumerate(segment_trace(mouse, window_ms=window_ms, min_events=min_events)):
            feats = extract_features(w)
            if feats is None:
                continue
            feats["group"] = gid
            feats["session_id"] = sid
            feats["window_idx"] = wi
            feats["is_bot"] = int(is_bot)
            if bot_type is not None:
                feats["bot_type"] = bot_type
            rows.append(feats)

    if not rows:
        return pd.DataFrame(
            columns=list(feature_cols)
            + ["group", "session_id", "window_idx", "is_bot", "bot_type"]
        )
    return pd.DataFrame(rows)
