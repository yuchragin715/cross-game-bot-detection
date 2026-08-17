import numpy as np
import pandas as pd

from src.config import WINDOW_MS, WINDOW_MIN_EVENTS

# In-domain (per-game)
feature_cols = [
    "total_movement",
    "avg_speed",
    "n_events",
    "mean_dt",
    "idle_ratio",
    "turn_angle_mean",
    "speed_std",
    "speed_max",
    "dist_std",
    "efficiency_sub",
    "turn_angle_std",
    "vh_ratio",
    "dir_change_rate",
    "xy_corr",
]

# Cross-game raw features
cross_game_feature_cols = [
    "avg_speed",
    "idle_ratio",
    "turn_angle_mean",
    "speed_std",
    "speed_max",
    "dist_std",
]

# Minimal scale invariant features for cross-game main result
SCALE_INVARIANT_COLS = [
    "speed_cv",
    "speed_peak",
    "turn_angle_mean",
    "accel_cv",
]

# Extended scale invariant features (= main 4 + 6)
SCALE_INVARIANT_EXT_COLS = SCALE_INVARIANT_COLS + [
    "turn_angle_std",
    "efficiency_sub",
    "vh_ratio",
    "dir_change_rate",
    "xy_corr",
    "step_autocorr",
]

_EPS = 1e-9 # 0.000000001
EFFICIENCY_SUB_EVENTS = 40  # fixed event count per sub-segment (not fixed time)


def _safe_pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2:
        return 0.0
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(r):
        return 0.0
    return float(r)


def _segment_efficiency(sdx, sdy):
    path = float(np.hypot(sdx, sdy).sum())
    if path <= _EPS:
        return None
    return float(np.hypot(sdx.sum(), sdy.sum()) / path)


def _efficiency_sub_mean(dx, dy, segment_events=EFFICIENCY_SUB_EVENTS):
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    event_count = int(dx.size)
    seg = int(segment_events)
    if event_count < 2 or seg < 2:
        return float("nan")
    if event_count < seg:
        efficiency = _segment_efficiency(dx, dy)
        return float("nan") if efficiency is None else efficiency
    efficiencies = []
    for start in range(0, event_count - seg + 1, seg):
        efficiency = _segment_efficiency(dx[start : start + seg], dy[start : start + seg])
        if efficiency is not None:
            efficiencies.append(efficiency)
    if not efficiencies:
        return float("nan")
    return float(np.mean(efficiencies))


def _dir_change_rate(dx, n_events):
    sx = np.sign(np.asarray(dx, dtype=float))
    sx = sx[sx != 0]
    if sx.size < 2 or n_events <= 0:
        return 0.0
    flips = int(np.sum(sx[1:] != sx[:-1]))
    return float(flips / float(n_events))


# calculate scale invariant features
def to_scale_invariant(feat_df):
    result = pd.DataFrame(index=feat_df.index)
    result["speed_cv"] = feat_df["speed_std"] / (feat_df["avg_speed"] + _EPS)
    result["speed_peak"] = feat_df["speed_max"] / (feat_df["avg_speed"] + _EPS)
    result["turn_angle_mean"] = feat_df["turn_angle_mean"]
    result["accel_cv"] = feat_df["accel_cv"]

    # calculate dist_cv
    mean_dist = feat_df["total_movement"] / (feat_df["n_events"] + _EPS)
    result["dist_cv"] = feat_df["dist_std"] / (mean_dist + _EPS)

    # add extended scale invariant features
    for col in (
        "turn_angle_std",
        "efficiency_sub",
        "vh_ratio",
        "dir_change_rate",
        "xy_corr",
        "step_autocorr",
    ):
        result[col] = feat_df[col]
    return result


def extract_features(mouse_df):
    df = mouse_df.sort_values("time")
    if len(df) < 2:
        return None

    dx = df["dx"].to_numpy(dtype=float)
    dy = df["dy"].to_numpy(dtype=float)
    distance = np.hypot(dx, dy)
    dt = df["time"].diff()
    valid = dt > 0
    speed = distance[valid] / dt[valid]
    raw_turn = np.arctan2(dy, dx)

    # wrap consecutive heading deltas into (-pi, pi], exclude minus pi and include pi
    turn = np.diff(raw_turn)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    valid_np = valid.to_numpy()
    step_idx = np.flatnonzero(valid_np)
    angle_vals = np.abs(turn[step_idx - 1]) if step_idx.size else np.array([])

    if speed.empty:
        return None

    speed_vals = speed.to_numpy(dtype=float)
    dt_vals = dt[valid].to_numpy(dtype=float)
    if speed_vals.size >= 2:
        accel = np.diff(speed_vals) / (dt_vals[1:] + _EPS)
        accel_cv = float(np.std(accel) / (np.mean(np.abs(accel)) + _EPS))
    else:
        accel_cv = float("nan")

    n_events = int(len(df))
    sum_abs_dx = float(np.abs(dx).sum())
    vh_ratio = float(np.abs(dy).sum() / sum_abs_dx) if sum_abs_dx > _EPS else 0.0

    return {
        "total_movement": float(distance.sum()),
        "avg_speed": float(speed.mean()),
        "n_events": n_events,
        "mean_dt": float(dt[valid].mean()),
        "idle_ratio": float((distance < 1).mean()),
        "turn_angle_mean": float(angle_vals.mean()) if angle_vals.size else float("nan"),
        "speed_std": float(speed.std()),
        "speed_max": float(speed.max()),
        "dist_std": float(distance.std()),
        "efficiency_sub": _efficiency_sub_mean(dx, dy),
        "turn_angle_std": float(np.std(angle_vals, ddof=1)) if angle_vals.size >= 2 else float("nan"),
        "vh_ratio": vh_ratio,
        "dir_change_rate": _dir_change_rate(dx, n_events),
        "xy_corr": _safe_pearson(dx, dy),
        # scale invariant features (not in feature_cols)
        "accel_cv": accel_cv,
        "step_autocorr": 0.5
        * (
            _safe_pearson(dx[:-1], dx[1:])
            + _safe_pearson(dy[:-1], dy[1:])
        ),
    }


def segment_trace(mouse_df, window_ms=WINDOW_MS, min_events=WINDOW_MIN_EVENTS):
    if mouse_df is None or len(mouse_df) < 2:
        return []

    df = mouse_df.sort_values("time").reset_index(drop=True)
    t_start = float(df["time"].iloc[0])
    t_end = float(df["time"].iloc[-1])
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        return []

    windows = []
    start = t_start
    window_ms = float(window_ms)

    # iterate over the trace and cut it into windows
    while start < t_end:
        end = start + window_ms
        window = df[(df["time"] >= start) & (df["time"] < end)].copy()
        if len(window) >= int(min_events):
            window["time"] = window["time"] - window["time"].iloc[0]
            windows.append(window.reset_index(drop=True))
        start = end
    return windows


def build_window_feature_table(
    mouse_dfs,
    groups,
    session_ids,
    is_bot,
    bot_type=None,
):
    mouse_dfs = list(mouse_dfs)
    groups = list(np.asarray(groups))
    session_ids = list(session_ids)
    if not (len(mouse_dfs) == len(groups) == len(session_ids)):
        raise ValueError(
            "build_window_feature_table: mouse_dfs / groups / session_ids length mismatch"
        )

    # extract features for each window
    rows = []
    for mouse, group_id, session_id in zip(mouse_dfs, groups, session_ids):
        for window_idx, window in enumerate(segment_trace(mouse)):
            feats = extract_features(window)
            if feats is None:
                continue
            feats["group"] = group_id
            feats["session_id"] = session_id
            feats["window_idx"] = window_idx
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
