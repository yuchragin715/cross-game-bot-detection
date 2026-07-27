import numpy as np
import pandas as pd

from src.config import WINDOW_MS, WINDOW_MIN_EVENTS

# In-domain (per-game). mean_dt / idle_ratio are logger-sensitive — OK here, not for SI.
feature_cols = [
    "total_movement",
    "avg_speed",
    "n_events",
    "mean_dt",
    "idle_ratio",
    "avg_turn_angle",
    "speed_std",
    "speed_max",
    "dist_std",
    "efficiency_sub",
    "turn_angle_std",
    "vh_ratio",
    "dir_change_rate",
    "xy_corr",
]

# Cross-game raw diagnostic control (unit-bound / mechanism leaks by design).
# mean_dt removed: pure sampler fingerprint, steals the "scale fails" story.
cross_game_feature_cols = [
    "avg_speed",
    "idle_ratio",
    "avg_turn_angle",
    "speed_std",
    "speed_max",
    "dist_std",
]

# Minimal SI — formal cross-game main result (parsimony).
SCALE_INVARIANT_COLS = [
    "speed_cv",
    "speed_peak",
    "turn_angle_mean",
    "accel_cv",
]

# Extended SI ablation (= main 4 + 6). Not for primary tables.
SCALE_INVARIANT_EXT_COLS = SCALE_INVARIANT_COLS + [
    "turn_angle_std",
    "efficiency_sub",
    "vh_ratio",
    "dir_change_rate",
    "xy_corr",
    "step_autocorr",
]

_EPS = 1e-9
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
    path = float(np.sqrt(sdx ** 2 + sdy ** 2).sum())
    if path <= _EPS:
        return None
    return float(np.hypot(sdx.sum(), sdy.sum()) / path)


def _efficiency_sub_mean(dx, dy, segment_events=EFFICIENCY_SUB_EVENTS):
    """Mean straightness over non-overlapping N-event chunks; drop incomplete tail.

    If the trace is shorter than N (e.g. short 10s windows with min_events < N),
    fall back to one whole-trace efficiency so the feature is still defined.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    n = int(dx.size)
    seg = int(segment_events)
    if n < 2 or seg < 2:
        return float("nan")
    if n < seg:
        eff = _segment_efficiency(dx, dy)
        return float("nan") if eff is None else eff
    vals = []
    for start in range(0, n - seg + 1, seg):
        eff = _segment_efficiency(dx[start : start + seg], dy[start : start + seg])
        if eff is not None:
            vals.append(eff)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _dir_change_rate(dx, n_events):
    """Sign flips of non-zero dx, divided by total event count (RE #87 style)."""
    sx = np.sign(np.asarray(dx, dtype=float))
    sx = sx[sx != 0]
    if sx.size < 2 or n_events <= 0:
        return 0.0
    flips = int(np.sum(sx[1:] != sx[:-1]))
    return float(flips / float(n_events))


def to_scale_invariant(feat_df):
    """Build SI columns from extract_features rows (min + ext + optional dist_cv)."""
    out = pd.DataFrame(index=feat_df.index)
    out["speed_cv"] = feat_df["speed_std"] / (feat_df["avg_speed"] + _EPS)
    out["speed_peak"] = feat_df["speed_max"] / (feat_df["avg_speed"] + _EPS)
    out["turn_angle_mean"] = feat_df["avg_turn_angle"]
    out["accel_cv"] = feat_df["accel_cv"]

    # Kept for ablation / revert checks (not in SCALE_INVARIANT_COLS).
    mean_dist = feat_df["total_movement"] / (feat_df["n_events"] + _EPS)
    out["dist_cv"] = feat_df["dist_std"] / (mean_dist + _EPS)

    # Extended SI — already dimensionless in extract_features.
    for col in (
        "turn_angle_std",
        "efficiency_sub",
        "vh_ratio",
        "dir_change_rate",
        "xy_corr",
        "step_autocorr",
    ):
        out[col] = feat_df[col]
    return out


def extract_features(mouse_df):
    df = mouse_df.sort_values("time")
    if len(df) < 2:
        return None

    dx = df["dx"].to_numpy(dtype=float)
    dy = df["dy"].to_numpy(dtype=float)
    distance = np.sqrt(dx ** 2 + dy ** 2)
    dt = df["time"].diff()
    valid = dt > 0
    speed = distance[valid] / dt[valid]
    raw_turn = np.arctan2(dy, dx)
    # wrap consecutive heading deltas into (-pi, pi]
    turn = np.diff(raw_turn)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    # align with valid speed rows: turn[i] is between event i and i+1 → index i+1
    valid_np = valid.to_numpy()
    # angles at events where dt>0 correspond to steps into those events (indices 1..)
    step_idx = np.flatnonzero(valid_np)
    # turn between (i-1, i) lives at turn[i-1]; keep turns whose end event has dt>0
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
        "avg_turn_angle": float(angle_vals.mean()) if angle_vals.size else float("nan"),
        "speed_std": float(speed.std()),
        "speed_max": float(speed.max()),
        "dist_std": float(distance.std()),
        "efficiency_sub": _efficiency_sub_mean(dx, dy),
        "turn_angle_std": float(np.std(angle_vals, ddof=1)) if angle_vals.size >= 2 else float("nan"),
        "vh_ratio": vh_ratio,
        "dir_change_rate": _dir_change_rate(dx, n_events),
        "xy_corr": _safe_pearson(dx, dy),
        # SI helpers (not in feature_cols)
        "accel_cv": accel_cv,
        "step_autocorr": 0.5
        * (
            _safe_pearson(dx[:-1], dx[1:])
            + _safe_pearson(dy[:-1], dy[1:])
        ),
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
