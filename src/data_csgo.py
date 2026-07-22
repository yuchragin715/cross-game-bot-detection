from pathlib import Path

import numpy as np
import pandas as pd

def check_axis_convention(eye_y, min_abs_deg=85.0):
    pitch = np.degrees(np.arcsin(np.clip(np.asarray(eye_y, dtype=float), -1.0, 1.0)))
    m = float(np.nanmax(np.abs(pitch)))
    ok = m >= min_abs_deg
    print(f"[once] max|pitch from Y|={m:.1f} deg -> {'OK' if ok else 'FAIL'}")
    return ok

def eye_vector_to_angles(x, y, z):
    pitch = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(z, x))
    return pitch, yaw

def angles_to_eye_vector(pitch_deg, yaw_deg):
    pitch, yaw = np.radians(np.asarray(pitch_deg, dtype=float)), np.radians(np.asarray(yaw_deg, dtype=float))
    cp = np.cos(pitch)
    return cp * np.cos(yaw), np.sin(pitch), cp * np.sin(yaw)

def eye_vectors_to_mouse_df(
    time_s,
    eye_x,
    eye_y,
    eye_z,
    teleport_deg=30.0,
):
    time_s = np.asarray(time_s, dtype=float)
    pitch, yaw = eye_vector_to_angles(eye_x, eye_y, eye_z)
    pitch = np.asarray(pitch, dtype=float)
    yaw = np.asarray(yaw, dtype=float)

    d_yaw = np.diff(yaw)
    d_yaw = (d_yaw + 180.0) % 360.0 - 180.0   # wrap across +/-180 seam
    d_pitch = np.diff(pitch)

    out = pd.DataFrame({"dx": d_yaw, "dy": d_pitch, "time": (time_s[1:] - time_s[0]) * 1000.0})
    keep = (out["dx"].abs() <= teleport_deg) & (out["dy"].abs() <= teleport_deg)
    n_drop = int((~keep).sum())
    out = out.loc[keep].reset_index(drop=True)


    meta = {
        "n_in": int(len(time_s)),
        "n_out": int(len(out)),
        "n_teleport_dropped": n_drop,
        "teleport_frac": n_drop / max(len(d_yaw), 1),
    }
    return out, meta

def validate_real_roundtrip(
    eye_x, eye_y, eye_z,
    yaw=None, pitch=None,
    pole_deg=89.0,
    err_threshold=1e-5,
):
    ex = np.asarray(eye_x, float)
    ey = np.asarray(eye_y, float)
    ez = np.asarray(eye_z, float)
    # eye_vector_to_angles returns (pitch, yaw)
    if yaw is None or pitch is None:
        pitch, yaw = eye_vector_to_angles(ex, ey, ez)
    pitch = np.asarray(pitch, float)
    yaw = np.asarray(yaw, float)

    xr, yr, zr = angles_to_eye_vector(pitch, yaw)
    err = np.maximum.reduce([
        np.abs(xr - ex), np.abs(yr - ey), np.abs(zr - ez)
    ])

    away = np.abs(pitch) < pole_deg
    err_away = float(err[away].max()) if away.any() else 0.0
    err_pole = float(err[~away].max()) if (~away).any() else 0.0
    ok = err_away < err_threshold

    print(
        f"[real round-trip] away from poles ({away.mean():.1%}): "
        f"max err = {err_away:.2e} "
        f"({'PASS' if ok else 'FAIL'}, threshold {err_threshold:g})"
    )
    print(
        f"[real round-trip] at poles |pitch|>={pole_deg:g} ({(~away).mean():.2%}): "
        f"max err = {err_pole:.2e} (looking straight up/down; larger error OK)"
    )
    print(f"[info] pitch range [{pitch.min():.1f}, {pitch.max():.1f}] deg")
    return {"ok": ok, "err_away": err_away, "err_pole": err_pole}


def _flt_t0_s(session_dir):
    # Only need the first time value; nrows=1 still parses header.
    t = pd.read_csv(session_dir / "gameFlt.csv", usecols=["time"], nrows=1)["time"]
    return float(t.iloc[0])


def find_round_start_s(session_dir, prefer_round=2, block_duration_s=660.0):
    mrk = pd.read_csv(session_dir / "gameMrk.csv", usecols=["time", "channel_0"])
    starts, ends = {}, {}
    for _, row in mrk.iterrows():
        label = str(row["channel_0"])
        parts = label.split()
        if len(parts) != 3 or parts[0] != "Round":
            continue
        try:
            n = int(parts[1])
        except ValueError:
            continue
        if parts[2] == "start":
            starts[n] = float(row["time"])
        elif parts[2] == "end":
            ends[n] = float(row["time"])

    for n in (prefer_round, 1):
        if n in starts:
            return starts[n], n
        if n in ends:
            return ends[n] - float(block_duration_s), n
    return None, None


def window_mouse_round_alive(
    mouse_df,
    session_dir,
    window_min=3,
    prefer_round=2,
    health_min=1.0,
):
    session_dir = Path(session_dir)
    t0 = _flt_t0_s(session_dir)
    round_start_s, round_n = find_round_start_s(session_dir, prefer_round=prefer_round)
    if round_start_s is None:
        return None, {
            "ok": False,
            "reason": "no_round_start",
            "round_n": None,
        }

    window_ms = float(window_min) * 60.0 * 1000.0
    start_ms = max(0.0, (round_start_s - t0) * 1000.0)
    end_ms = start_ms + window_ms

    abs_s = t0 + mouse_df["time"].to_numpy(dtype=float) / 1000.0
    gint = pd.read_csv(session_dir / "gameInt.csv", usecols=["time", "health"])
    g_t = gint["time"].to_numpy(dtype=float)
    g_hp = gint["health"].to_numpy(dtype=float)
    idx = np.searchsorted(g_t, abs_s, side="left")
    idx = np.clip(idx, 0, len(g_t) - 1)
    prev = np.clip(idx - 1, 0, len(g_t) - 1)
    use_prev = np.abs(g_t[prev] - abs_s) <= np.abs(g_t[idx] - abs_s)
    idx = np.where(use_prev, prev, idx)
    alive = g_hp[idx] >= health_min

    in_window = (mouse_df["time"].to_numpy(dtype=float) >= start_ms) & (
        mouse_df["time"].to_numpy(dtype=float) < end_ms
    )
    keep = in_window & alive
    out = mouse_df.loc[keep].copy()
    if len(out) < 2:
        return None, {
            "ok": False,
            "reason": "too_few_events",
            "round_n": round_n,
            "n_window": int(in_window.sum()),
            "n_alive_in_window": int(keep.sum()),
            "start_ms": start_ms,
        }

    t_base = float(out["time"].iloc[0])
    out["time"] = out["time"] - t_base
    out = out.reset_index(drop=True)
    meta = {
        "ok": True,
        "round_n": round_n,
        "round_start_s": round_start_s,
        "start_ms": start_ms,
        "n_in": int(len(mouse_df)),
        "n_window": int(in_window.sum()),
        "n_out": int(len(out)),
        "alive_frac_in_window": float(keep.sum() / max(in_window.sum(), 1)),
    }
    return out, meta