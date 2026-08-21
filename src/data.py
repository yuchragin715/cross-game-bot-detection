import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CSGO_DATA_ROOT,
    LOL_DATA_ROOT,
    LOL_DERIVED_ROOT,
    LOL_MATCH_MIN_EVENTS,
    LOL_MATCH_WINDOW_END_MIN,
    LOL_MATCH_WINDOW_START_MIN,
    LOL_TIMESTAMP_PARSE,
    RE_DATA_ROOT,
)

# Keylogger timestamps are wall-clock UTC+1 (dataset README).
_LOL_KEYLOGGER_TZ = timezone(timedelta(hours=1))


# load red eclipse data
def find_red_eclipse_files():
    return sorted(
        # find all json files path in the red eclipse folder
        # and filter out the users and sessions files
        path for path in Path(RE_DATA_ROOT).rglob("*.json")
        if "users" not in path.name and path.name != "sessions.json"
    )


def load_red_eclipse_mouse(file_path):
    with open(file_path) as file:
        game = json.load(file)
    meta = {
        "userId": game["userId"],
        "gameId": game["id"],
        "source_file": Path(file_path).name,
    }
    mouse = [event for event in game["events"] if event.get("type") == "MouseEvent"]
    df = pd.DataFrame(mouse)
    if not df.empty:
        df = df[["dx", "dy", "time"]]
    return meta, df


# load lol dataset shift function
def _load_lol_shift_fn():
    shift_path = Path(LOL_DATA_ROOT) / "shift.py"
    if not shift_path.is_file():
        raise FileNotFoundError(f"LoL shift.py not found: {shift_path}")
    spec = importlib.util.spec_from_file_location("lol_shift", shift_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.getShift


# parse lol timestamp to milliseconds
def _parse_lol_timestamp_ms(ts_str):
    date_part, time_part = ts_str.split("T")
    if "." in time_part:
        hms, frac = time_part.rsplit(".", 1)
        extra_ms = int(frac[:3].rjust(3, "0"))
    else:
        hms, extra_ms = time_part, 0
    base = datetime.strptime(f"{date_part}T{hms}", "%Y-%m-%dT%H:%M:%S")
    base = base.replace(tzinfo=_LOL_KEYLOGGER_TZ)
    return int(base.timestamp() * 1000) + extra_ms


# read lol moved absolute data
def _read_lol_moved_abs(file_path):
    rows = []
    with open(file_path, encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(("Subject", "File ", "Version", "Released", "Elapsed")):
                continue
            parts = line.split("\t")
            if len(parts) < 4 or parts[1].strip() != "Moved":
                continue
            try:
                rows.append({
                    "time_ms": _parse_lol_timestamp_ms(parts[0]),
                    "x": int(parts[2]),
                    "y": int(parts[3]),
                })
            except ValueError:
                continue
    if not rows:
        return pd.DataFrame(columns=["time_ms", "x", "y"])
    return pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)


def _abs_rows_to_mouse_df(abs_df):
    if abs_df is None or len(abs_df) < 2:
        return None
    df = abs_df.sort_values("time_ms").copy()
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["time"] = df["time_ms"] - df["time_ms"].iloc[0]
    out = df.iloc[1:][["dx", "dy", "time"]].reset_index(drop=True)
    return out if len(out) >= 1 else None

# list lol matches in a session directory
def list_lol_matches(session_dir, get_shift=None):
    session_dir = Path(session_dir)
    if get_shift is None:
        get_shift = _load_lol_shift_fn()
    matches = []
    for mp in sorted(session_dir.glob("matchid-*.json")):
        match_id = mp.stem.split("-", 1)[1]
        with open(mp) as f:
            meta = json.load(f)
        try:
            shift_ms = int(get_shift(str(match_id)))
        except KeyError:
            shift_ms = 0
        start_ms = int(meta["gameCreation"]) + shift_ms
        duration_s = int(meta["gameDuration"])
        matches.append({
            "match_id": str(match_id),
            "start_ms": start_ms,
            "duration_s": duration_s,
            "end_ms": start_ms + duration_s * 1000,
            "shift_ms": shift_ms,
            "match_file": mp.name,
        })
    return matches


# load lol match windows if /derived have no data
def _load_lol_match_windows_from_raw(
    window_start_min=LOL_MATCH_WINDOW_START_MIN,
    window_end_min=LOL_MATCH_WINDOW_END_MIN,
    min_events=LOL_MATCH_MIN_EVENTS,
):
    if window_end_min <= window_start_min:
        raise ValueError("window_end_min must be > window_start_min")

    get_shift = _load_lol_shift_fn()
    root = Path(LOL_DATA_ROOT) / "sessions"
    session_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    records = []
    n_keyloggers = 0
    n_skip_short_match = 0
    n_skip_sparse = 0
    n_skip_empty = 0

    for session_dir in session_dirs:
        matches = list_lol_matches(session_dir, get_shift=get_shift)
        if not matches:
            continue
        keyloggers = sorted(session_dir.glob("lol-*-keylogger-new.txt"))

        for keylogger_path in keyloggers:
            n_keyloggers += 1
            participant = keylogger_path.stem.split("-")[1]
            abs_df = _read_lol_moved_abs(keylogger_path)
            if abs_df.empty:
                n_skip_empty += 1
                continue

            # get match 10-13 minutes
            for match in matches:
                if match["duration_s"] < window_end_min * 60:
                    n_skip_short_match += 1
                    continue
                window_start_ms = match["start_ms"] + int(window_start_min * 60 * 1000)
                window_end_ms = match["start_ms"] + int(window_end_min * 60 * 1000)
                window_end_ms = min(window_end_ms, match["end_ms"])
                if window_end_ms <= window_start_ms:
                    n_skip_short_match += 1
                    continue

                window_abs = abs_df[(abs_df["time_ms"] >= window_start_ms) & (abs_df["time_ms"] < window_end_ms)]
                if len(window_abs) < int(min_events):
                    n_skip_sparse += 1
                    continue
                mouse = _abs_rows_to_mouse_df(window_abs)
                if mouse is None or len(mouse) < 2:
                    n_skip_sparse += 1
                    continue

                session_date = session_dir.name
                game_id = f"{session_date}_m{match['match_id']}_p{participant}"
                records.append({
                    "mouse": mouse,
                    "userId": f"lol_{participant}",
                    "gameId": game_id,
                    "source_file": keylogger_path.name,
                    "session_date": session_date,
                    "participant": participant,
                    "match_id": match["match_id"],
                    "window_start_min": float(window_start_min),
                    "window_end_min": float(window_end_min),
                    "match_duration_s": match["duration_s"],
                    "n_events_abs": int(len(window_abs)),
                    "keylogger_path": str(keylogger_path),
                })

    summary = {
        "n_records": len(records),
        "n_keyloggers": n_keyloggers,
        "n_skip_short_match": n_skip_short_match,
        "n_skip_sparse": n_skip_sparse,
        "n_skip_empty": n_skip_empty,
        "window_start_min": window_start_min,
        "window_end_min": window_end_min,
        "min_events": min_events,
        "source": "raw",
    }
    return records, summary


def _cache_manifest_matches(manifest, window_start_min, window_end_min, min_events):
    if not manifest:
        return False
    return (
        manifest.get("timestamp_parse") == LOL_TIMESTAMP_PARSE
        and float(manifest.get("window_start_min", -1)) == float(window_start_min)
        and float(manifest.get("window_end_min", -1)) == float(window_end_min)
        and int(manifest.get("min_events", -1)) == int(min_events)
    )


def load_lol_match_windows_from_cache(
    window_start_min=LOL_MATCH_WINDOW_START_MIN,
    window_end_min=LOL_MATCH_WINDOW_END_MIN,
    min_events=LOL_MATCH_MIN_EVENTS,
):
    cache_dir = Path(LOL_DERIVED_ROOT) / f"mouse_windows_{int(window_start_min)}_{int(window_end_min)}"

    manifest_path = cache_dir / "MANIFEST.json"
    index_path = cache_dir / "index.csv"
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"LoL window cache incomplete: {cache_dir}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    # check if manifest matches the window start, end, and min events
    if not _cache_manifest_matches(manifest, window_start_min, window_end_min, min_events):
        raise ValueError(
            f"Cache MANIFEST mismatch at {cache_dir}: {manifest} "
            f"(want start={window_start_min} end={window_end_min} "
            f"min_events={min_events} parse={LOL_TIMESTAMP_PARSE})"
        )

    index_df = pd.read_csv(
        index_path,
        dtype={"match_id": str, "participant": str, "gameId": str, "userId": str},
    )

    records = []
    # load records from index.csv
    for row in index_df.itertuples(index=False):
        mouse_path = cache_dir / row.mouse_file
        mouse = pd.read_csv(mouse_path)
        records.append({
            "mouse": mouse,
            "userId": row.userId,
            "gameId": row.gameId,
            "source_file": row.source_file,
            "session_date": row.session_date,
            "participant": str(row.participant),
            "match_id": str(row.match_id),
            "window_start_min": float(row.window_start_min),
            "window_end_min": float(row.window_end_min),
            "match_duration_s": int(row.match_duration_s),
            "n_events_abs": int(row.n_events_abs),
            "keylogger_path": getattr(row, "keylogger_path", "") or "",
        })

    # count number of keylogger sessions
    if len(index_df):
        n_keylogger_sessions = int(
            index_df[["session_date", "source_file"]].drop_duplicates().shape[0]
        )
    else:
        n_keylogger_sessions = 0
    summary = {
        "n_records": len(records),
        "n_keyloggers": int(manifest.get("n_keyloggers", n_keylogger_sessions)),
        "n_skip_short_match": int(manifest.get("n_skip_short_match", 0)),
        "n_skip_sparse": int(manifest.get("n_skip_sparse", 0)),
        "n_skip_empty": int(manifest.get("n_skip_empty", 0)),
        "window_start_min": window_start_min,
        "window_end_min": window_end_min,
        "min_events": min_events,
        "source": "cache",
        "cache_dir": str(cache_dir),
    }
    return records, summary


def load_lol_match_windows(
    window_start_min=LOL_MATCH_WINDOW_START_MIN,
    window_end_min=LOL_MATCH_WINDOW_END_MIN,
    min_events=LOL_MATCH_MIN_EVENTS,
):
    try:
        return load_lol_match_windows_from_cache(
            window_start_min=window_start_min,
            window_end_min=window_end_min,
            min_events=min_events,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(
            f"LoL window cache unavailable ({exc}); falling back to raw parse. "
            f"Rebuild with: python scripts/build_lol_mouse_windows.py"
        )
        return _load_lol_match_windows_from_raw(
            window_start_min=window_start_min,
            window_end_min=window_end_min,
            min_events=min_events,
        )


# --- CSGO: eye-vector -> mouse conversion & windowing ---

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

    result = pd.DataFrame({"dx": d_yaw, "dy": d_pitch, "time": (time_s[1:] - time_s[0]) * 1000.0})
    keep = (result["dx"].abs() <= teleport_deg) & (result["dy"].abs() <= teleport_deg)
    n_drop = int((~keep).sum())
    result = result.loc[keep].reset_index(drop=True)


    meta = {
        "n_in": int(len(time_s)),
        "n_out": int(len(result)),
        "n_teleport_dropped": n_drop,
        "teleport_frac": n_drop / max(len(d_yaw), 1),
    }
    return result, meta

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


_CSGO_EYE_COLS = ["time", "eyeVectorX", "eyeVectorY", "eyeVectorZ"]


def find_gameflt_files(root=CSGO_DATA_ROOT):
    files = sorted(Path(root).rglob("gameFlt.csv"))
    print(f"Found {len(files)} gameFlt.csv under {root}")
    return files


def load_eye_csv(path):
    df = pd.read_csv(path, usecols=_CSGO_EYE_COLS)
    finite = np.isfinite(df[["eyeVectorX", "eyeVectorY", "eyeVectorZ"]]).all(axis=1)
    return df.loc[finite].reset_index(drop=True)


def gameflt_to_mouse(path):
    flt = load_eye_csv(path)
    mouse_df, meta = eye_vectors_to_mouse_df(
        flt["time"], flt["eyeVectorX"], flt["eyeVectorY"], flt["eyeVectorZ"]
    )
    return mouse_df, meta, flt
