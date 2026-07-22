import numpy as np
import pandas as pd

from src.config import (
    SEGMENT_MS_RANGE,
    MIN_EVENTS,
    TARGET_DURATION_MS,
    RNG_SEED,
)


def _subsample(arr, max_samples, rng):
    arr = np.asarray(arr, dtype=float)
    if len(arr) > max_samples:
        arr = rng.choice(arr, size=max_samples, replace=False)
    return arr


def _clip_percentiles(arr, p_low=10.0, p_high=90.0):
    """Keep the bulk of a heavy-tailed distribution (drop extreme flicks)."""
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.percentile(arr, [p_low, p_high])
    kept = arr[(arr >= lo) & (arr <= hi)]
    return kept if len(kept) else arr


def collect_human_motion_samples(
    mouse_dfs,
    max_samples=200_000,
    min_step=None,
    step_p_low=25.0,
    step_p_high=90.0,
    rng=None,
):
    """
    Pool human timing + kinematics for bot generation.

    Steps/angles ignore near-zero jitter (adaptive floor) and step lengths are
    clipped to [p_low, p_high] so smooth bots do not lock a rare huge flick
    for an entire 20–60 event segment (that made trajectories explode and
    collapsed cross-game smooth separation).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    dt_chunks, step_chunks = [], []
    dx_chunks, dy_chunks = [], []
    for mouse in mouse_dfs:
        if mouse is None or len(mouse) < 2:
            continue
        df = mouse.sort_values("time")
        dt = df["time"].diff().to_numpy(dtype=float)
        dx = df["dx"].to_numpy(dtype=float)
        dy = df["dy"].to_numpy(dtype=float)
        valid_dt = np.isfinite(dt) & (dt > 0)
        if valid_dt.any():
            dt_chunks.append(dt[valid_dt])
        step = np.hypot(dx, dy)
        ok = np.isfinite(step)
        if ok.any():
            step_chunks.append(step[ok])
            dx_chunks.append(dx[ok])
            dy_chunks.append(dy[ok])

    if not dt_chunks:
        raise ValueError("collect_human_motion_samples: no positive dt values found")
    if not step_chunks:
        raise ValueError("collect_human_motion_samples: no steps found")

    all_steps = np.concatenate(step_chunks)
    positive = all_steps[all_steps > 0]
    if min_step is None:
        # Ignore micro-jitter / near-idle events when learning angles & steps.
        min_step = float(np.percentile(positive, 40)) if len(positive) else 1e-9
        min_step = max(min_step, 1e-9)

    dx_all = np.concatenate(dx_chunks)
    dy_all = np.concatenate(dy_chunks)
    step_all = np.hypot(dx_all, dy_all)
    moving = step_all >= min_step
    if not moving.any():
        raise ValueError("collect_human_motion_samples: no moving steps above floor")

    raw_steps = step_all[moving]
    # Bulk of the moving distribution only — not rare flicks (would be held
    # for a whole smooth segment and explode the path).
    step_samples = _clip_percentiles(raw_steps, step_p_low, step_p_high)
    lo, hi = np.percentile(raw_steps, [step_p_low, step_p_high])
    ang_all = np.arctan2(dy_all[moving], dx_all[moving])
    ang_keep = (raw_steps >= lo) & (raw_steps <= hi)
    angle_samples = ang_all[ang_keep] if ang_keep.any() else ang_all

    return {
        "dt_samples": _subsample(np.concatenate(dt_chunks), max_samples, rng),
        "step_samples": _subsample(step_samples, max_samples, rng),
        "angle_samples": _subsample(angle_samples, max_samples, rng),
        "min_step_floor": float(min_step),
        "step_clip": (float(lo), float(hi)),
    }



def collect_dt_samples(mouse_dfs, max_samples=200_000, rng=None):
    """Backward-compatible alias: only the dt array."""
    return collect_human_motion_samples(mouse_dfs, max_samples=max_samples, rng=rng)[
        "dt_samples"
    ]


def median_trace_duration_ms(mouse_dfs, default_ms=TARGET_DURATION_MS):
    durs = [
        float(m["time"].iloc[-1])
        for m in mouse_dfs
        if m is not None and len(m) > 0
    ]
    return float(np.median(durs)) if durs else float(default_ms)


def _sample_dt_ms(rng, dt_samples):
    return max(1, int(round(float(rng.choice(dt_samples)))))


# stitch bot
def build_segments(
    mouse_df,
    segment_ms_range=SEGMENT_MS_RANGE,
    min_events=MIN_EVENTS,
    rng=None,
    segment_ms=None,  # ignored; kept so old keyword calls do not crash
):
    """
    Cut human traces into chunks with randomized duration to avoid a fixed
    ~2s periodicity (segment_ms_range default 1500–2500 ms).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    lo, hi = int(segment_ms_range[0]), int(segment_ms_range[1])
    if hi < lo:
        lo, hi = hi, lo

    sorted_events = mouse_df.sort_values("time").reset_index(drop=True)
    segments = []
    if len(sorted_events) < min_events:
        return segments

    times = sorted_events["time"].to_numpy()
    start = 0
    target_ms = int(rng.integers(lo, hi + 1))
    for index in range(len(times)):
        if times[index] - times[start] >= target_ms:
            if index + 1 - start >= min_events:
                segment = sorted_events.iloc[start:index + 1].copy()
                segment["rel_time"] = segment["time"] - segment["time"].iloc[0]
                segments.append(
                    segment[["dx", "dy", "rel_time"]].reset_index(drop=True)
                )
            start = index + 1
            target_ms = int(rng.integers(lo, hi + 1))
    return segments


def stitch_bot_game(
    segments,
    dt_samples,
    target_duration_ms=TARGET_DURATION_MS,
    rng=None,
):
    """Block-bootstrap bot; inter-segment gaps drawn from human dt samples."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    dt_samples = np.asarray(dt_samples, dtype=float)
    if len(dt_samples) == 0:
        raise ValueError("stitch_bot_game: dt_samples is empty")

    parts = []
    current_time = 0
    while current_time < target_duration_ms:
        segment = segments[rng.integers(0, len(segments))]
        t = current_time + segment["rel_time"].to_numpy()
        keep = t < target_duration_ms
        parts.append(pd.DataFrame({
            "dx": segment["dx"].to_numpy()[keep],
            "dy": segment["dy"].to_numpy()[keep],
            "time": t[keep],
        }))
        gap = _sample_dt_ms(rng, dt_samples)
        current_time = current_time + segment["rel_time"].iloc[-1] + gap

    return pd.concat(parts, ignore_index=True)


# smooth bot
def generate_smooth_bot_game(
    n_events=5800,
    dt_samples=None,
    step_samples=None,
    angle_samples=None,
    segment_len_range=(20, 60),
    jitter=1.5,
    seed=None,
    round_deltas=True,
    # accepted from **estimate_smooth_params / old callers; unused for sampling
    mean_interval_ms=None,
    step_median=None,
    base_speed_range=None,
    legacy_base_speed_times_dt=None,
    min_step_floor=None,
    step_clip=None,
):
    """
    Mechanical bot:
      - intervals from human dt
      - heading from human atan2(dy, dx) (moving events only)
      - step length from human hypot(dx, dy) (percentile-clipped bulk)
    """
    if dt_samples is None or step_samples is None or angle_samples is None:
        raise ValueError(
            "generate_smooth_bot_game: dt_samples, step_samples, and "
            "angle_samples are required"
        )
    dt_samples = np.asarray(dt_samples, dtype=float)
    step_samples = np.asarray(step_samples, dtype=float)
    angle_samples = np.asarray(angle_samples, dtype=float)
    if min(len(dt_samples), len(step_samples), len(angle_samples)) == 0:
        raise ValueError("generate_smooth_bot_game: empty motion sample array")

    rng = np.random.default_rng(seed)
    dx_list, dy_list, times = [], [], []
    current_time = 0
    events_done = 0

    while events_done < n_events:
        angle = float(rng.choice(angle_samples))
        step = float(rng.choice(step_samples))
        seg_len = int(rng.integers(*segment_len_range))

        base_dx = np.cos(angle) * step
        base_dy = np.sin(angle) * step

        for _ in range(seg_len):
            if events_done >= n_events:
                break

            dx = base_dx + rng.normal(0, jitter)
            dy = base_dy + rng.normal(0, jitter)
            current_time += _sample_dt_ms(rng, dt_samples)

            if round_deltas:
                dx, dy = round(dx), round(dy)
            dx_list.append(dx)
            dy_list.append(dy)
            times.append(current_time)
            events_done += 1

    return pd.DataFrame({"dx": dx_list, "dy": dy_list, "time": times})


def estimate_smooth_params(
    human_df,
    dt_samples,
    step_samples=None,
    angle_samples=None,
    min_step_floor=None,
    step_clip=None,
    **_extra,
):
    """
    Build smooth-bot kwargs from empirical human motion.

    step_median / jitter use median hypot of the (already clipped) step pool,
    not avg_speed * mean_dt — same units as dx/dy, no 1000x unit trap.
    """
    dt_samples = np.asarray(dt_samples, dtype=float)
    if len(dt_samples) == 0:
        raise ValueError("estimate_smooth_params: dt_samples is empty")
    if step_samples is None or angle_samples is None:
        raise ValueError(
            "estimate_smooth_params: step_samples and angle_samples are required"
        )
    step_samples = np.asarray(step_samples, dtype=float)
    angle_samples = np.asarray(angle_samples, dtype=float)
    if len(step_samples) == 0 or len(angle_samples) == 0:
        raise ValueError("estimate_smooth_params: empty step/angle samples")

    step_median = float(np.median(step_samples))
    if human_df is not None and "avg_speed" in human_df and "mean_dt" in human_df:
        legacy = float(human_df["avg_speed"].median() * human_df["mean_dt"].median())
    else:
        legacy = float("nan")

    out = {
        "mean_interval_ms": float(np.median(dt_samples)),
        "step_median": step_median,
        "jitter": step_median * 0.1,
        "dt_samples": dt_samples,
        "step_samples": step_samples,
        "angle_samples": angle_samples,
        "legacy_base_speed_times_dt": legacy,
    }
    if min_step_floor is not None:
        out["min_step_floor"] = min_step_floor
    if step_clip is not None:
        out["step_clip"] = step_clip
    return out


def smooth_params_for_print(params):
    """Omit large sample arrays when logging."""
    dt = np.asarray(params["dt_samples"], dtype=float)
    step = np.asarray(params["step_samples"], dtype=float)
    ang = np.asarray(params["angle_samples"], dtype=float)
    out = {
        "mean_interval_ms": params["mean_interval_ms"],
        "step_median": params["step_median"],
        "jitter": params["jitter"],
        "n_dt_samples": len(dt),
        "n_step_samples": len(step),
        "n_angle_samples": len(ang),
        "dt_std": float(np.std(dt)),
        "step_p05": float(np.percentile(step, 5)),
        "step_p95": float(np.percentile(step, 95)),
    }
    if "legacy_base_speed_times_dt" in params:
        out["legacy_base_speed_times_dt"] = params["legacy_base_speed_times_dt"]
    if "step_clip" in params:
        out["step_clip"] = params["step_clip"]
    if "min_step_floor" in params:
        out["min_step_floor"] = params["min_step_floor"]
    return out
