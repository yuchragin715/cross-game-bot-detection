import numpy as np
import pandas as pd

from src.config import SEGMENT_MS, MIN_EVENTS, TARGET_DURATION_MS, RNG_SEED

# stitch bot
def build_segments(mouse_df, segment_ms=SEGMENT_MS, min_events=MIN_EVENTS):
    sorted_events = mouse_df.sort_values("time").reset_index(drop=True)
    segments = []
    if len(sorted_events) < min_events:
        return segments

    times = sorted_events["time"].to_numpy()
    start = 0
    for index in range(len(times)):
        if times[index] - times[start] >= segment_ms:
            if index + 1 - start >= min_events:
                segment = sorted_events.iloc[start:index + 1].copy()
                segment["rel_time"] = segment["time"] - segment["time"].iloc[0]
                segments.append(
                    segment[["dx", "dy", "rel_time"]].reset_index(drop=True)
                )
            start = index + 1
    return segments


def stitch_bot_game(segments, target_duration_ms=TARGET_DURATION_MS, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

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
        current_time = current_time + segment["rel_time"].iloc[-1] + int(rng.integers(20, 80))

    return pd.concat(parts, ignore_index=True)

# smooth bot
def generate_smooth_bot_game(
    n_events=5800,
    mean_interval_ms=17,
    segment_len_range=(20, 60),
    base_speed_range=(5, 25),
    jitter=1.5,
    seed=None,
    round_deltas=True,
):
    rng = np.random.default_rng(seed)

    dx_list, dy_list, times = [], [], []
    current_time = 0
    events_done = 0

    while events_done < n_events:
        angle = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(*base_speed_range)
        seg_len = rng.integers(*segment_len_range)

        base_dx = np.cos(angle) * speed
        base_dy = np.sin(angle) * speed

        for _ in range(seg_len):
            if events_done >= n_events:
                break

            dx = base_dx + rng.normal(0, jitter)
            dy = base_dy + rng.normal(0, jitter)
            interval_ms = max(1, int(rng.normal(mean_interval_ms, 2)))
            current_time += interval_ms

            if round_deltas:
                dx, dy = round(dx), round(dy)
            dx_list.append(dx)
            dy_list.append(dy)
            times.append(current_time)
            events_done += 1

    return pd.DataFrame({"dx": dx_list, "dy": dy_list, "time": times})


def estimate_smooth_params(human_df):
    mean_interval_ms = float(human_df["mean_dt"].median())
    avg_speed = float(human_df["avg_speed"].median())

    base = avg_speed * mean_interval_ms
    return {
        "mean_interval_ms": mean_interval_ms,
        "base_speed_range": (base * 0.5, base * 1.5),
        "jitter": base * 0.1,
    }