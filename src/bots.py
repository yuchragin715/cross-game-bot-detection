import numpy as np
import pandas as pd
from src.config import (
    SEGMENT_MS_RANGE,
    MIN_EVENTS,
    TARGET_DURATION_MS,
    RNG_SEED,
    VAE_POOL_SEGMENTS,
)


def _subsample(arr, max_samples, rng):
    arr = np.asarray(arr, dtype=float)
    if len(arr) > max_samples:
        arr = rng.choice(arr, size=max_samples, replace=False)
    return arr


def _clip_percentiles(arr, p_low=10.0, p_high=90.0):
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
        valid_step = np.isfinite(step)
        if valid_step.any():
            step_chunks.append(step[valid_step])
            dx_chunks.append(dx[valid_step])
            dy_chunks.append(dy[valid_step])

    if not dt_chunks:
        raise ValueError("collect_human_motion_samples: no positive dt values found")
    if not step_chunks:
        raise ValueError("collect_human_motion_samples: no steps found")

    all_steps = np.concatenate(step_chunks)
    valid_steps = all_steps[all_steps > 0]
    if min_step is None:
        # Ignore micro-jitter / near-idle events when learning angles & steps
        min_step = float(np.percentile(valid_steps, 40)) if len(valid_steps) else 1e-9
        min_step = max(min_step, 1e-9)

    dx_all = np.concatenate(dx_chunks)
    dy_all = np.concatenate(dy_chunks)
    step_all = np.hypot(dx_all, dy_all)
    moving = step_all >= min_step
    if not moving.any():
        raise ValueError("collect_human_motion_samples: no moving steps above floor")

    raw_steps = step_all[moving]
    step_samples = _clip_percentiles(raw_steps, step_p_low, step_p_high)
    p_low, p_high = np.percentile(raw_steps, [step_p_low, step_p_high])
    angle_all = np.arctan2(dy_all[moving], dx_all[moving])
    angle_keep = (raw_steps >= p_low) & (raw_steps <= p_high)
    angle_samples = angle_all[angle_keep] if angle_keep.any() else angle_all

    return {
        "dt_samples": _subsample(np.concatenate(dt_chunks), max_samples, rng),
        "dt_by_session": [np.asarray(c, dtype=float) for c in dt_chunks],
        "step_samples": _subsample(step_samples, max_samples, rng),
        "angle_samples": _subsample(angle_samples, max_samples, rng),
        "min_step_floor": float(min_step),
        "step_clip": (float(p_low), float(p_high)),
    }


def median_trace_duration_ms(mouse_dfs, default_ms=TARGET_DURATION_MS):
    # Return the median duration of the mouse traces
    durations = [
        float(m["time"].iloc[-1])
        for m in mouse_dfs
        if m is not None and len(m) > 0
    ]
    return float(np.median(durations)) if durations else float(default_ms)


def _sample_dt_ms(rng, dt_samples):
    return max(1, int(round(float(rng.choice(dt_samples)))))

# if dt_by_session exists return a random session, else return dt_samples
def _resolve_dt_samples(rng, dt_samples=None, dt_by_session=None):
    if dt_by_session is not None:
        sessions = [np.asarray(s, dtype=float) for s in dt_by_session if len(s) > 0]
        if sessions:
            return sessions[int(rng.integers(0, len(sessions)))]
    if dt_samples is None:
        raise ValueError("_resolve_dt_samples: need dt_samples or dt_by_session")
    dt_samples = np.asarray(dt_samples, dtype=float)
    if len(dt_samples) == 0:
        raise ValueError("_resolve_dt_samples: empty dt_samples")
    return dt_samples


def _mean_interval_ms(dt_samples, dt_by_session=None):
    if dt_by_session is not None:
        means = [
            float(np.mean(np.asarray(s, dtype=float)))
            for s in dt_by_session
            if len(s) > 0
        ]
        if means:
            return float(np.median(means))
    return float(np.median(np.asarray(dt_samples, dtype=float)))


# stitch bot
def build_segments(
    mouse_df,
    segment_ms_range=SEGMENT_MS_RANGE,
    min_events=MIN_EVENTS,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    lower_ms, upper_ms = int(segment_ms_range[0]), int(segment_ms_range[1])
    if upper_ms < lower_ms:
        lower_ms, upper_ms = upper_ms, lower_ms

    sorted_events = mouse_df.sort_values("time").reset_index(drop=True)
    segments = []
    if len(sorted_events) < min_events:
        return segments

    times = sorted_events["time"].to_numpy()
    start = 0
    # Randomly sample a segment length between 1500ms and 2500ms
    target_ms = int(rng.integers(lower_ms, upper_ms + 1))
    for index in range(len(times)):
        if times[index] - times[start] < target_ms:
            continue
        if index + 1 - start >= min_events:
            segment = sorted_events.iloc[start:index + 1].copy()
            segment["rel_time"] = segment["time"] - segment["time"].iloc[0]
            segments.append(
                segment[["dx", "dy", "rel_time"]].reset_index(drop=True)
            )
        start = index + 1
        target_ms = int(rng.integers(lower_ms, upper_ms + 1))
    return segments


def stitch_bot_game(
    segments,
    dt_samples,
    target_duration_ms=TARGET_DURATION_MS,
    rng=None,
    dt_by_session=None,
):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    # select the dt samples, expecially for red eclipse
    dt_samples = _resolve_dt_samples(rng, dt_samples=dt_samples, dt_by_session=dt_by_session)

    parts = []
    current_time = 0
    # stitch the segments together
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
SMOOTH_GENERATOR_KEYS = (
    "dt_samples",
    "dt_by_session",
    "step_samples",
    "angle_samples",
    "jitter",
)

def smooth_generator_params(params):
    return {k: params[k] for k in SMOOTH_GENERATOR_KEYS if k in params}

def generate_smooth_bot_game(
    n_events=5800,
    dt_samples=None,
    step_samples=None,
    angle_samples=None,
    segment_len_range=(20, 60),
    jitter=1.5,
    seed=None,
    round_deltas=True,
    target_duration_ms=None,
    dt_by_session=None,
):
    if dt_samples is None or step_samples is None or angle_samples is None:
        raise ValueError(
            "generate_smooth_bot_game: dt_samples, step_samples, and "
            "angle_samples are required"
        )
    step_samples = np.asarray(step_samples, dtype=float)
    angle_samples = np.asarray(angle_samples, dtype=float)
    if min(len(step_samples), len(angle_samples)) == 0:
        raise ValueError("generate_smooth_bot_game: empty motion sample array")

    rng = np.random.default_rng(seed)
    dt_samples = _resolve_dt_samples(
        rng, dt_samples=dt_samples, dt_by_session=dt_by_session
    )
    n_events = int(n_events)
    if n_events < 1:
        raise ValueError("generate_smooth_bot_game: n_events < 1")

    dx_list, dy_list, times = [], [], []
    current_time, events_done = 0, 0

    while events_done < n_events:
        if target_duration_ms is not None and current_time >= target_duration_ms:
            break

        angle = float(rng.choice(angle_samples))
        step = float(rng.choice(step_samples))
        seg_len = int(rng.integers(*segment_len_range))

        base_dx = np.cos(angle) * step
        base_dy = np.sin(angle) * step

        for _ in range(seg_len):
            if events_done >= n_events:
                break
            if target_duration_ms is not None and current_time >= target_duration_ms:
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

# generate the smooth bot parameters
def estimate_smooth_params(
    human_df,
    dt_samples,
    step_samples=None,
    angle_samples=None,
    min_step_floor=None,
    step_clip=None,
    dt_by_session=None,
    **_extra,
):
    if dt_by_session is None:
        dt_by_session = _extra.get("dt_by_session")
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

    result = {
        "mean_interval_ms": _mean_interval_ms(dt_samples, dt_by_session),
        "step_median": step_median,
        "jitter": step_median * 0.1,
        "dt_samples": dt_samples,
        "step_samples": step_samples,
        "angle_samples": angle_samples,
        "legacy_base_speed_times_dt": legacy,
    }
    if dt_by_session is not None:
        result["dt_by_session"] = dt_by_session
    if min_step_floor is not None:
        result["min_step_floor"] = min_step_floor
    if step_clip is not None:
        result["step_clip"] = step_clip
    return result


def smooth_params_for_print(params):
    dt = np.asarray(params["dt_samples"], dtype=float)
    step = np.asarray(params["step_samples"], dtype=float)
    ang = np.asarray(params["angle_samples"], dtype=float)
    print_result = {
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
    if "dt_by_session" in params and params["dt_by_session"] is not None:
        print_result["n_dt_sessions"] = len(params["dt_by_session"])
    if "legacy_base_speed_times_dt" in params:
        print_result["legacy_base_speed_times_dt"] = params["legacy_base_speed_times_dt"]
    if "step_clip" in params:
        print_result["step_clip"] = params["step_clip"]
    if "min_step_floor" in params:
        print_result["min_step_floor"] = params["min_step_floor"]
    return print_result

# bezier bot
def _ease_out_quad(u):
    u = np.asarray(u, dtype=float)
    return 1.0 - (1.0 - u) ** 2

def _cubic_bezier_points(p0, p1, p2, p3, n_points, ease=True):
    p0 = np.asarray(p0, dtype=float).reshape(1, 2)
    p1 = np.asarray(p1, dtype=float).reshape(1, 2)
    p2 = np.asarray(p2, dtype=float).reshape(1, 2)
    p3 = np.asarray(p3, dtype=float).reshape(1, 2)
    u = np.linspace(0.0, 1.0, int(n_points))[:, None]
    t = _ease_out_quad(u) if ease else u
    w0, w1, w2, w3 = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3

def _add_normal_distortion(pts, rng, distortion):
    pts = np.asarray(pts, dtype=float).copy()
    distortion = float(distortion)
    if len(pts) < 3 or distortion <= 0:
        return pts

    tangent = np.gradient(pts, axis=0)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangent = tangent / norms
    perp = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    offset = perp * rng.normal(0.0, distortion, size=(len(pts), 1))
    offset[0] = 0.0
    offset[-1] = 0.0
    return pts + offset

def _random_stroke_controls(rng, step_samples, angle_samples, stroke_steps, bend_scale):
    step = float(rng.choice(np.asarray(step_samples, dtype=float)))
    angle = float(rng.choice(np.asarray(angle_samples, dtype=float)))
    n_events = max(int(stroke_steps), 1)
    span = step * n_events

    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    p0 = np.zeros(2)
    p3 = direction * span
    p1 = direction * (span / 3.0) + normal * rng.normal(0.0, bend_scale * span)
    p2 = direction * (2.0 * span / 3.0) + normal * rng.normal(0.0, bend_scale * span)
    return p0, p1, p2, p3

def generate_bezier_stroke(
    rng,
    step_samples,
    angle_samples,
    stroke_steps,
    distortion,
    bend_scale=0.15,
    ease=True,
):
    n_points = max(int(stroke_steps) + 1, 3)
    p0, p1, p2, p3 = _random_stroke_controls(
        rng, step_samples, angle_samples, stroke_steps, bend_scale
    )
    # create the cubic bezier points
    pts = _cubic_bezier_points(p0, p1, p2, p3, n_points, ease=ease)
    # add the normal distortion
    pts = _add_normal_distortion(pts, rng, distortion)
    d = np.diff(pts, axis=0)
    return d[:, 0], d[:, 1]

def estimate_bezier_params(
    human_df,
    dt_samples,
    step_samples=None,
    angle_samples=None,
    stroke_points_range=(20, 60),
    bend_scale=0.15,
    min_step_floor=None,
    step_clip=None,
    dt_by_session=None,
    **_extra,
):
    if dt_by_session is None:
        dt_by_session = _extra.get("dt_by_session")
    base = estimate_smooth_params(
        human_df,
        dt_samples,
        step_samples=step_samples,
        angle_samples=angle_samples,
        min_step_floor=min_step_floor,
        step_clip=step_clip,
        dt_by_session=dt_by_session,
    )
    base.update({
        "stroke_points_range": tuple(stroke_points_range),
        "distortion": float(base["step_median"] * 0.1),
        "bend_scale": float(bend_scale),
        "ease": True,
    })
    return base


def bezier_params_for_print(params):
    print_result = smooth_params_for_print(params)
    for k in ("stroke_points_range", "distortion", "bend_scale", "ease"):
        if k in params:
            print_result[k] = params[k]
    return print_result


def generate_bezier_bot_game(
    n_events=5800,
    dt_samples=None,
    step_samples=None,
    angle_samples=None,
    stroke_points_range=(20, 60),
    distortion=1.5,
    bend_scale=0.15,
    ease=True,
    seed=None,
    round_deltas=True,
    target_duration_ms=None,
    dt_by_session=None,
    mean_interval_ms=None,
    step_median=None,
    jitter=None,
    base_speed_range=None,
    legacy_base_speed_times_dt=None,
    min_step_floor=None,
    step_clip=None,
):
    if dt_samples is None or step_samples is None or angle_samples is None:
        raise ValueError(
            "generate_bezier_bot_game: dt_samples, step_samples, and "
            "angle_samples are required"
        )
    step_samples = np.asarray(step_samples, dtype=float)
    angle_samples = np.asarray(angle_samples, dtype=float)
    if min(len(step_samples), len(angle_samples)) == 0:
        raise ValueError("generate_bezier_bot_game: empty motion sample array")

    rng = np.random.default_rng(seed)
    dt_samples = _resolve_dt_samples(
        rng, dt_samples=dt_samples, dt_by_session=dt_by_session
    )
    min_steps, max_steps = int(stroke_points_range[0]), int(stroke_points_range[1])
    if max_steps < min_steps:
        min_steps, max_steps = max_steps, min_steps

    dx_list, dy_list, times = [], [], []
    current_time, events_done = 0, 0

    while events_done < n_events:
        if target_duration_ms is not None and current_time >= target_duration_ms:
            break

        stroke_steps = int(rng.integers(min_steps, max_steps + 1))
        stroke_steps = min(stroke_steps, n_events - events_done)
        if stroke_steps < 1:
            break

        dx_s, dy_s = generate_bezier_stroke(
            rng,
            step_samples,
            angle_samples,
            stroke_steps,
            distortion=distortion,
            bend_scale=bend_scale,
            ease=ease,
        )

        for dx, dy in zip(dx_s, dy_s):
            if events_done >= n_events:
                break
            if target_duration_ms is not None and current_time >= target_duration_ms:
                break
            current_time += _sample_dt_ms(rng, dt_samples)
            if round_deltas:
                dx, dy = round(dx), round(dy)
                if dx == 0 and dy == 0:
                    continue
            dx_list.append(float(dx))
            dy_list.append(float(dy))
            times.append(current_time)
            events_done += 1

    return pd.DataFrame({"dx": dx_list, "dy": dy_list, "time": times})


def generate_bot_mouse_games(
    mouse_dfs,
    human_feat_df,
    rng_seed=RNG_SEED,
    round_deltas=True,
    id_prefix="bot",
    vae_bundle=None,
):
    traces = [mouse for mouse in mouse_dfs if mouse is not None and len(mouse) >= 2]
    if not traces:
        raise ValueError("generate_bot_mouse_games: no usable mouse traces")
    n_bots = len(human_feat_df)
    if n_bots < 1:
        raise ValueError("generate_bot_mouse_games: n_bots < 1")
    if vae_bundle is None:
        raise ValueError("generate_bot_mouse_games: vae_bundle is required")

    from src.vae_bot import generate_vae_bot_games

    rng = np.random.default_rng(rng_seed)
    # build the segment pool
    segment_pool = []
    for mouse in traces:
        segment_pool.extend(build_segments(mouse, rng=rng))
    if not segment_pool:
        raise ValueError("generate_bot_mouse_games: empty segment pool")

    motion = collect_human_motion_samples(traces, rng=rng)
    target_ms = median_trace_duration_ms(traces)
    median_events = int(human_feat_df["n_events"].median())

    mean_dt = float(np.median(motion["dt_samples"])) if len(motion["dt_samples"]) else 1.0
    event_cap = max(median_events, int(target_ms / max(mean_dt, 1.0)) * 3, 1)
    smooth_params = estimate_smooth_params(human_feat_df, **motion)
    bezier_params = estimate_bezier_params(human_feat_df, **motion)

    # generate the bot traces for each bot type
    stitch_traces, smooth_traces, bezier_traces, vae_traces, stitch_ids, smooth_ids, bezier_ids, vae_ids = [], [], [], [], [], [], [], []
    for i in range(n_bots):
        stitch_traces.append(
            stitch_bot_game(
                segment_pool,
                dt_samples=motion["dt_samples"],
                dt_by_session=motion["dt_by_session"],
                target_duration_ms=target_ms,
                rng=rng,
            )
        )

        smooth_traces.append(
            generate_smooth_bot_game(
                n_events=event_cap,
                seed=rng_seed + 1000 + i,
                round_deltas=round_deltas,
                target_duration_ms=target_ms,
                **smooth_generator_params(smooth_params),
            )
        )

        bezier_traces.append(
            generate_bezier_bot_game(
                n_events=event_cap,
                seed=rng_seed + 2000 + i,
                round_deltas=round_deltas,
                target_duration_ms=target_ms,
                **bezier_params,
            )
        )

        stitch_ids.append(f"{id_prefix}_stitch_{i}")
        smooth_ids.append(f"{id_prefix}_smooth_{i}")
        bezier_ids.append(f"{id_prefix}_bezier_{i}")
        vae_ids.append(f"{id_prefix}_vae_{i}")

    vae_traces = generate_vae_bot_games(
        vae_bundle,
        n_games=n_bots,
        dt_samples=motion["dt_samples"],
        dt_by_session=motion["dt_by_session"],
        target_duration_ms=target_ms,
        n_pool_segments=VAE_POOL_SEGMENTS,
        rng=rng,
    )

    return {
        "stitch": stitch_traces,
        "smooth": smooth_traces,
        "bezier": bezier_traces,
        "vae": vae_traces,
        "stitch_ids": stitch_ids,
        "smooth_ids": smooth_ids,
        "bezier_ids": bezier_ids,
        "vae_ids": vae_ids,
        "target_ms": target_ms,
    }
