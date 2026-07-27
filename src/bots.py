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
        "dt_by_session": [np.asarray(c, dtype=float) for c in dt_chunks],
        "step_samples": _subsample(step_samples, max_samples, rng),
        "angle_samples": _subsample(angle_samples, max_samples, rng),
        "min_step_floor": float(min_step),
        "step_clip": (float(lo), float(hi)),
    }



def collect_dt_samples(mouse_dfs, max_samples=200_000, rng=None):
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
    segment_ms=None,  # ignored; kept so old keyword calls do not crash
):
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
    dt_by_session=None,
):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    dt_samples = _resolve_dt_samples(rng, dt_samples=dt_samples, dt_by_session=dt_by_session)

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
    target_duration_ms=None,
    dt_by_session=None,
    mean_interval_ms=None,
    step_median=None,
    base_speed_range=None,
    legacy_base_speed_times_dt=None,
    min_step_floor=None,
    step_clip=None,
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
    current_time = 0
    events_done = 0

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

    out = {
        "mean_interval_ms": _mean_interval_ms(dt_samples, dt_by_session),
        "step_median": step_median,
        "jitter": step_median * 0.1,
        "dt_samples": dt_samples,
        "step_samples": step_samples,
        "angle_samples": angle_samples,
        "legacy_base_speed_times_dt": legacy,
    }
    if dt_by_session is not None:
        out["dt_by_session"] = dt_by_session
    if min_step_floor is not None:
        out["min_step_floor"] = min_step_floor
    if step_clip is not None:
        out["step_clip"] = step_clip
    return out


def smooth_params_for_print(params):
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
    if "dt_by_session" in params and params["dt_by_session"] is not None:
        out["n_dt_sessions"] = len(params["dt_by_session"])
    if "legacy_base_speed_times_dt" in params:
        out["legacy_base_speed_times_dt"] = params["legacy_base_speed_times_dt"]
    if "step_clip" in params:
        out["step_clip"] = params["step_clip"]
    if "min_step_floor" in params:
        out["min_step_floor"] = params["min_step_floor"]
    return out

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
    pts = _cubic_bezier_points(p0, p1, p2, p3, n_points, ease=ease)
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
    out = smooth_params_for_print(params)
    for k in ("stroke_points_range", "distortion", "bend_scale", "ease"):
        if k in params:
            out[k] = params[k]
    return out


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
    lo_s, hi_s = int(stroke_points_range[0]), int(stroke_points_range[1])
    if hi_s < lo_s:
        lo_s, hi_s = hi_s, lo_s

    dx_list, dy_list, times = [], [], []
    current_time = 0
    events_done = 0

    while events_done < n_events:
        if target_duration_ms is not None and current_time >= target_duration_ms:
            break

        stroke_steps = int(rng.integers(lo_s, hi_s + 1))
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
    n_bots=None,
    rng_seed=RNG_SEED,
    round_deltas=True,
    id_prefix="bot",
    vae_bundle=None,
    vae_n_pool_segments=256,
):
    mice = [m for m in mouse_dfs if m is not None and len(m) >= 2]
    if not mice:
        raise ValueError("generate_bot_mouse_games: no usable mouse traces")
    if n_bots is None:
        n_bots = len(human_feat_df)
    n_bots = int(n_bots)
    if n_bots < 1:
        raise ValueError("generate_bot_mouse_games: n_bots < 1")

    rng = np.random.default_rng(rng_seed)
    segment_pool = []
    for mouse in mice:
        segment_pool.extend(build_segments(mouse, rng=rng))
    if not segment_pool:
        raise ValueError("generate_bot_mouse_games: empty segment pool")

    motion = collect_human_motion_samples(mice, rng=rng)
    target_ms = median_trace_duration_ms(mice)
    median_events = int(human_feat_df["n_events"].median())

    mean_dt = float(np.median(motion["dt_samples"])) if len(motion["dt_samples"]) else 1.0
    event_cap = max(median_events, int(target_ms / max(mean_dt, 1.0)) * 3, 1)
    smooth_params = estimate_smooth_params(human_feat_df, **motion)
    bezier_params = estimate_bezier_params(human_feat_df, **motion)

    stitch_mice, smooth_mice, bezier_mice = [], [], []
    stitch_ids, smooth_ids, bezier_ids = [], [], []
    for i in range(n_bots):
        stitch_mice.append(
            stitch_bot_game(
                segment_pool,
                dt_samples=motion["dt_samples"],
                dt_by_session=motion["dt_by_session"],
                target_duration_ms=target_ms,
                rng=rng,
            )
        )
        stitch_ids.append(f"{id_prefix}_stitch_{i}")

        smooth_mice.append(
            generate_smooth_bot_game(
                n_events=event_cap,
                seed=rng_seed + 1000 + i,
                round_deltas=round_deltas,
                target_duration_ms=target_ms,
                **smooth_params,
            )
        )
        smooth_ids.append(f"{id_prefix}_smooth_{i}")

        bezier_mice.append(
            generate_bezier_bot_game(
                n_events=event_cap,
                seed=rng_seed + 2000 + i,
                round_deltas=round_deltas,
                target_duration_ms=target_ms,
                **bezier_params,
            )
        )
        bezier_ids.append(f"{id_prefix}_bezier_{i}")

    out = {
        "stitch": stitch_mice,
        "smooth": smooth_mice,
        "bezier": bezier_mice,
        "stitch_ids": stitch_ids,
        "smooth_ids": smooth_ids,
        "bezier_ids": bezier_ids,
        "n_segments": len(segment_pool),
        "n_mice": len(mice),
        "target_ms": target_ms,
        "median_events": median_events,
    }

    if vae_bundle is not None:
        from src.vae_bot import generate_vae_bot_game

        vae_mice, vae_ids = [], []
        for i in range(n_bots):
            vae_mice.append(
                generate_vae_bot_game(
                    vae_bundle,
                    dt_samples=motion["dt_samples"],
                    dt_by_session=motion["dt_by_session"],
                    target_duration_ms=target_ms,
                    n_pool_segments=vae_n_pool_segments,
                    seed=rng_seed + 3000 + i,
                )
            )
            vae_ids.append(f"{id_prefix}_vae_{i}")
        out["vae"] = vae_mice
        out["vae_ids"] = vae_ids

    return out


def generate_bot_feature_tables(
    mouse_dfs,
    human_feat_df,
    n_bots=None,
    rng_seed=RNG_SEED,
    round_deltas=True,
    id_prefix="bot",
    vae_bundle=None,
    vae_n_pool_segments=256,
):
    from src.features import extract_features

    games = generate_bot_mouse_games(
        mouse_dfs,
        human_feat_df,
        n_bots=n_bots,
        rng_seed=rng_seed,
        round_deltas=round_deltas,
        id_prefix=id_prefix,
        vae_bundle=vae_bundle,
        vae_n_pool_segments=vae_n_pool_segments,
    )

    def _rows(mice, ids, user_id, bot_type):
        rows = []
        for mouse, gid in zip(mice, ids):
            feats = extract_features(mouse)
            if feats is None:
                continue
            feats.update({
                "userId": user_id,
                "gameId": gid,
                "is_bot": 1,
                "bot_type": bot_type,
            })
            rows.append(feats)
        return pd.DataFrame(rows)

    out = {
        "stitch": _rows(games["stitch"], games["stitch_ids"], -1, "stitch"),
        "smooth": _rows(games["smooth"], games["smooth_ids"], -2, "smooth"),
        "bezier": _rows(games["bezier"], games["bezier_ids"], -3, "bezier"),
        "n_segments": games["n_segments"],
        "n_mice": games["n_mice"],
        "target_ms": games["target_ms"],
        "median_events": games["median_events"],
    }
    if "vae" in games:
        out["vae"] = _rows(games["vae"], games["vae_ids"], -4, "vae")
    return out
