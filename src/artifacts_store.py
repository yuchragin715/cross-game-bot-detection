from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import pandas as pd

from src.config import ARTIFACTS_CLASSIFIERS_DIR, ARTIFACTS_DATA_DIR

_VALID_SPLITS = {"human", "bot"}
_VALID_GAMES = {"re", "lol", "csgo"}
_VALID_KINDS = {"traces", "features"}
_VALID_FEATURE_SETS = {"raw", "si_min", "si_ext"}
_VALID_BOT_TYPES = {"stitch", "smooth", "bezier", "vae"}


def ensure_artifact_dirs() -> None:
    ARTIFACTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)


# --- Data CSV cache (artifacts/data/) ---


def _data_path(split: str, game: str, kind: str) -> Path:
    split_key = split.strip().lower()
    game_key = game.strip().lower()
    kind_key = kind.strip().lower()
    if split_key not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
    if game_key not in _VALID_GAMES:
        raise ValueError(f"game must be one of {_VALID_GAMES}, got {game!r}")
    if kind_key not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    return ARTIFACTS_DATA_DIR / f"{game_key}_{split_key}_{kind_key}.csv"


def save_data_csv(df: pd.DataFrame, split: str, game: str, kind: str) -> Path:
    ensure_artifact_dirs()
    path = _data_path(split=split, game=game, kind=kind)
    df.to_csv(path, index=False)
    return path


def load_data_csv(split: str, game: str, kind: str) -> pd.DataFrame:
    path = _data_path(split=split, game=game, kind=kind)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_or_build_data(
    split: str,
    game: str,
    kind: str,
    build_fn: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    path = _data_path(split=split, game=game, kind=kind)
    if path.is_file():
        return pd.read_csv(path)
    df = build_fn()
    save_data_csv(df, split=split, game=game, kind=kind)
    return df


def traces_to_long_df(
    traces: Iterable[pd.DataFrame],
    session_ids: Iterable[str],
    split: str,
    game: str,
    bot_types: Iterable[str] | None = None,
    user_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    rows = []
    traces = list(traces)
    session_ids = list(session_ids)
    if bot_types is None:
        bot_types = ["human"] * len(traces)
    if user_ids is None:
        user_ids = ["-1"] * len(traces)
    bot_types = list(bot_types)
    user_ids = list(user_ids)
    if not (len(traces) == len(session_ids) == len(bot_types) == len(user_ids)):
        raise ValueError("traces/session_ids/bot_types/user_ids length mismatch")
    for sid, uid, bt, trace in zip(session_ids, user_ids, bot_types, traces):
        t = trace.copy().reset_index(drop=True)
        t["event_idx"] = range(len(t))
        t["session_id"] = str(sid)
        t["user_id"] = str(uid)
        t["bot_type"] = str(bt)
        t["split"] = split
        t["source_game"] = game
        rows.append(t[["source_game", "split", "bot_type", "session_id", "user_id", "event_idx", "time", "dx", "dy"]])
    if not rows:
        return pd.DataFrame(columns=["source_game", "split", "bot_type", "session_id", "user_id", "event_idx", "time", "dx", "dy"])
    return pd.concat(rows, ignore_index=True)


def long_df_to_traces(long_df: pd.DataFrame) -> dict[str, list]:
    if long_df.empty:
        return {"traces": [], "session_ids": [], "bot_types": [], "user_ids": []}
    work = long_df.sort_values(["session_id", "event_idx"]).reset_index(drop=True)
    traces, session_ids, bot_types, user_ids = [], [], [], []
    for sid, g in work.groupby("session_id", sort=False):
        traces.append(g[["dx", "dy", "time"]].reset_index(drop=True))
        session_ids.append(str(sid))
        bot_types.append(str(g["bot_type"].iloc[0]))
        user_ids.append(str(g["user_id"].iloc[0]))
    return {
        "traces": traces,
        "session_ids": session_ids,
        "bot_types": bot_types,
        "user_ids": user_ids,
    }


def try_load_game_cache(game: str, split: str) -> dict | None:
    try:
        feat_df = load_data_csv(split=split, game=game, kind="features")
    except FileNotFoundError:
        return None
    try:
        trace_long = load_data_csv(split=split, game=game, kind="traces")
        trace_pack = long_df_to_traces(trace_long)
    except FileNotFoundError:
        trace_pack = {"traces": [], "session_ids": [], "bot_types": [], "user_ids": []}
    return {
        "features": feat_df,
        **trace_pack,
    }


def save_game_cache(
    game: str,
    split: str,
    features_df: pd.DataFrame,
    traces,
    session_ids,
    bot_types=None,
    user_ids=None,
) -> None:
    save_data_csv(features_df, split=split, game=game, kind="features")
    trace_long = traces_to_long_df(
        traces,
        session_ids=session_ids,
        split=split,
        game=game,
        bot_types=bot_types,
        user_ids=user_ids,
    )
    save_data_csv(trace_long, split=split, game=game, kind="traces")


def split_bot_cache(cache: dict, n_preview: int = 5) -> dict[str, dict]:
    feat = cache["features"]
    trace_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for trace, sid, trace_bt in zip(cache["traces"], cache["session_ids"], cache["bot_types"]):
        trace_lookup[(str(trace_bt), str(sid))] = trace
    has_traces = bool(trace_lookup)

    out: dict[str, dict] = {}
    for bt in feat["bot_type"].astype(str).unique():
        bt_feat = feat.loc[feat["bot_type"].astype(str) == str(bt)].reset_index(drop=True)
        if "gameId" not in bt_feat.columns:
            raise ValueError(f"bot cache features missing gameId for bot_type={bt!r}")
        bt_traces = []
        if has_traces:
            for game_id in bt_feat["gameId"].astype(str):
                key = (str(bt), str(game_id))
                if key not in trace_lookup:
                    raise ValueError(f"bot cache missing trace for bot_type={bt!r}, session_id={game_id!r}")
                bt_traces.append(trace_lookup[key])
        out[str(bt)] = {
            "features": bt_feat,
            "traces": bt_traces,
            "samples": [trace.copy() for trace in bt_traces[: int(n_preview)]],
        }
    return out


BOT_NOTEBOOK_VARS = {
    "re": {
        "stitch": ("re_stitch_df", "sample_re_stitch_trajectories"),
        "smooth": ("re_smooth_df", "sample_re_smooth_trajectories"),
        "bezier": ("re_bezier_df", "sample_re_bezier_trajectories"),
        "vae": ("re_vae_df", "sample_vae_trajectories"),
    },
    "lol": {
        "stitch": ("lol_bots_stitch_df", "sample_lol_stitch_trajectories"),
        "smooth": ("lol_bots_smooth_df", "sample_lol_smooth_trajectories"),
        "bezier": ("lol_bots_bezier_df", "sample_lol_bezier_trajectories"),
        "vae": ("lol_bots_vae_df", "sample_lol_vae_trajectories"),
    },
    "csgo": {
        "stitch": ("csgo_bots_stitch_df", "sample_csgo_stitch_trajectories"),
        "smooth": ("csgo_bots_smooth_df", "sample_csgo_smooth_trajectories"),
        "bezier": ("csgo_bots_bezier_df", "sample_csgo_bezier_trajectories"),
        "vae": ("csgo_bots_vae_df", "sample_csgo_vae_trajectories"),
    },
}


def apply_bot_cache(game: str, cache: dict, namespace: dict, n_preview: int = 5) -> None:
    parts = split_bot_cache(cache, n_preview=n_preview)
    mapping = BOT_NOTEBOOK_VARS[game]
    for bt, (feat_name, sample_name) in mapping.items():
        if bt not in parts:
            raise ValueError(f"bot cache for {game} missing bot_type={bt!r}")
        namespace[feat_name] = parts[bt]["features"]
        namespace[sample_name] = parts[bt]["samples"]
    if game == "re":
        namespace["re_vae_traces"] = parts["vae"]["traces"]
    elif game == "lol":
        namespace["lol_vae_traces"] = parts["vae"]["traces"]
    elif game == "csgo":
        namespace["csgo_vae_traces"] = parts["vae"]["traces"]


# --- Classifier cache (artifacts/classifiers/) ---


def classifier_path(train_game: str, feature_set: str, bot_type: str) -> Path:
    game_key = train_game.strip().lower()
    feat_key = feature_set.strip().lower()
    bot_key = bot_type.strip().lower()
    if game_key not in _VALID_GAMES:
        raise ValueError(f"train_game must be one of {_VALID_GAMES}, got {train_game!r}")
    if feat_key not in _VALID_FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {_VALID_FEATURE_SETS}, got {feature_set!r}")
    if bot_key not in _VALID_BOT_TYPES:
        raise ValueError(f"bot_type must be one of {_VALID_BOT_TYPES}, got {bot_type!r}")
    return ARTIFACTS_CLASSIFIERS_DIR / f"{game_key}_{feat_key}_{bot_key}.joblib"


def save_classifier(model: Any, train_game: str, feature_set: str, bot_type: str) -> Path:
    ensure_artifact_dirs()
    path = classifier_path(train_game, feature_set, bot_type)
    joblib.dump(model, path)
    return path


def load_classifier(train_game: str, feature_set: str, bot_type: str) -> Any:
    path = classifier_path(train_game, feature_set, bot_type)
    if not path.is_file():
        raise FileNotFoundError(path)
    return joblib.load(path)


def load_or_train_classifier(
    train_game: str,
    feature_set: str,
    bot_type: str,
    train_fn: Callable[[], tuple[Any, Any]],
) -> tuple[Any, Any]:
    path = classifier_path(train_game, feature_set, bot_type)
    if path.is_file():
        print(f"Loaded classifier cache: {path.name}")
        return joblib.load(path), None
    model, metric = train_fn()
    save_classifier(model, train_game, feature_set, bot_type)
    print(f"Saved classifier cache: {path.name}")
    return model, metric


def train_or_load_bot_detector(
    human_df: pd.DataFrame,
    bot_df: pd.DataFrame,
    feature_cols,
    train_game: str,
    feature_set: str,
    bot_type: str,
    **train_kwargs,
) -> tuple[Any, Any]:
    from src.evaluation import train_bot_detector

    def _train() -> tuple[Any, Any]:
        return train_bot_detector(human_df, bot_df, feature_cols, **train_kwargs)

    return load_or_train_classifier(train_game, feature_set, bot_type, _train)
