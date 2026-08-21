from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.bots import _resolve_dt_samples, _sample_dt_ms, stitch_bot_game
from src.config import (
    ARTIFACTS_VAE_WEIGHTS_DIR,
    PROJECT_ROOT,
    RNG_SEED,
    TARGET_DURATION_MS,
    VAE_BATCH_SIZE,
    VAE_BETA,
    VAE_EPOCHS,
    VAE_HIDDEN,
    VAE_LR,
    VAE_POOL_SEGMENTS,
    VAE_SEG_LEN,
    VAE_Z_DIM,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_RE_WEIGHTS = ARTIFACTS_VAE_WEIGHTS_DIR / "vae_re_v2.pt"
DEFAULT_LOL_WEIGHTS = ARTIFACTS_VAE_WEIGHTS_DIR / "vae_lol_v4.pt"
DEFAULT_CSGO_WEIGHTS = ARTIFACTS_VAE_WEIGHTS_DIR / "vae_csgo_v2.pt"
NORM_AXIS_STD_V1 = "axis_std_v1"


class MouseSegVAE(nn.Module):
    def __init__(self, seg_len=VAE_SEG_LEN, z_dim=VAE_Z_DIM, hidden=VAE_HIDDEN):
        super().__init__()
        self.seg_len = int(seg_len)
        self.z_dim = int(z_dim)
        in_dim = self.seg_len * 2
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden // 2, self.z_dim)
        self.fc_logvar = nn.Linear(hidden // 2, self.z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def _vae_loss(recon, x, mu, logvar, beta=VAE_BETA):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + float(beta) * kl, recon_loss.detach(), kl.detach()


def collect_fixed_length_segments(mouse_dfs, seg_len=VAE_SEG_LEN):
    seg_len = int(seg_len)
    chunks = []
    for mouse in mouse_dfs:
        if mouse is None or len(mouse) < seg_len:
            continue
        df = mouse.sort_values("time")
        dx = df["dx"].to_numpy(dtype=np.float64)
        dy = df["dy"].to_numpy(dtype=np.float64)
        for start in range(0, len(dx) - seg_len + 1, seg_len):
            chunks.append(np.stack([dx[start : start + seg_len], dy[start : start + seg_len]], axis=1))
    if not chunks:
        raise ValueError("collect_fixed_length_segments: no segments found")
    return np.stack(chunks, axis=0)


def _axis_scale_from_segments(segments):
    """Per-axis std of (N, T, 2) dx/dy — avoids MSE being dominated by larger axis."""
    arr = np.asarray(segments, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"_axis_scale_from_segments: expected (N,T,2), got {arr.shape}")
    sx = float(np.std(arr[..., 0]))
    sy = float(np.std(arr[..., 1]))
    sx = sx if np.isfinite(sx) and sx > 1e-6 else 1e-6
    sy = sy if np.isfinite(sy) and sy > 1e-6 else 1e-6
    return np.array([sx, sy], dtype=np.float64)


def _normalize_segments(segments, step_median=None, axis_scale=None):
    """Normalize segments. Prefer per-axis ``axis_scale``; else scalar ``step_median``."""
    arr = np.asarray(segments, dtype=np.float64)
    if axis_scale is not None:
        scale = np.asarray(axis_scale, dtype=np.float64).reshape(2)
        if scale.shape != (2,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"invalid axis_scale: {axis_scale}")
        return (arr / scale.reshape(1, 1, 2)).astype(np.float32), scale
    scale = float(step_median)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid step_median: {step_median}")
    return (arr / scale).astype(np.float32), scale


def _denormalize_recon(recon, bundle):
    """recon: (N, T, 2) in model space → pixel deltas."""
    if bundle.get("norm") == NORM_AXIS_STD_V1 and bundle.get("axis_scale") is not None:
        scale = np.asarray(bundle["axis_scale"], dtype=np.float64).reshape(1, 1, 2)
        return recon * scale
    return recon * float(bundle["step_median"])


def train_mouse_vae(
    mouse_dfs,
    step_median=None,
    *,
    seg_len=VAE_SEG_LEN,
    z_dim=VAE_Z_DIM,
    hidden=VAE_HIDDEN,
    epochs=VAE_EPOCHS,
    batch_size=VAE_BATCH_SIZE,
    lr=VAE_LR,
    beta=VAE_BETA,
    seed=RNG_SEED,
):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cpu")

    raw = collect_fixed_length_segments(mouse_dfs, seg_len=seg_len)
    axis_scale = _axis_scale_from_segments(raw)
    data = _normalize_segments(raw, axis_scale=axis_scale)
    norm_tag = NORM_AXIS_STD_V1
    step_med_log = float(np.sqrt(axis_scale[0] * axis_scale[1]))
    print(
        f"VAE train: n_segments={len(data)} seg_len={seg_len} "
        f"norm={norm_tag} axis_scale=({axis_scale[0]:.4f},{axis_scale[1]:.4f}) "
        f"epochs={epochs} beta={beta}"
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.reshape(len(data), -1))),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=len(data) >= int(batch_size),
    )
    model = MouseSegVAE(seg_len=seg_len, z_dim=z_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    history = []
    model.train()
    for epoch in range(int(epochs)):
        tot = recon_tot = kl_tot = 0.0
        n_batch = 0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, recon_l, kl_l = _vae_loss(recon, xb, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            recon_tot += float(recon_l)
            kl_tot += float(kl_l)
            n_batch += 1
        row = {
            "epoch": epoch + 1,
            "loss": tot / max(n_batch, 1),
            "recon": recon_tot / max(n_batch, 1),
            "kl": kl_tot / max(n_batch, 1),
        }
        history.append(row)
        if ((epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == int(epochs)):
            print(
                f"  epoch {row['epoch']:3d}/{epochs}: "
                f"loss={row['loss']:.5f} recon={row['recon']:.5f} kl={row['kl']:.5f}"
            )

    bundle = {
        "model_state": model.state_dict(),
        "seg_len": int(seg_len),
        "z_dim": int(z_dim),
        "hidden": int(hidden),
        "step_median": float(step_med_log),
        "norm": norm_tag,
        "axis_scale": None if axis_scale is None else [float(axis_scale[0]), float(axis_scale[1])],
        "beta": float(beta),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "n_segments": int(len(data)),
        "history": history,
    }
    return bundle


def save_vae_bundle(bundle, path=DEFAULT_RE_WEIGHTS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in bundle.items() if k != "_model_cache"}
    torch.save(payload, path)
    return path


def load_vae_bundle(path=DEFAULT_RE_WEIGHTS, map_location="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"VAE weights not found: {path}")
    bundle = torch.load(path, map_location=map_location, weights_only=False)
    required = ("model_state", "seg_len", "z_dim", "hidden")
    for key in required:
        if key not in bundle:
            raise ValueError(f"VAE bundle missing key: {key}")
    if bundle.get("axis_scale") is None and bundle.get("step_median") is None:
        raise ValueError("VAE bundle needs axis_scale or step_median for denorm")
    return bundle


def vae_from_bundle(bundle, device=None):
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    cache = bundle.get("_model_cache")
    if cache is not None and cache.get("device") == device:
        return cache["model"]
    model = MouseSegVAE(
        seg_len=bundle["seg_len"],
        z_dim=bundle["z_dim"],
        hidden=bundle["hidden"],
    ).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    bundle["_model_cache"] = {"device": device, "model": model}
    return model


def ensure_vae_bundle(
    mouse_dfs,
    step_median,
    path,
    force_retrain=False,
    **train_kwargs,
):
    path = Path(path)
    if path.is_file() and not force_retrain:
        bundle = load_vae_bundle(path)
        print(f"Loaded VAE weights: {path} (n_segments={bundle.get('n_segments')})")
        return bundle
    bundle = train_mouse_vae(mouse_dfs, step_median, **train_kwargs)
    save_vae_bundle(bundle, path)
    print(f"Saved VAE weights: {path}")
    return bundle


def _rel_time_from_dts(n_events, dt_samples, rng):
    if n_events < 1:
        return np.zeros(0, dtype=float)
    if n_events == 1:
        return np.zeros(1, dtype=float)
    dts = np.array(
        [_sample_dt_ms(rng, dt_samples) for _ in range(n_events - 1)],
        dtype=float,
    )
    rel = np.zeros(n_events, dtype=float)
    rel[1:] = np.cumsum(dts)
    return rel


@torch.no_grad()
def sample_vae_segments(
    bundle,
    n_segments,
    dt_samples=None,
    dt_by_session=None,
    rng=None,
    seed=None,
    device=None,
):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED if seed is None else seed)
    if device is None:
        device = torch.device("cpu")

    model = vae_from_bundle(bundle, device=device)
    seg_len = int(bundle["seg_len"])
    z_dim = int(bundle["z_dim"])
    dt_pool = _resolve_dt_samples(rng, dt_samples=dt_samples, dt_by_session=dt_by_session)

    z = torch.randn(int(n_segments), z_dim, device=device)
    recon = model.decode(z).cpu().numpy().reshape(int(n_segments), seg_len, 2)
    recon = _denormalize_recon(recon, bundle)

    segments = []
    for i in range(int(n_segments)):
        dx = recon[i, :, 0]
        dy = recon[i, :, 1]
        rel_time = _rel_time_from_dts(seg_len, dt_pool, rng)
        segments.append(
            pd.DataFrame({
                "dx": dx.astype(float),
                "dy": dy.astype(float),
                "rel_time": rel_time,
            })
        )
    return segments


def generate_vae_bot_games(
    bundle,
    n_games,
    dt_samples=None,
    dt_by_session=None,
    target_duration_ms=TARGET_DURATION_MS,
    n_pool_segments=VAE_POOL_SEGMENTS,
    rng=None,
    seed=None,
    device=None,
    segment_pool=None,
):
    n_games = int(n_games)
    if n_games < 1:
        raise ValueError("generate_vae_bot_games: n_games < 1")
    if rng is None:
        rng = np.random.default_rng(RNG_SEED if seed is None else seed)
    if segment_pool is None:
        segment_pool = sample_vae_segments(
            bundle,
            n_segments=n_pool_segments,
            dt_samples=dt_samples,
            dt_by_session=dt_by_session,
            rng=rng,
            device=device,
        )
    return [
        stitch_bot_game(
            segment_pool,
            dt_samples=dt_samples,
            dt_by_session=dt_by_session,
            target_duration_ms=target_duration_ms,
            rng=rng,
        )
        for _ in range(n_games)
    ]
