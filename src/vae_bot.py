from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.bots import _resolve_dt_samples, _sample_dt_ms, stitch_bot_game
from src.config import (
    PROJECT_ROOT,
    RNG_SEED,
    TARGET_DURATION_MS,
    VAE_BATCH_SIZE,
    VAE_BETA,
    VAE_EPOCHS,
    VAE_HIDDEN,
    VAE_LR,
    VAE_SEG_LEN,
    VAE_Z_DIM,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_RE_WEIGHTS = ARTIFACTS_DIR / "vae_re_v1.pt"
DEFAULT_LOL_WEIGHTS = ARTIFACTS_DIR / "vae_lol_v1.pt"
DEFAULT_CSGO_WEIGHTS = ARTIFACTS_DIR / "vae_csgo_v1.pt"


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


def collect_fixed_length_segments(mouse_dfs, seg_len=VAE_SEG_LEN, stride=None):
    seg_len = int(seg_len)
    if stride is None:
        stride = seg_len
    stride = int(stride)
    chunks = []
    for mouse in mouse_dfs:
        if mouse is None or len(mouse) < seg_len:
            continue
        df = mouse.sort_values("time")
        dx = df["dx"].to_numpy(dtype=np.float64)
        dy = df["dy"].to_numpy(dtype=np.float64)
        for start in range(0, len(dx) - seg_len + 1, stride):
            chunks.append(np.stack([dx[start : start + seg_len], dy[start : start + seg_len]], axis=1))
    if not chunks:
        raise ValueError("collect_fixed_length_segments: no segments found")
    return np.stack(chunks, axis=0)


def _normalize_segments(segments, step_median):
    scale = float(step_median)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid step_median: {step_median}")
    return (segments / scale).astype(np.float32), scale


def train_mouse_vae(
    mouse_dfs,
    step_median,
    *,
    seg_len=VAE_SEG_LEN,
    z_dim=VAE_Z_DIM,
    hidden=VAE_HIDDEN,
    epochs=VAE_EPOCHS,
    batch_size=VAE_BATCH_SIZE,
    lr=VAE_LR,
    beta=VAE_BETA,
    seed=RNG_SEED,
    device=None,
    stride=None,
    verbose=True,
):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device is None:
        device = torch.device("cpu")

    raw = collect_fixed_length_segments(mouse_dfs, seg_len=seg_len, stride=stride)
    data, scale = _normalize_segments(raw, step_median)
    if verbose:
        print(
            f"VAE train: n_segments={len(data)} seg_len={seg_len} "
            f"step_median={scale:.4f} epochs={epochs} beta={beta} device={device}"
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
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == int(epochs)):
            print(
                f"  epoch {row['epoch']:3d}/{epochs}: "
                f"loss={row['loss']:.5f} recon={row['recon']:.5f} kl={row['kl']:.5f}"
            )

    bundle = {
        "model_state": model.state_dict(),
        "seg_len": int(seg_len),
        "z_dim": int(z_dim),
        "hidden": int(hidden),
        "step_median": float(scale),
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
    torch.save(bundle, path)
    return path


def load_vae_bundle(path=DEFAULT_RE_WEIGHTS, map_location="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"VAE weights not found: {path}")
    bundle = torch.load(path, map_location=map_location, weights_only=False)
    required = ("model_state", "seg_len", "z_dim", "hidden", "step_median")
    for key in required:
        if key not in bundle:
            raise ValueError(f"VAE bundle missing key: {key}")
    return bundle


def vae_from_bundle(bundle, device=None):
    if device is None:
        device = torch.device("cpu")
    model = MouseSegVAE(
        seg_len=bundle["seg_len"],
        z_dim=bundle["z_dim"],
        hidden=bundle["hidden"],
    ).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model


def ensure_vae_bundle(
    mouse_dfs,
    step_median,
    path=DEFAULT_RE_WEIGHTS,
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
    scale = float(bundle["step_median"])
    dt_pool = _resolve_dt_samples(rng, dt_samples=dt_samples, dt_by_session=dt_by_session)

    z = torch.randn(int(n_segments), z_dim, device=device)
    recon = model.decode(z).cpu().numpy().reshape(int(n_segments), seg_len, 2)
    recon = recon * scale

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


def generate_vae_bot_game(
    bundle,
    dt_samples=None,
    dt_by_session=None,
    target_duration_ms=TARGET_DURATION_MS,
    n_pool_segments=256,
    rng=None,
    seed=None,
    device=None,
):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED if seed is None else seed)
    segments = sample_vae_segments(
        bundle,
        n_segments=n_pool_segments,
        dt_samples=dt_samples,
        dt_by_session=dt_by_session,
        rng=rng,
        device=device,
    )
    return stitch_bot_game(
        segments,
        dt_samples=dt_samples,
        dt_by_session=dt_by_session,
        target_duration_ms=target_duration_ms,
        rng=rng,
    )
