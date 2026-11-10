"""The map estimator: multi-contrast images -> physical parameter maps theta.

Not a VAE -- the "latent" is the interpretable parameter-map field at full
spatial resolution.

Encoder (``encoder=``):
  "mlp"  (default) -- per-voxel MLP (1x1x1 convs). Recovering theta at a voxel
      from that voxel's contrast values is a pointwise inverse (qMRI fitting), so
      no spatial context is needed. Cannot collapse to a uniform output.
  "unet" -- 3D U-Net; smoothing bias (kept for a future spatial-prior variant).

The model estimates the parameters in ``estimate`` and fills the rest from
``fixed`` (constants), so ``render_contrast`` always sees a full {PD,T1,T2,D}
dict. This is the bi-parametric lever (estimate {PD,T2,D}, fix T1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from physharm.models.unet3d import UNet3D

CHANNELS = ["PD", "T1", "T2", "D"]   # full ordered parameter set

DEFAULT_RANGES = {
    "PD": (0.0, 1.0),
    "T1": (100.0, 3000.0),
    "T2": (10.0, 300.0),
    "D": (0.0, 3.0e-3),
}
DEFAULT_INIT = {"PD": 0.5, "T1": 1200.0, "T2": 80.0, "D": 1.0e-3}


class MapEstimator(nn.Module):
    def __init__(self, in_channels: int, estimate=None, fixed: dict | None = None,
                 base_ch: int = 32, ch_mults=(1, 2, 4, 8), ranges: dict | None = None,
                 init: dict | None = None, head_weight_scale: float = 1.0,
                 attn_resolutions=(), encoder: str = "mlp",
                 mlp_hidden: int = 256, mlp_layers: int = 5, cond_dim: int = 0):
        super().__init__()
        self.estimate = list(estimate) if estimate is not None else list(CHANNELS)
        self.fixed = dict(fixed or {})
        self.ranges = ranges or DEFAULT_RANGES
        self.init = init or DEFAULT_INIT
        self.encoder = encoder
        # acquisition-param conditioning (see physharm.data.acq_cond): TR/TE/b are
        # appended as constant channels so the inverse is protocol-aware.
        self.cond_dim = int(cond_dim)
        n_out = len(self.estimate)
        in_total = in_channels + self.cond_dim

        if encoder == "mlp":
            layers, ch = [], in_total
            for _ in range(mlp_layers - 1):
                layers += [nn.Conv3d(ch, mlp_hidden, 1), nn.SiLU()]
                ch = mlp_hidden
            self.head = nn.Conv3d(ch, n_out, 1)
            layers.append(self.head)
            self.net = nn.Sequential(*layers)
        elif encoder == "unet":
            self.unet = UNet3D(in_channels=in_total, out_channels=n_out,
                               cond_channels=0, num_classes=0, base_ch=base_ch,
                               ch_mults=tuple(ch_mults),
                               attn_resolutions=tuple(attn_resolutions))
            self.head = self.unet.conv_out
            self.register_buffer("t0", torch.zeros(1))
        else:
            raise ValueError(f"unknown encoder {encoder!r} (use 'mlp' or 'unet')")

        self._init_output_head(head_weight_scale)

    def _init_output_head(self, weight_scale: float) -> None:
        """Bias the final layer so initial (estimated) maps sit at physiological centers."""
        with torch.no_grad():
            self.head.weight.mul_(weight_scale)
            for i, name in enumerate(self.estimate):
                lo, hi = self.ranges[name]
                frac = min(max((self.init[name] - lo) / (hi - lo), 1e-4), 1 - 1e-4)
                self.head.bias[i] = math.log(frac / (1 - frac))

    def _activate(self, raw: torch.Tensor) -> dict:
        out = {}
        for i, name in enumerate(self.estimate):
            lo, hi = self.ranges[name]
            out[name] = lo + (hi - lo) * torch.sigmoid(raw[:, i:i + 1])
        # fill fixed (un-estimated) parameters with constants, broadcast in space
        ref = out[self.estimate[0]]
        for name, val in self.fixed.items():
            out[name] = torch.full_like(ref, float(val))
        return out

    def _with_cond(self, x: torch.Tensor, cond) -> torch.Tensor:
        """Append acquisition-param channels (broadcast over space) to the input."""
        if not self.cond_dim:
            return x
        if cond is None:
            raise ValueError(
                f"model was built with cond_dim={self.cond_dim} (acquisition-param "
                "conditioning) but forward() got cond=None -- pass "
                "acq_cond_batch(batch['params'], cond_contrasts(inputs))")
        c = cond.to(device=x.device, dtype=x.dtype)
        if c.shape[0] != x.shape[0] or c.shape[1] != self.cond_dim:
            raise ValueError(f"cond shape {tuple(c.shape)} != (B={x.shape[0]}, "
                             f"cond_dim={self.cond_dim})")
        c = c.view(x.shape[0], self.cond_dim, 1, 1, 1).expand(-1, -1, *x.shape[2:])
        return torch.cat([x, c], dim=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict:
        """x: (B, C_in, D, H, W), cond: (B, cond_dim) -> {PD,T1,T2,D} (estimated + fixed)."""
        x = self._with_cond(x, cond)
        raw = self.net(x) if self.encoder == "mlp" else self.unet(x, self.t0.expand(x.shape[0]))
        return self._activate(raw)
