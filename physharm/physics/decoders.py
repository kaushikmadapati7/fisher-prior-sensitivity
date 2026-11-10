"""Differentiable MR signal-equation forward models -- the physics decoder.

These render a measured contrast from the shared physical parameter maps
``theta = {PD, T1, T2, D}`` using a scan's acquisition parameters. They are the
*fixed* (non-learnable) decoder half of Plan A's physics-informed autoencoder:
the only learnable physics-side quantity is a per-contrast receive gain ``g_c``
(:class:`ContrastGains`), which absorbs the unknown arbitrary intensity scale of
clinical MR so that PD can live in ``[0, 1]``.

UNITS CONVENTION (validate with :func:`assert_units`):

    TR, TE, TI, T1, T2 : milliseconds (ms)
    b-value            : s / mm^2
    D                  : mm^2 / s          (so ``b * D`` is dimensionless)
    flip angle (FA)    : degrees           (converted to radians internally)
    PD                 : arbitrary, ~[0, 1]   (absolute scale absorbed by g_c)

``theta`` channels are tensors of shape ``(B, 1, D, H, W)`` (anything
broadcastable works). Acquisition parameters are python floats or 0-d tensors.

Note on the Philips ``TE=0`` header gotcha: ``exp(-TE/T2)`` at ``TE=0`` is just
``1`` and never NaNs as long as ``T2 > 0`` (guaranteed by the model's output
activations), so no special clamping is required -- we keep a small ``_EPS`` in
denominators only for standalone/test use with arbitrary inputs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

_EPS = 1e-6


def _t(x, ref: torch.Tensor) -> torch.Tensor:
    """Coerce a scalar / 0-d param to a tensor on ``ref``'s device and dtype."""
    if torch.is_tensor(x):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(float(x), device=ref.device, dtype=ref.dtype)


# ---------------------------------------------------------------------------
# Signal equations
# ---------------------------------------------------------------------------
def render_tse_t2(PD, T1, T2, g, TR, TE):
    """Turbo/Fast spin-echo (T2-weighted) signal.

    ``S = g * PD * (1 - exp(-TR/T1)) * exp(-TE/T2)``
    """
    TR_, TE_ = _t(TR, PD), _t(TE, PD)
    recovery = 1.0 - torch.exp(-TR_ / (T1 + _EPS))
    decay = torch.exp(-TE_ / (T2 + _EPS))
    return g * PD * recovery * decay


def render_se_epi(PD, T1, T2, D, g, TR, TE, b):
    """Single-shot SE-EPI diffusion-weighted signal at a given b-value.

    ``S(b) = [g * PD * (1 - exp(-TR/T1)) * exp(-TE/T2)] * exp(-b * D)``
    """
    b_ = _t(b, PD)
    s0 = render_tse_t2(PD, T1, T2, g, TR, TE)
    return s0 * torch.exp(-b_ * D)


def render_spgr(PD, T1, g, TR, TE, T2, FA):
    """Spoiled gradient-echo (SPGR / VIBE / LAVA / FFE) steady-state signal.

    ``S = g * PD * sin(a) * (1 - E1) / (1 - cos(a) E1) * exp(-TE/T2)``,
    with ``E1 = exp(-TR/T1)`` and ``a = FA`` in radians.
    """
    TR_, TE_, FA_ = _t(TR, PD), _t(TE, PD), _t(FA, PD)
    a = FA_ * math.pi / 180.0
    E1 = torch.exp(-TR_ / (T1 + _EPS))
    sin_a, cos_a = torch.sin(a), torch.cos(a)
    spgr = sin_a * (1.0 - E1) / (1.0 - cos_a * E1 + _EPS)
    decay = torch.exp(-TE_ / (T2 + _EPS))
    return g * PD * spgr * decay


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
# Keyed by the manifest "render" field. The raw functions have heterogeneous
# signatures; :func:`render_contrast` marshals theta + params into each.
DECODERS = {
    "tse_t2": render_tse_t2,
    "se_epi": render_se_epi,
    "spgr": render_spgr,
}


def render_contrast(render_kind: str, theta: dict, params: dict, g, b=None):
    """Render one contrast from ``theta`` and a manifest ``params`` dict.

    ``theta``  : {"PD","T1","T2","D"} tensors.
    ``params`` : {"TR_ms","TE_ms"[,"FA_deg"][,"b_values"]} from the manifest.
    ``g``      : scalar gain tensor for this contrast.
    ``b``      : explicit b-value for ``se_epi`` (defaults to max manifest b).
    """
    PD, T1, T2, D = theta["PD"], theta["T1"], theta["T2"], theta["D"]
    if render_kind == "tse_t2":
        return render_tse_t2(PD, T1, T2, g, params["TR_ms"], params["TE_ms"])
    if render_kind == "spgr":
        return render_spgr(PD, T1, g, params["TR_ms"], params["TE_ms"], T2,
                           params["FA_deg"])
    if render_kind == "se_epi":
        if b is None:
            b = max(params.get("b_values") or [0.0])   # `or` also guards an empty list
        return render_se_epi(PD, T1, T2, D, g, params["TR_ms"], params["TE_ms"], b)
    raise ValueError(f"unknown render kind: {render_kind!r}")


# ---------------------------------------------------------------------------
# Learnable per-contrast receive gain
# ---------------------------------------------------------------------------
class ContrastGains(nn.Module):
    """Per-contrast learnable receive gain ``g_c > 0``.

    Parameterized in log-space so the gain stays strictly positive. Initialized
    to 1.0 (log-gain 0). v1 uses a single global scalar per contrast; a smooth
    per-voxel bias field ``B(x)`` is a deferred extension.
    """

    def __init__(self, contrasts):
        super().__init__()
        self.log_g = nn.ParameterDict(
            {c: nn.Parameter(torch.zeros(())) for c in contrasts}
        )

    def __getitem__(self, contrast: str) -> torch.Tensor:
        return torch.exp(self.log_g[contrast])

    def forward(self, contrast: str) -> torch.Tensor:  # convenience alias
        return self[contrast]


# ---------------------------------------------------------------------------
# Unit validation
# ---------------------------------------------------------------------------
def assert_units(params: dict, name: str = "", D_typical=None) -> None:
    """Sanity-check a manifest contrast params dict against the unit convention.

    ``D_typical`` is optional but should be passed wherever a diffusion map is in scope:
    it is the ONLY check on D, and a factor-10^6 error in D is silent everywhere else.
    See :mod:`physharm.data.units`.
    """
    tr = params.get("TR_ms")
    if tr is not None:
        assert tr > 0, f"{name}: TR_ms must be > 0, got {tr}"
    te = params.get("TE_ms")
    if te is not None:
        assert te >= 0, f"{name}: TE_ms must be >= 0, got {te}"
    fa = params.get("FA_deg")
    if fa is not None:
        assert 0 < fa <= 180, f"{name}: FA_deg must be in (0, 180], got {fa}"
    for b in params.get("b_values", []) or []:
        assert b >= 0, f"{name}: b-value must be >= 0, got {b}"
    if D_typical is not None:
        # b*D must be dimensionless. the site stores ADC in 10^-6 mm^2/s; unconverted it makes
        # every diffusion term wrong by 10^6, and nothing downstream would notice.
        from physharm.data.units import assert_diffusion_units
        assert_diffusion_units(D_typical, name=f"{name}: D" if name else "D")
