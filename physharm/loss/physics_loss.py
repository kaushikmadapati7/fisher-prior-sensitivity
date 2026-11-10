"""Stage-1 reconstruction-through-physics loss + ADC anchor.

    L = sum_c w_c * || m (.) (S_c - f_c(theta; p_c, g_c)) ||_1
        + lambda_adc * || m (.) (D_hat - ADC) ||_1

Target contrasts are passed in (mode-driven: {T2W,DWI,DCE_pre} for full,
{T2W,DWI} for bi-parametric). Acquisition params differ per case, so per-contrast
rendering is done sample-by-sample over the batch. Returns
``(total, components_dict)`` mirroring ``CustomLoss``.

Set ``roi_weight=1`` and ``lambda_adc=0`` for the single-case overfit gate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from physharm.modes import apply_measured
from physharm.physics.decoders import render_contrast

_EPS = 1e-8


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              roi_weight: float) -> torch.Tensor:
    """ROI-reweighted mean L1 (w = 1 + (roi-1)*mask)."""
    w = 1.0 + (roi_weight - 1.0) * mask
    return (w * (pred - target).abs()).sum() / (w.sum() + _EPS)


class PhysicsReconLoss(nn.Module):
    def __init__(self, target_contrasts, weights: dict | None = None,
                 lambda_adc: float = 1.0, roi_weight: float = 10.0,
                 measured: dict | None = None, per_case_gain: bool = False,
                 prior_refs: dict | None = None, lambda_prior: float = 0.0,
                 prior_weights: dict | None = None):
        super().__init__()
        self.target_contrasts = list(target_contrasts)
        self.w = weights or {c: 1.0 for c in self.target_contrasts}
        self.lambda_adc = lambda_adc
        self.roi_weight = roi_weight
        self.measured = dict(measured or {})
        # weak physiological anchor: relative-L1 pulling estimated maps toward
        # physiological centers (e.g. T1~1300, T2~90 ms). Breaks the T1-gain rail
        # degeneracy (freeing T1 lets it rail short so the recovery term saturates
        # into the gain, and T2 rails high) WITHOUT dictating values -- the recon
        # still moves them. Only applies to params present + not measured/fixed.
        self.prior_refs = {k: float(v) for k, v in (prior_refs or {}).items()}
        self.lambda_prior = lambda_prior
        # relative weight per parameter (multiplies lambda_prior). Lets the anchor
        # be tight where physiology is tight (PD) and loose/off where the map is
        # what we actually want to measure (T2). Default 1.0 = previous behaviour.
        self.prior_weights = {k: float(v) for k, v in (prior_weights or {}).items()}
        # per_case_gain: solve the optimal per-scan intensity scale in closed form
        # (variable projection) instead of a learned global gain. This absorbs
        # g*PD*(1-e^{-TR/T1}) per image, so the wrong-fixed-T1 recovery term (and
        # its TR/vendor dependence) drops out and T2 is set by the TE-ratio alone.
        self.per_case_gain = per_case_gain

    def forward(self, theta: dict, gains, batch: dict, bias=None):
        targets, adc, mask, params = (batch["targets"], batch["adc"],
                                      batch["mask"], batch["params"])
        # D <- measured ADC (etc.) when the mode declares it; leaves theta as-is
        # otherwise. Done before rendering so the DWI decoder sees the fixed D.
        theta = apply_measured(theta, self.measured, batch)
        bsz = adc.shape[0]
        comps = {}
        total = adc.new_zeros(())

        one = adc.new_ones(())
        for jc, c in enumerate(self.target_contrasts):
            l_c = adc.new_zeros(())
            for bi in range(bsz):
                p = params[bi][c]
                th = {k: theta[k][bi:bi + 1] for k in theta}
                t, m = targets[c][bi:bi + 1], mask[bi:bi + 1]
                if self.per_case_gain:
                    # closed-form ROI-optimal scale g* = <t,r>_m / <r,r>_m
                    r = render_contrast(p["render"], th, p, one)
                    g = (m * t * r).sum() / ((m * r * r).sum() + _EPS)
                    pred = g.clamp(1e-3, 1e3) * r
                else:
                    pred = render_contrast(p["render"], th, p, gains[c])
                if bias is not None:                 # per-voxel B1/receive field
                    # a single shared field (1 channel) cancels in the T2W/DWI
                    # ratio -> keeps T2 identifiable; per-contrast would free it
                    bc = 0 if bias.shape[1] == 1 else jc
                    pred = pred * bias[bi:bi + 1, bc:bc + 1]
                l_c = l_c + masked_l1(pred, t, m, self.roi_weight)
            l_c = l_c / bsz
            comps[c] = l_c.detach().item()
            total = total + self.w.get(c, 1.0) * l_c

        # ADC anchor only when D is a *free* estimated parameter. If D is measured
        # (D <- adc), the anchor is identically zero and redundant, so skip it.
        if self.lambda_adc > 0 and "D" not in self.measured:
            l_adc = masked_l1(theta["D"], adc, mask, self.roi_weight)
            comps["adc"] = l_adc.detach().item()
            total = total + self.lambda_adc * l_adc

        # Physiological anchor on estimated maps. PER-PARAM weights matter: at
        # dTE~0 the data only sees the product PD*exp(-TE/T2), so SOME anchor is
        # needed -- but which one is a physics choice. Soft-tissue PD spans ~+-13%
        # (water content .65-.85) while T2 spans 5x (40-200 ms), so anchoring PD
        # hard and leaving T2 free lets T2 be *inferred* from the measured signal
        # (T2 = -TE/ln(A/(g*PD))) instead of being read back out of its own prior.
        if self.lambda_prior > 0 and self.prior_refs:
            l_prior = adc.new_zeros(())
            for pname, ref in self.prior_refs.items():
                if pname in self.measured or pname not in theta:
                    continue                       # measured/fixed: no free gradient
                w = self.prior_weights.get(pname, 1.0)
                if w == 0.0:
                    continue
                dev = (theta[pname] - ref) / ref    # scale-free relative deviation
                l_prior = l_prior + w * (mask * dev.abs()).sum() / (mask.sum() + _EPS)
            comps["prior"] = l_prior.detach().item()
            total = total + self.lambda_prior * l_prior

        comps["total"] = total.detach().item()
        return total, comps
