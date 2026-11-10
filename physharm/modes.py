"""Contrast/parameter modes -- the bi-parametric lever.

A mode declares which physical parameters the model *estimates*, which are held
*fixed* (folded into the gain), which are taken from a *measured* map, which
contrasts feed the encoder, and which are render targets for the reconstruction
loss.

- ``full``            : {PD,T1,T2,D} from T2W+DWI+ADC+DCE_pre; renders T2W,DWI,DCE_pre.
                        The original DCE pipeline.
- ``biparametric``    : {PD,T2,D} from T2W+DWI+ADC; renders T2W,DWI. D is a *free*
                        estimated parameter anchored toward ADC. Kept for
                        comparison -- free D is degenerate with the DWI gain, so it
                        needs the ADC anchor and the anchor fights DWI recon.
- ``biparametric_dadc`` : {PD,T2} from T2W+DWI+ADC; D is *set directly* to the
                        measured ADC (not estimated), renders T2W,DWI, no anchor.
                        This removes the D/gain degeneracy: D is the one channel we
                        actually measure, so we clamp it instead of inverting for it.
                        T2W and DWI have different TEs, so keeping both as targets
                        is what separately identifies PD and T2.

- ``triparametric_t1w`` : {PD,T1,T2} from T1W+T2W+DWI+ADC; renders T1W,T2W,DWI. D is
                        the measured ADC. The pre-contrast T1w spin echo (short TE,
                        short TR) is the second-TE anchor that makes T2 identifiable
                        for every vendor (incl Philips), and its short TR makes T1
                        estimable -- so T1 is UN-fixed here.

T1 is fixed at 1200 ms in the bi-parametric modes: with the long T2W/DWI TR the
recovery term ``1 - exp(-TR/T1)`` is ~constant, so it folds into the per-contrast
gain and T1 carries no usable signal *from those contrasts alone*. The
``triparametric_t1w`` mode escapes this by adding a short-TR T1w spin echo, whose
``1 - exp(-TR/T1)`` term varies with T1 -> T1 becomes estimable (and unfixed).

``measured`` maps a parameter name -> a batch key ("adc"); the value is a
per-voxel measured map that replaces the estimated/fixed parameter at render
time (see :func:`apply_measured`).
"""
from __future__ import annotations

MODES = {
    "full": {
        "params": ["PD", "T1", "T2", "D"],
        "fixed": {},
        "measured": {},
        "inputs": ["T2W", "DWI", "ADC", "DCE_pre"],
        "targets": ["T2W", "DWI", "DCE_pre"],
    },
    "biparametric": {
        "params": ["PD", "T2", "D"],
        "fixed": {"T1": 1200.0},
        "measured": {},
        "inputs": ["T2W", "DWI", "ADC"],
        "targets": ["T2W", "DWI"],
    },
    "biparametric_dadc": {
        "params": ["PD", "T2"],
        "fixed": {"T1": 1200.0},
        "measured": {"D": "adc"},
        "inputs": ["T2W", "DWI", "ADC"],
        "targets": ["T2W", "DWI"],
    },
    # Multi-b: adds the recovered low-b band as an "S0" contrast (se_epi at b~0,
    # so the diffusion term is 1 -> a diffusion-FREE T2-weighted image at the DWI
    # TE). It is a clean second-TE anchor for T2 -- the current DWI target is
    # diffusion-attenuated at high b, which corrupts its T2 constraint; S0 is not.
    # D is still the measured ADC. Requires manifests with an S0 contrast (built
    # from the multi-b re-registration).
    "biparametric_s0": {
        "params": ["PD", "T2"],
        "fixed": {"T1": 1200.0},
        "measured": {"D": "adc"},
        "inputs": ["T2W", "DWI", "S0", "ADC"],
        "targets": ["T2W", "DWI", "S0"],
    },
    # S0 as INPUT ONLY (dropped from targets vs biparametric_s0): gives the encoder
    # the clean diffusion-free second-TE anchor to identify T2, without a
    # reconstruction target forcing it to reproduce S0's vendor texture. Tests
    # whether an identifiable (physiological) T2 can be made less vendor-separable
    # than when S0 is reconstructed. Same contrast set as biparametric_s0 -> reuses
    # the bip_s0 cache. (High-b DWI alone does NOT identify T2 -> it rails; S0 is
    # what anchors it.)
    "biparametric_s0_in": {
        "params": ["PD", "T2"],
        "fixed": {"T1": 1200.0},
        "measured": {"D": "adc"},
        "inputs": ["T2W", "DWI", "S0", "ADC"],
        "targets": ["T2W", "DWI"],
    },
    # Ablation of biparametric_dadc: DWI dropped as a render target (kept as an
    # encoder input so features are identical). Tests whether DWI's different-TE
    # term is what identifies T2, or whether T2 stays physiological from T2W +
    # priors alone. Same contrast set -> reuses the biparametric cache.
    "biparametric_dadc_t2wonly": {
        "params": ["PD", "T2"],
        "fixed": {"T1": 1200.0},
        "measured": {"D": "adc"},
        "inputs": ["T2W", "DWI", "ADC"],
        "targets": ["T2W"],
    },
    # T1w upgrade: a PRE-contrast T1w spin echo (TE~8-13 ms, short TR) is present in
    # 100% of cases across GE/Siemens/Philips/UIH. Paired with T2W (TE 80-215) it
    # gives a big SAME-family dTE -> T2 is identifiable for EVERY vendor, INCLUDING
    # Philips (where the T2W-vs-DWI dTE~1 ms fails). And the three TRs (T1W ~500 /
    # DWI ~2200 / T2W ~4000) make the 1-exp(-TR/T1) term informative -> T1 is now
    # ESTIMATED, not fixed. D stays the measured ADC. T1W renders through the SAME
    # render_tse_t2 as T2W (different TR/TE) -- no new physics. Requires manifests
    # with a T1W contrast (add_t1w reg + add_t1w_manifest); the ~2/925 cases lacking
    # a pre-contrast T1w are auto-dropped by the dataset's need.issubset check.
    "triparametric_t1w": {
        "params": ["PD", "T1", "T2"],
        "fixed": {},
        "measured": {"D": "adc"},
        "inputs": ["T1W", "T2W", "DWI", "ADC"],
        "targets": ["T1W", "T2W", "DWI"],
    },
}


def get_mode(name: str) -> dict:
    if name not in MODES:
        raise ValueError(f"unknown mode {name!r}; choose from {list(MODES)}")
    return MODES[name]


def apply_measured(theta: dict, measured: dict, batch: dict) -> dict:
    """Augment ``theta`` with measured-source params (e.g. ``D <- adc``).

    Returns a new dict so the estimated maps are left untouched; a no-op (returns
    ``theta`` unchanged) when ``measured`` is empty. The measured map is taken from
    ``batch[src]`` and must be broadcastable to the theta channel shape
    ``(B, 1, D, H, W)`` -- ``batch["adc"]`` already is.
    """
    if not measured:
        return theta
    out = dict(theta)
    for pname, src in measured.items():
        out[pname] = batch[src]
    return out
