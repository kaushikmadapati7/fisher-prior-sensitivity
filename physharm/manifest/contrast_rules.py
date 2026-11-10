"""Series-classification rules and per-vendor default acquisition parameters.

This is *data*, not logic: the regexes/guards that map a DICOM SeriesDescription
to a contrast label, and the fallback acquisition parameters keyed by vendor.

The per-vendor defaults were read directly from the Bao raw-DICOM samples
(Siemens VIBE / GE LAVA / Philips mDIXON, plus the matching T2-TSE and DWI
series). They are the v1 training path because, in the sample, processed cases
and raw-DICOM patients have **no name overlap**, so per-case linkage is
impossible (see :mod:`physharm.manifest.build_manifest`).

Units follow :mod:`physharm.physics.decoders`: ms for TR/TE, degrees for FA,
s/mm^2 for b-values.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Series classification
# ---------------------------------------------------------------------------
# Ordered (label, description-regex) rules; first match wins. Numeric guards are
# applied in :func:`classify_series`.
_RULES = [
    ("REJECT", r"localizer|loc\b|scout|mpr|screen\s*save|survey|report|dat$"),
    ("ADC", r"adc|apparent\s*diffusion"),
    ("DWI", r"dwi|diff|resolve|\bdw\b"),
    ("DCE", r"vibe.*dyn|lava|mdixon.*dyn|dixon-w\s*dyn|dynamic|ffe.*dyn|twist"),
    ("T2W", r"t2.*tse|t2w|t2_|tse.*t2|\bt2\b"),
]


def classify_series(desc: str, tr=None, te=None, fa=None, has_bvalue=False) -> str | None:
    """Classify a series into one of {T2W, DWI, ADC, DCE, None} from header fields."""
    d = (desc or "").lower()
    for label, pat in _RULES:
        if re.search(pat, d):
            if label == "REJECT":
                return None
            if label == "ADC":
                return "ADC"
            if label == "DWI":
                # require a non-zero b somewhere (tag or description)
                return "DWI" if (has_bvalue or re.search(r"b\s*=?\s*[1-9]", d)) else None
            if label == "DCE":
                # spoiled GRE dynamic: small flip, short TR
                if fa is not None and fa > 20:
                    continue
                if tr is not None and tr >= 15:
                    continue
                return "DCE"
            if label == "T2W":
                if te is not None and not (40 <= te <= 200):
                    continue
                if tr is not None and not (1500 <= tr <= 12000):
                    continue
                return "T2W"
    return None


def parse_bvalues_from_desc(desc: str) -> list[float]:
    """Best-effort b-value parse from a description like 'b100_1500' or 'B=1500'."""
    d = (desc or "").lower()
    nums = re.findall(r"b\s*=?\s*(\d{2,4})", d)
    # also catch 'b100_1500' style joined ranges
    nums += re.findall(r"(\d{2,4})", re.sub(r"[a-z=]", " ", d)) if "b" in d else []
    vals = sorted({float(n) for n in nums if 10 <= float(n) <= 4000})
    return vals


def decode_ge_bvalue(raw) -> float | None:
    """GE private tag (0043,1039) packs b as e.g. 1000001500 -> 1500."""
    try:
        v = int(raw[0]) if isinstance(raw, (list, tuple)) else int(raw)
    except (TypeError, ValueError):
        return None
    return float(v % 100000)


# ---------------------------------------------------------------------------
# Per-vendor fallback acquisition parameters (read from the Bao DICOM samples)
# ---------------------------------------------------------------------------
def _contrast_set(t2w, dwi, dce):
    return {
        "T2W": {"render": "tse_t2", "TR_ms": t2w[0], "TE_ms": t2w[1],
                "source": "vendor_default"},
        "DWI": {"render": "se_epi", "TR_ms": dwi[0], "TE_ms": dwi[1],
                "b_values": dwi[2], "source": "vendor_default"},
        "ADC": {"render": "anchor", "is_anchor": True, "source": "vendor_default"},
        "DCE_pre": {"render": "spgr", "TR_ms": dce[0], "TE_ms": dce[1],
                    "FA_deg": dce[2], "source": "vendor_default"},
    }


VENDOR_DEFAULTS = {
    # vendor: (T2W: TR,TE), (DWI: TR,TE,[b]), (DCE: TR,TE,FA)
    "GE": _contrast_set((3540.0, 111.8), (5000.0, 78.0, [0.0, 1500.0]),
                        (4.6, 2.15, 12.0)),
    "Siemens": _contrast_set((7300.0, 135.0), (6540.0, 63.0, [100.0, 1500.0]),
                             (3.22, 1.18, 9.0)),
    "Philips": _contrast_set((4800.0, 80.0), (2215.0, 79.0, [100.0, 1200.0]),
                             (4.26, 1.0, 10.0)),  # Philips DCE TE=0 in header -> use ~1ms
}

# Best-known center -> vendor mapping (editable). zhongyiyuan is GE LAVA-Flex;
# the others are assigned their modal vendor (approximate -- some centers are
# multi-vendor). Correct per-case at full scale via real DICOM linkage.
CENTER_VENDOR = {
    "zhongyiyuan": "GE",
    "changshu": "Siemens",
    "jiulong": "Siemens",
    "fuyiyuan": "Siemens",
    "taizhou": "Siemens",
    "zhangjiagang": "Siemens",
}

# Any center not in CENTER_VENDOR falls back to this (with a warning) so the
# full-cohort manifest build never crashes on an unexpected center.
DEFAULT_VENDOR = "Siemens"


# Fixed vendor -> class-index map for the adversarial invariance head. Order is
# stable so checkpoints/labels stay consistent across runs. "unknown" is last.
VENDOR_TO_IDX = {"GE": 0, "Siemens": 1, "Philips": 2, "UIH": 3,
                 "Canon": 4, "Hitachi": 5, "unknown": 6}
N_VENDORS = len(VENDOR_TO_IDX)


def vendor_to_idx(raw: str) -> int:
    """Canonical-vendor class index for a raw DICOM Manufacturer string."""
    return VENDOR_TO_IDX.get(canonical_vendor(raw), VENDOR_TO_IDX["unknown"])


def canonical_vendor(raw: str) -> str:
    """Map a DICOM Manufacturer string to a canonical vendor label.

    DICOM Manufacturer is inconsistent ('GE MEDICAL SYSTEMS', 'SIEMENS',
    'Siemens', 'Philips Medical Systems', 'UIH', ...). Phase-B invariance groups
    by scanner, so collapse to one label per vendor.
    """
    s = (raw or "").strip().upper()
    if not s:
        return "unknown"
    if s.startswith("GE") or "GENERAL ELECTRIC" in s:
        return "GE"
    if "SIEMENS" in s:
        return "Siemens"
    if "PHILIPS" in s:
        return "Philips"
    if s.startswith("UIH") or "UNITED IMAGING" in s:
        return "UIH"
    if "TOSHIBA" in s or "CANON" in s:
        return "Canon"
    if "HITACHI" in s:
        return "Hitachi"
    return raw.strip()  # keep original if unrecognized
