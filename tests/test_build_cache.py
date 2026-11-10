"""Tests for the cache builder (issue #5).

Run with pytest, or standalone:  python tests/test_build_cache.py

Everything up to touching an image is tested here. The image path itself needs nibabel and
real volumes and must be exercised on the cluster -- `--dry-run` exists partly so that
boundary is a supported mode rather than an untested one.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

from physharm.data.build_cache import cohort_id, plan, read_manifests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_cohort(md, n=200):
    """Manifests with repeats, a gated-out subset, and two scanner models."""
    ids = []
    for i in range(n):
        model = "SIGNA Premier" if i % 2 else "MAGNETOM Vida"
        # every 10th case is wrong-sequence T2W; every 15th has a non-diffusion b band
        tr = 900.0 if i % 10 == 0 else 6000.0
        b_hi = 50.0 if i % 15 == 0 else 1000.0
        for j in range(2 if i % 4 == 0 else 1):
            cid = f"c{i:04d}_{j}"
            ids.append(cid)
            m = {"case_id": cid, "patient_id": f"p{i:04d}", "scanner_model": model,
                 "b_hi": b_hi,
                 "contrasts": {"T2W": {"TR_ms": tr, "TE_ms": 150.0},
                               "DWI": {"TR_ms": 4000.0, "TE_ms": 70.0,
                                       "b_values": [b_hi]},
                               "ADC": {"render": "anchor"}}}
            json.dump(m, open(os.path.join(md, f"{cid}.json"), "w"))
    return ids


def test_default_norm_scheme_preserves_the_inter_contrast_ratio():
    """The default must be `shared`, and this is not a style preference.

    `ref_median` normalizes each contrast independently to an ROI median of 1.0, so the
    T2W/DWI ratio is identically 1 for every case -- and that ratio is exactly where T2 lives
    (S ~ exp(-TE/T2)). T2 is then determined by the learned gains rather than by the data.
    The archived notes record the symptom: prostate T2 pinned at ~33 ms, IDENTICAL ACROSS
    VENDORS -- an artifact indistinguishable from the result this project measures.

    I had originally defaulted to ref_median, which would have produced a clean,
    scanner-invariant T2 for the wrong reason.
    """
    import argparse
    import physharm.data.build_cache as BC
    src = open(BC.__file__).read()
    assert 'ap.add_argument("--norm-scheme", default="shared"' in src, \
        "the default must be `shared`"
    assert "DESTROYS T2 identifiability" in src, \
        "choosing ref_median must warn at runtime, not only in a docstring"


def test_cohort_id_is_stable_and_order_independent():
    a = cohort_id(["c3", "c1", "c2"])
    assert a == cohort_id(["c1", "c2", "c3"])
    assert a != cohort_id(["c1", "c2"])
    assert len(a) == 16


def test_plan_composes_gate_then_split():
    with tempfile.TemporaryDirectory() as d:
        _write_cohort(d)
        sp, rep = plan(d)
        ex = rep["exclusions"]
        assert ex["n_kept"] < ex["n_in"], "the planted bad cases must be gated"
        assert any("different sequence" in r for r in ex["reason_counts"])
        assert any("not diffusion-weighted" in r for r in ex["reason_counts"])
        # split is over the KEPT cases only
        assert sum(len(v) for v in sp.values()) == ex["n_kept"]


def test_gated_cases_never_reach_the_cache_plan():
    with tempfile.TemporaryDirectory() as d:
        _write_cohort(d)
        sp, _ = plan(d)
        cached = {m["case_id"] for v in sp.values() for m in v}
        for m in read_manifests(d):
            tr = m["contrasts"]["T2W"]["TR_ms"]
            if tr < 2000 or m["b_hi"] < 500:
                assert m["case_id"] not in cached, f"{m['case_id']} should be gated"


def test_traveling_subjects_are_planned_but_never_in_train():
    with tempfile.TemporaryDirectory() as d:
        _write_cohort(d)
        tj = os.path.join(d, "trav.json")
        json.dump({"usable": {"cross-model": [f"p{i:04d}" for i in range(1, 40, 2)]}},
                  open(tj, "w"))
        sp, rep = plan(d, traveling=tj)
        trav = {m["patient_id"] for m in sp["traveling"]}
        assert trav, "traveling set must be populated"
        for k in ("train", "val", "test"):
            assert not ({m["patient_id"] for m in sp[k]} & trav)
        # they ARE cached — evaluated, just never trained on
        assert rep["splits"]["exams"]["traveling"] > 0


def test_dry_run_reads_no_images_and_writes_nothing():
    """The image path needs nibabel, which the laptop lacks; --dry-run must not import it."""
    with tempfile.TemporaryDirectory() as d:
        md = os.path.join(d, "m")
        os.makedirs(md)
        _write_cohort(md, n=60)
        cache = os.path.join(d, "cache")
        r = subprocess.run(
            [sys.executable, "-m", "physharm.data.build_cache",
             "--manifest-dir", md, "--image-root", "/nonexistent",
             "--cache-dir", cache, "--dry-run"],
            capture_output=True, text=True, cwd=REPO,
            env=dict(os.environ, PYTHONPATH=REPO))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "DRY RUN" in r.stdout and "cohort_id" in r.stdout
        assert not os.path.exists(os.path.join(cache, "_cohort.json"))
        assert not any(f.endswith(".npz") for f in os.listdir(cache)) if os.path.isdir(cache) else True


def test_dry_run_warns_when_traveling_is_omitted():
    """Silently training on the gold-standard control is the worst available mistake."""
    with tempfile.TemporaryDirectory() as d:
        md = os.path.join(d, "m")
        os.makedirs(md)
        _write_cohort(md, n=40)
        r = subprocess.run(
            [sys.executable, "-m", "physharm.data.build_cache",
             "--manifest-dir", md, "--image-root", "/nonexistent",
             "--cache-dir", os.path.join(d, "c"), "--dry-run"],
            capture_output=True, text=True, cwd=REPO,
            env=dict(os.environ, PYTHONPATH=REPO))
        assert "WARNING" in r.stdout and "gold-standard control" in r.stdout


def test_shard_partitions_without_overlap_or_loss():
    with tempfile.TemporaryDirectory() as d:
        md = os.path.join(d, "m")
        os.makedirs(md)
        _write_cohort(md, n=120)
        sp, _ = plan(md)
        ids = sorted(m["case_id"] for v in sp.values() for m in v)
        k = 5
        shards = [[c for i, c in enumerate(ids) if i % k == n] for n in range(k)]
        flat = [c for s in shards for c in s]
        assert sorted(flat) == ids and len(set(flat)) == len(ids)


def test_plan_is_deterministic_in_the_seed():
    with tempfile.TemporaryDirectory() as d:
        _write_cohort(d)
        a, _ = plan(d, seed=3)
        b, _ = plan(d, seed=3)
        c, _ = plan(d, seed=4)
        ida = [m["case_id"] for m in a["test"]]
        assert ida == [m["case_id"] for m in b["test"]]
        assert ida != [m["case_id"] for m in c["test"]]


if __name__ == "__main__":
    import traceback
    npass = nfail = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            try:
                f(); npass += 1; print(f"  PASS {n}")
            except Exception:
                nfail += 1; print(f"  FAIL {n}"); traceback.print_exc(limit=2)
    print(f"\n{npass} passed, {nfail} failed")
    raise SystemExit(1 if nfail else 0)
