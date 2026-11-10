r"""Does ensembling fix the seed disagreement, and is the architecture acting as the prior? (#84)

    python experiments/ensemble_prior.py --cache-dir CACHE --manifest-dir MANIFESTS

THE OBJECTION THIS ANSWERS. C60 says five self-supervised networks differing only in seed produce
maps that disagree 2.18x more along the unconstrained direction `v` than across it. The first
thing a reader says is "so ensemble them". This measures whether that works, and the answer is
not obvious in advance:

  IF THE SCATTER ALONG v IS SEED-RANDOM, averaging k networks shrinks it like 1/sqrt(k), exactly
      as it shrinks the orthogonal part, and two independent ensembles agree far better than two
      single networks. Ensembling fixes it and the finding is a caution about single runs.
  IF THE SCATTER ALONG v IS SYSTEMATIC -- set by the architecture and the initialisation
      distribution rather than by the particular draw -- averaging does not touch it. Two
      independent ensembles of the same architecture still disagree along v by as much as two
      single networks do, and what the measurements cannot determine is decided by the
      ARCHITECTURE, not merely by the run.

The orthogonal component is the control built into the same measurement: it is well determined by
the data, so it must shrink like 1/sqrt(k) under either hypothesis. A shrink factor that differs
between the two components is the whole result, and it needs no external baseline.

THE SHARPER VERSION, same networks. If the architecture is what decides the undetermined
directions, then CHANGING the architecture should move the answer along v and nowhere else. We
train three widths, average each into an ensemble, and decompose the difference between ensemble
means. A difference concentrated along v says architectural choices act as a prior in exactly the
directions the data cannot see, and only there.

WHAT WOULD KILL IT. Ensemble-vs-ensemble disagreement along v shrinking like 1/sqrt(k), the same
as orthogonal. That would mean the disagreement is sampling noise over training runs, ensembling
removes it, and the practical claim collapses to "do not trust a single run".

EVERY NUMBER HERE IS AN AGGREGATE. Per-exam and per-voxel values are computed and discarded; the
JSON and the log carry cohort summaries only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from physharm.analysis.crlb_table import acquisition                      # noqa: E402
from physharm.analysis.selfsup_arm import acq_features, physics_loss      # noqa: E402
from physharm.analysis.synthetic_arm import (PARAMS, TISSUE, _jacobian,   # noqa: E402
                                             map_estimate)
from physharm.analysis.traveling_arm import _rows_and_signal, load_case   # noqa: E402
from physharm.data.manifests import read_manifests                        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sst", __file__.rsplit("/", 1)[0] + "/selfsup_torch.py")
_sst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sst)

SIGMA, PRIOR_SD, ARM = 0.05, 10.0, "A1"

#: A trained network must beat the network that just outputs the prior by this factor on held-out
#: exams. NOT AN ABSOLUTE CAP. The old constant was 1e5, and a network scoring 867.6 -- against
#: 17.45 and 25.13 for the widths that converged, and a prior baseline in the low thousands --
#: sailed through it and contributed a row to run B's table. A network barely better than
#: the prior has not learned the inverse problem and cannot say anything about where its errors
#: concentrate. The baseline is computed from the same held-out exams, so the bar moves with the
#: data instead of being guessed.
MIN_IMPROVEMENT_OVER_PRIOR = 10.0

#: Saturation above this means the output bound is shaping the estimates rather than only
#: preventing divergence, and the comparison against an unconstrained MAP fit stops being clean.
SATURATION_WARN = 0.01


def sloppy_direction(x, rows, w):
    """Near-null eigenvector of F = J'WJ at the fitted point, per exam."""
    J = _jacobian(x, rows)
    F = np.einsum("mkn,mln->kln", J, J).mean(2) * w
    ev, V = np.linalg.eigh(F)
    return V[:, 0], ev


def _split(d, v):
    """Decompose a (4, n) difference into its component along v and the norm across it.

    Returns (median |v.d|, median ||d - v (v.d)||) over voxels, matching C60's convention so the
    single-seed number here is comparable to the one already reported.
    """
    p = v @ d
    return float(np.median(np.abs(p))), float(np.median(np.linalg.norm(d - np.outer(v, p), axis=0)))


def compare(preds_a, preds_b, v):
    """Disagreement between the MEANS of two groups of predictions, along v and across it."""
    return _split(np.mean(preds_a, axis=0) - np.mean(preds_b, axis=0), v)


def evaluate(nets_by_label, eval_exams, x0, w, base_mean, cross_pairs=()):
    """Every comparison the experiment needs, accumulated per exam and returned as aggregates."""
    labels = list(nets_by_label)
    acc = {c: {"single": ([], []), "ens": ([], [])} for c in labels}
    cross = {f"{a} vs {b}": ([], []) for a, b in cross_pairs}

    for rows, Y, af in eval_exams:
        v, _ = sloppy_direction(map_estimate(Y, rows, base_mean, prior_sd=PRIOR_SD, sigma=SIGMA),
                                rows, w)
        means = {}
        for wd in labels:
            P = [_sst.predict(n, Y, af, sc, x0) for n, sc in nets_by_label[wd]]
            means[wd] = np.mean(P, axis=0)

            # single-seed: every pair, so the k-network ensemble below is compared against the
            # same quantity C60 reported rather than against a differently-defined baseline
            a_, o_ = [], []
            for i in range(len(P)):
                for j in range(i + 1, len(P)):
                    al, orth = _split(P[i] - P[j], v)
                    a_.append(al); o_.append(orth)
            acc[wd]["single"][0].append(np.mean(a_))
            acc[wd]["single"][1].append(np.mean(o_))

            # two disjoint half-ensembles of the same architecture
            h = len(P) // 2
            al, orth = compare(P[:h], P[h:2 * h], v)
            acc[wd]["ens"][0].append(al)
            acc[wd]["ens"][1].append(orth)

        for a, b in cross_pairs:
            al, orth = _split(means[a] - means[b], v)
            cross[f"{a} vs {b}"][0].append(al)
            cross[f"{a} vs {b}"][1].append(orth)

    def summarise(pair):
        A, O = np.array(pair[0]), np.array(pair[1])
        return {"along": float(np.median(A)), "orth": float(np.median(O)),
                "ratio": float(np.median(A) / np.median(O)),
                "share": float(np.mean(A > O)), "n_exams": int(A.size)}

    return ({wd: {k: summarise(p) for k, p in d.items()} for wd, d in acc.items()},
            {k: summarise(p) for k, p in cross.items()})


def prior_loss(ev, x0, w, lam):
    """Held-out loss of the network that ignores its input and returns the prior mean.

    The bar every trained network has to clear, and the reason it is computed rather than
    guessed: it is what "has not learned anything" scores on THIS data.
    """
    n = sum(Y.shape[1] for _, Y, _ in ev)
    return sum(physics_loss(np.repeat(x0, Y.shape[1], axis=1), Y, rows, w, lam, x0) * Y.shape[1]
               for rows, Y, af in ev) / n


def train_config(pool, ev, x0, w, lam, seeds, steps, width, lr, clip, bar):
    """Train `seeds` networks, dropping any that diverge or fail to beat the prior by `bar`."""
    nets, losses, sats, dropped = [], [], [], 0
    for s in range(seeds):
        t_seed = time.perf_counter()
        try:
            net, scale = _sst.train(pool, x0, w, lam, steps=steps, seed=s, width=width,
                                    lr=lr, clip=clip)
        except SystemExit as e:
            print(f"      seed {s} dropped: {e}", flush=True)
            dropped += 1
            continue
        n = sum(Y.shape[1] for _, Y, _ in ev)
        l_eval = sum(physics_loss(_sst.predict(net, Y, af, scale, x0), Y, rows, w, lam, x0)
                     * Y.shape[1] for rows, Y, af in ev) / n
        if not np.isfinite(l_eval) or l_eval > bar:
            print(f"      seed {s} dropped: held-out loss {l_eval:.4g} does not beat the prior "
                  f"baseline by {MIN_IMPROVEMENT_OVER_PRIOR:g}x (bar {bar:.4g})", flush=True)
            dropped += 1
            continue
        # How close the predictions come to the output bound, PER PARAMETER. The bound exists to
        # stop the forward model's two exponential terms from running away; if it is ALSO shaping
        # the estimates then this experiment is partly measuring the box. Run A reported
        # the max over parameters, which said only that something was pinned, and identifying
        # which cost a second cohort run.
        print(f"      seed {s}: held-out loss {l_eval:.4g}, "
              f"{(time.perf_counter() - t_seed)/60:.1f} min", flush=True)
        sats.append(np.max(np.stack([_sst.saturation_on(net, Y, af, scale)
                                     for _, Y, af in ev[:20]]), axis=0))
        nets.append((net, scale))
        losses.append(l_eval)
    return nets, losses, sats, dropped


def load_exams(cache_dir, manifest_dir, limit, n_vox):
    """(rows, log-signal, acquisition features) per usable exam, in manifest order."""
    mans = read_manifests(manifest_dir)
    if limit:
        mans = mans[:limit]
    exams = []
    for m in mans:
        acq, _ = acquisition(m)
        if acq is None:
            continue
        cached = load_case(cache_dir, m.get("case_id"))
        if cached is None:
            continue
        got = _rows_and_signal(cached, acq, arm=ARM, n_vox=n_vox, rng=np.random.default_rng(0))
        if got is None:
            continue
        rows, y = got
        exams.append((rows, y, acq_features(acq).T))
    return exams


def loss_curve(pool, ev, x0, w, lam, base_mean, seeds, width, lr, clip, checkpoints, bar):
    """Ratio and held-out loss at the same step counts, so two training-set sizes can be
    compared AT MATCHED CONVERGENCE rather than at a matched step budget.

    WHY THIS EXISTS. Run C ran n=500 and n=4525 for the same 40k steps and the larger set
    came out WORSE on held-out loss, 35.64 against 18.06. The trainer draws one exam per step, so
    a fixed step budget gives each exam ~80 passes at n=500 and ~9 at n=4525: training-set size
    and optimisation budget move together, and any difference between them measures both at once.
    Reading the ratio off each size's curve at a common loss separates them.

    ONE RUN PER SEED, not one per step budget, which needs a learning-rate schedule that is a
    function of t ALONE. Cosine is built from the total step count, so step k of a long run has a
    different learning-rate history from a run of k steps and predicts something no truncated run
    would; sampling one cosine run would compare networks that do not exist. The tests pin both
    halves of that -- equality under `inv`, inequality under cosine.

    NOT `constant`, which also satisfies the t-only requirement and is not enough: with no
    annealing the network never settles. Run D used it and finished n=500 at held-out losses
    of 325 and 8.2e4 against 18.06 for the same data under cosine.

    THE SEED SET IS FIXED ACROSS CHECKPOINTS. A seed is admitted on its FINAL loss and then used
    at every checkpoint, including early ones where it has not yet cleared the bar. Admitting
    seeds per checkpoint would change the population along the curve, and the curve would then
    describe two things at once -- which is the mistake this function exists to undo.
    """
    marks = sorted(checkpoints)
    per_seed, dropped = [], 0
    for sd in range(seeds):
        snaps = {}

        def grab(net, scale, step, _s=snaps):
            n = sum(Y.shape[1] for _, Y, _ in ev)
            _s[step] = ([_sst.predict(net, Y, af, scale, x0) for _, Y, af in ev],
                        sum(physics_loss(_sst.predict(net, Y, af, scale, x0), Y, rows, w, lam, x0)
                            * Y.shape[1] for rows, Y, af in ev) / n)

        t_seed = time.perf_counter()
        try:
            _sst.train(pool, x0, w, lam, steps=marks[-1], seed=sd, width=width, lr=lr, clip=clip,
                       checkpoints=marks, on_checkpoint=grab,
                       schedule="inv", warmup=2000)
        except SystemExit as e:
            print(f"      seed {sd} dropped: {e}", flush=True)
            dropped += 1
            continue
        losses = {st: l for st, (_, l) in snaps.items()}
        best_step = min(losses, key=losses.get) if losses else None
        best = losses[best_step] if best_step is not None else float("inf")
        # ADMIT ON THE BEST CHECKPOINT, NOT THE LAST. A curve exists because different
        # checkpoints have different losses, so testing the final one throws away seeds whose
        # good part is early. n=500 at 160k steps is 320 passes over each exam and overfits: job
        # 1801666 dropped a seed at a final loss of 1225 whose earlier points may have been fine,
        # and that is the run this rule was written for.
        if not np.isfinite(best) or best > bar:
            print(f"      seed {sd} dropped: best held-out loss {best:.4g} (step {best_step}) "
                  f"does not beat the prior baseline by {MIN_IMPROVEMENT_OVER_PRIOR:g}x",
                  flush=True)
            dropped += 1
            continue
        # ELAPSED TIME PER SEED, so a run measures its own throughput instead of relying on a
        # proxy. The first timing probe predicted 0.21 h for a 160k-step network at these
        # settings and run E took 6h50m, a factor of 32 that nothing in the run's own
        # output could have revealed.
        dt = time.perf_counter() - t_seed
        print(f"      seed {sd}: best {best:.4g} at step {best_step}, "
              f"final {losses[marks[-1]]:.4g} at step {marks[-1]}; "
              f"{dt/60:.1f} min, {marks[-1]/max(dt, 1e-9):.1f} steps/s"
              + ("  <-- overfits with training" if losses[marks[-1]] > 2 * best else ""),
              flush=True)
        per_seed.append(snaps)

    if len(per_seed) < 2:
        return [], dropped

    vs = [sloppy_direction(map_estimate(Y, rows, base_mean, prior_sd=PRIOR_SD, sigma=SIGMA),
                           rows, w)[0] for rows, Y, af in ev]
    out = []
    for step in marks:
        A, O = [], []
        for e_i in range(len(ev)):
            a_, o_ = [], []
            for i in range(len(per_seed)):
                for j in range(i + 1, len(per_seed)):
                    al, orth = _split(per_seed[i][step][0][e_i] - per_seed[j][step][0][e_i],
                                      vs[e_i])
                    a_.append(al); o_.append(orth)
            A.append(np.mean(a_)); O.append(np.mean(o_))
        losses = [ps[step][1] for ps in per_seed]
        out.append({"step": step, "loss": float(np.median(losses)),
                    "along": float(np.median(A)), "orth": float(np.median(O)),
                    "ratio": float(np.median(A) / np.median(O)),
                    "share": float(np.mean(np.array(A) > np.array(O))),
                    "n_seeds": len(per_seed)})
    return out, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--seeds", type=int, default=6,
                    help="networks per width. Must be EVEN: the ensemble test splits them into "
                         "two disjoint halves, and an odd count would compare ensembles of "
                         "different size, which changes the shrink factor being measured.")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="exams to read; 0 means all")
    ap.add_argument("--n-vox", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--widths", type=int, nargs="*", default=[64, 256, 512],
                    help="architectures for the ensemble and cross-width comparisons. Pass it "
                         "with no values to skip them, which is what a run that only wants the "
                         "loss curves should do -- the width sweep costs six networks per "
                         "architecture and answers a different question.")
    ap.add_argument("--size-sweep", type=int, nargs="*", default=[],
                    help="training-set sizes, all trained at --size-width. Does more data reduce "
                         "the concentration along v? If it does not, the disagreement is "
                         "unidentifiability rather than estimation error -- the distinction the "
                         "paper lists as open. The full pool is already covered by the "
                         "--size-width entry of the width sweep, so pass only smaller sizes.")
    ap.add_argument("--size-width", type=int, default=256)
    ap.add_argument("--loss-curve", type=int, nargs="*", default=[],
                    help="training-set sizes to run as a loss-versus-step CURVE, at "
                         "--size-width; 0 means the full pool. Each size is trained once per "
                         "seed to the largest --checkpoint and sampled along the way, so the "
                         "ratio can be read off at a loss the other size also reaches. This is "
                         "the only way to separate training-set size from optimisation budget: "
                         "a fixed step count gives each exam ~80 passes at n=500 and ~9 at "
                         "n=4525.")
    ap.add_argument("--checkpoints", type=int, nargs="*",
                    default=[20000, 40000, 80000, 160000, 240000],
                    help="step counts at which to sample each --loss-curve run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.seeds % 2:
        raise SystemExit("--seeds must be even; see its help text")

    w, lam = 1.0 / SIGMA ** 2, 1.0 / PRIOR_SD ** 2
    base_mean = {k: v[0] for k, v in TISSUE.items()}
    x0 = np.array([[np.log(base_mean[k])] for k in PARAMS])

    exams = load_exams(args.cache_dir, args.manifest_dir, args.limit, args.n_vox)
    if len(exams) <= args.n_eval:
        raise SystemExit(f"only {len(exams)} usable exams; need more than --n-eval")
    _sst.check_forward_model(exams[0][0])

    ev, pool = exams[:args.n_eval], exams[args.n_eval:]
    # (label, width, n_train). The width sweep trains on everything; the size sweep holds the
    # width fixed and varies the data. The size sweep's full-pool point IS the width sweep's
    # entry for that width, so it is not trained twice.
    configs = [(f"width {wd}", wd, len(pool)) for wd in args.widths]
    configs += [(f"n={n}", args.size_width, n) for n in args.size_sweep if n <= len(pool)]
    print(f"{len(exams)} exams: {len(pool)} for training, {len(ev)} held out. "
          f"{args.seeds} seeds x {len(configs)} configurations = "
          f"{args.seeds * len(configs)} networks, {args.steps} steps each", flush=True)

    base = prior_loss(ev, x0, w, lam)
    bar = base / MIN_IMPROVEMENT_OVER_PRIOR
    print(f"the network that just returns the prior scores {base:.4g} on the held-out exams, "
          f"so a trained one must come in under {bar:.4g}", flush=True)

    nets_by_width, meta = {}, {}
    for label, wd, n_train in configs:
        print(f"\n  {label}" + (f"  (width {wd}, {n_train} exams)"
                                 if label.startswith("n=") else ""), flush=True)
        nets, losses, sats, dropped = train_config(pool[:n_train], ev, x0, w, lam, args.seeds,
                                                   args.steps, wd, args.lr, args.clip, bar)
        if len(nets) < 4 or len(nets) % 2:
            # An odd survivor count would split into unequal halves; drop the last to keep the
            # ensemble comparison symmetric rather than quietly changing what is measured.
            if len(nets) % 2 and len(nets) >= 4:
                print(f"      dropping seed {len(nets)-1} to keep the halves equal", flush=True)
                nets, losses = nets[:-1], losses[:-1]
            else:
                print(f"      {label} ABANDONED: {len(nets)} usable seeds", flush=True)
                continue
        sat = np.max(np.stack(sats), axis=0) if sats else np.full(4, np.nan)
        nets_by_width[label] = nets
        meta[label] = {"width": wd, "n_train": n_train, "n_seeds": len(nets),
                       "dropped": dropped, "loss_median": float(np.median(losses)),
                       "saturation": dict(zip(_sst.PARAM_NAMES, map(float, sat)))}
        print(f"      {len(nets)}/{args.seeds} seeds, held-out loss {np.median(losses):.4g}, "
              f"saturation " + " ".join(f"{p} {v:.4f}" for p, v in
                                        zip(_sst.PARAM_NAMES, sat)), flush=True)
        hot = [p for p, v in zip(_sst.PARAM_NAMES, sat) if v > SATURATION_WARN]
        if hot:
            print(f"      WARNING: saturation exceeds {SATURATION_WARN:g} for {', '.join(hot)}. "
                  f"The output bound is shaping the estimates, not only preventing divergence, "
                  f"and these numbers are not clean.", flush=True)

    report_curves = {}
    if args.loss_curve:
        print("\n=== RATIO AGAINST CONVERGENCE, NOT AGAINST STEP COUNT ===", flush=True)
        print("  Each size trained once per seed and sampled along the way. Compare the blocks", flush=True)
        print("  at a COMMON HELD-OUT LOSS, not at a common step: a fixed step budget gives each", flush=True)
        print("  exam ~80 passes at n=500 and ~9 at n=4525, so matching steps confounds", flush=True)
        print("  training-set size with how converged the network is.", flush=True)
        for n in args.loss_curve:
            n_train = len(pool) if n == 0 else min(n, len(pool))
            print(f"\n  n={n_train}", flush=True)
            curve, dropped = loss_curve(pool[:n_train], ev, x0, w, lam, base_mean, args.seeds,
                                        args.size_width, args.lr, args.clip, args.checkpoints,
                                        bar)
            report_curves[str(n_train)] = {"dropped": dropped, "points": curve}
            if not curve:
                print("      ABANDONED: fewer than two usable seeds", flush=True)
                continue
            print(f"      {'step':>8} {'loss':>9} {'along v':>9} {'orth':>9} {'ratio':>7} "
                  f"{'share':>7} {'seeds':>6}", flush=True)
            for c in curve:
                print(f"      {c['step']:>8} {c['loss']:>9.4g} {c['along']:>9.4f} "
                      f"{c['orth']:>9.4f} {c['ratio']:>7.2f} {100*c['share']:>6.0f}% "
                      f"{c['n_seeds']:>6}", flush=True)

    if not nets_by_width:
        if report_curves:
            # the curve answers a different question and does not need the ensemble comparison;
            # a width sweep that abandons must not take it down with it
            print("\nno width produced a usable ensemble; loss curves reported above", flush=True)
            if args.out:
                with open(args.out, "w") as fh:
                    json.dump({"loss_curves": report_curves, "prior_baseline": float(base),
                               "bar": float(bar), "seeds": args.seeds}, fh, indent=1)
                print(f"wrote {args.out} (aggregates only)", flush=True)
            return
        raise SystemExit("no width produced a usable ensemble")

    width_labels = [c for c in nets_by_width if c.startswith("width ")]
    pairs = [(a, b) for i, a in enumerate(width_labels) for b in width_labels[i + 1:]]
    per_width, cross = evaluate(nets_by_width, ev, x0, w, base_mean, cross_pairs=pairs)

    print("\n=== DOES ENSEMBLING FIX IT? ===", flush=True)
    print("  single = mean over pairs of individual networks (the C60 measurement)", flush=True)
    print("  ens    = two disjoint half-ensembles of the same configuration", flush=True)
    print("  shrink = ens / single. Averaging k independent draws shrinks a RANDOM component by")
    print("           1/sqrt(k); a systematic one does not shrink at all. The `rand` column is")
    print("           that value FOR THIS ROW -- a configuration that lost seeds has smaller")
    print("           halves and a different expectation, and one number for the whole table")
    print("           was wrong for two of run C's three rows.", flush=True)
    print("  The ORTHOGONAL column is the control: the data determines those directions, so they")
    print("  must shrink at the random rate whatever happens along v.\n", flush=True)
    hdr = (f"  {'config':>10} {'loss':>9} {'rand':>6} {'along:single':>13} {'ens':>8} "
           f"{'shrink':>7}   {'orth:single':>12} {'ens':>8} {'shrink':>7}   {'ratio s/e':>10}")

    def row(label):
        a, e = per_width[label]["single"], per_width[label]["ens"]
        rand = 1.0 / np.sqrt(max(meta[label]["n_seeds"] // 2, 1))
        print(f"  {label:>10} {meta[label]['loss_median']:>9.4g} {rand:>6.3f} "
              f"{a['along']:>13.4f} {e['along']:>8.4f} {e['along']/a['along']:>7.3f}   "
              f"{a['orth']:>12.4f} {e['orth']:>8.4f} {e['orth']/a['orth']:>7.3f}   "
              f"{a['ratio']:>4.2f}/{e['ratio']:<5.2f}", flush=True)

    print(hdr, flush=True)
    for label in width_labels:
        row(label)

    size_labels = [c for c in nets_by_width if c.startswith("n=")]
    if size_labels:
        print("\n=== DOES MORE DATA FIX IT? ===", flush=True)
        print(f"  All at width {args.size_width}. A concentration that does not fall as the", flush=True)
        print("  training set grows is unidentifiability rather than estimation error, which is")
        print("  the distinction the paper lists as open. Read the `ratio` column down the page.")
        print("  The full-pool point is the width row above, not repeated here.\n", flush=True)
        print(hdr, flush=True)
        for label in sorted(size_labels, key=lambda c: meta[c]["n_train"]):
            row(label)
        full = f"width {args.size_width}"
        if full in per_width:
            row(full)

    if cross:
        print("\n=== IS THE ARCHITECTURE THE PRIOR? ===", flush=True)
        print("  Difference between the ENSEMBLE MEANS of two architectures, decomposed. Compare", flush=True)
        print("  its magnitude against the `ens` column above, which is how far two ensembles of", flush=True)
        print("  the SAME architecture sit apart: that is the seed-level baseline this has to", flush=True)
        print("  beat before architecture can be said to decide anything.\n", flush=True)
        print(f"  {'pair':>26} {'along v':>9} {'orth':>9} {'ratio':>7} {'share':>7}", flush=True)
        for k_, r in cross.items():
            print(f"  {k_:>26} {r['along']:>9.4f} {r['orth']:>9.4f} {r['ratio']:>7.2f} "
                  f"{100*r['share']:>6.0f}%", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"meta": {str(k_): v for k_, v in meta.items()},
                       "per_width": {str(k_): v for k_, v in per_width.items()},
                       "cross_width": cross, "n_eval": len(ev), "n_pool": len(pool),
                       "prior_baseline": float(base), "bar": float(bar),
                       "loss_curves": report_curves,
                       "seeds": args.seeds, "steps": args.steps,
                       "bound": _sst.DEVIATION_BOUND,
                       "bounded": {_sst.PARAM_NAMES[k]: v for k, v in _sst.BOUNDED.items()}},
                      fh, indent=1)
        print(f"\nwrote {args.out} (aggregates only)", flush=True)


if __name__ == "__main__":
    main()
