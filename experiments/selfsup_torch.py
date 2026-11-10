r"""Self-supervised arm, in torch. (#82)

    python experiments/selfsup_torch.py --cache-dir CACHE --manifest-dir MANIFESTS

WHY THIS EXISTS SEPARATELY FROM physharm. `physharm` depends only on numpy and the test suite
runs anywhere because of it. This needs a real optimiser, so it lives outside the package and is
imported by nothing in it -- the same arrangement as the figure script and matplotlib.

WHY IT EXISTS AT ALL. The numpy version of this arm (physharm/analysis/selfsup_arm.py) is correct
but could not be trained to loss parity with the MAP fit: across 12k, 30k and 60k steps its loss
ran 6.1x, 2.4x, 4.4x the MAP's -- not even monotone -- and the measured displacement swung across
zero. The failure was the hand-rolled Adam, not the design, so the design is kept and the
optimiser replaced. Loss parity is the precondition for the measurement to mean anything: an
under-converged estimator under-responds to ANY change in the objective, so a low ratio would be
optimisation error masquerading as a property of amortisation.

WHAT IT MEASURES. The theory (run log, phases 25-27) locates the failure in the per-exam
optimisation: the correction is signed because the FIT moves along the sloppy direction in
response to misspecification. An amortised estimator never solves that optimisation, so the theory
predicts it does not inherit the signed shortfall. Training is self-supervised, minimising the MAP
objective itself,

    L(x) = w ||y - f(x)||^2 + lambda ||x - x0||^2

which needs no ground-truth parameter maps and so runs on real images. Because the prior is
explicit it can be shifted and the network retrained, which is the intervention used throughout
this project. The network then differs from the MAP fit in exactly one respect: it is amortised.

The forward model is reimplemented in torch so autograd supplies the physics gradient. It is
checked against the numpy model on every run; a silent divergence between the two would invalidate
everything downstream.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from physharm.analysis.synthetic_arm import PARAMS, TISSUE, _model_log, _rows_for  # noqa: E402

SIGMA, PRIOR_SD = 0.05, 10.0


def model_torch(x, rows):
    """log-signal for each measurement row. `x` is (n, 4) in log units; returns (n, m).

    Mirrors physharm.analysis.synthetic_arm._model_log. `rows` entries are either the string
    "anchor" or a (TR, TE, b) triple, matching that module's convention.
    """
    PD, T1, T2, D = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    out = []
    for r in rows:
        if r == "anchor":
            out.append(D)
        else:
            TR, TE, b = r
            E1 = torch.exp(-TR / torch.exp(T1))
            out.append(PD + torch.log(torch.clamp(1.0 - E1, min=1e-12))
                       - TE / torch.exp(T2) - b * torch.exp(D))
    return torch.stack(out, dim=1)


def check_forward_model(rows, seed=0, tol=1e-8):
    """The torch model must agree with the numpy one, or nothing downstream is meaningful."""
    rng = np.random.default_rng(seed)
    x = np.stack([np.full(64, np.log(TISSUE[k][0])) for k in PARAMS]) + 0.3 * rng.normal(size=(4, 64))
    a = _model_log(x, rows).T
    b = model_torch(torch.tensor(x.T, dtype=torch.float64), rows).numpy()
    err = float(np.abs(a - b).max())
    if err > tol:
        raise SystemExit(f"torch and numpy forward models disagree by {err:.3e}")
    return err


#: How far, in log units, a BOUNDED parameter may move against the divergent direction. e^3 is a
#: factor of 20: T2 floored at 4.0 ms, D capped at 2.6e-2. Both are far outside anything the
#: prostate produces, and the worst-case loss they permit is 1.8e6 -- finite, and below the
#: divergence guard's threshold, where the unbounded model reaches 1e38 and beyond.
DEVIATION_BOUND = 3.0

#: Decay exponent for the `inv` schedule: lr(t) = peak * (n_warm / max(t, n_warm)) ** this.
#: Chosen by sweep against cosine on a fixture in the cohort's loss regime, 12k steps, 3 seeds:
#: 0.5 gave a median held-out loss of 6.10, 0.75 gave 3.73 and 1.0 gave 3.63, against cosine's
#: 3.03 and a flat schedule's 19.54. 1.0 matches cosine and had the tightest spread across seeds.
DECAY_POWER = 1.0

#: Which parameters are bounded, and in which direction. ONLY THE TWO THAT CAN ACTUALLY DIVERGE.
#:   T2 (-1): the T2W row carries -TE/exp(T2), which explodes as T2 is driven DOWN, so T2 needs
#:       a floor and nothing else.
#:   D  (+1): the DWI row carries -b*exp(D), which explodes as D is driven UP, so D needs a cap.
#:   PD, T1: absent, deliberately. PD enters the log-signal linearly and T1 enters through
#:       log(1 - exp(-TR/exp(T1))), which is linear in T1 at one extreme and constant at the
#:       other. Neither can run away: the loss is quadratic in a linear term, which is a bowl
#:       that pulls back.
#:
#: BOUNDING PD AND T1 WAS A REAL BUG, NOT A CONSERVATIVE CHOICE. Run A bounded all four at
#: 2.5 and came back with 52% of predictions pinned at the bound, which voided the run. The cache
#: is normalised per case, so the scale becomes an additive constant in log space that PD must
#: absorb -- a quantity with no physiological range at all. Reproducing that offset on a fixture
#: puts saturation at 0.43, of which PD is 0.32 and T1 0.27 while T2 is 0.019 and D 0.001: the
#: bound was binding on exactly the two parameters that never needed it.
BOUNDED = {2: -1.0, 3: +1.0}


class Net(torch.nn.Module):
    """MLP predicting a BOUNDED deviation from a fixed parameterisation point.

    WHY THE BOUND. The forward model is exponential in two of its four parameters and unbounded
    in both: the T2W row carries -TE/exp(T2), which grows without limit as T2 is driven down,
    and the DWI row carries -b*exp(D), which grows without limit as D is driven up. At a
    deviation of -8 log units in T2 the log-signal already reaches 5.7e3, so the loss -- carrying
    a factor w = 1/sigma^2 = 400 -- reaches 1e10, and it keeps going. That is the whole mechanism
    behind the losses of 1e20, 1e72 and 1e117 that three sweeps produced and one of them smuggled
    past a finite-value guard. Gradient clipping bounds the STEP, not the region, so it cannot
    prevent this; nothing did, because nothing stopped the network from emitting physically
    impossible parameters.

    ONE-SIDED, AND ON TWO PARAMETERS ONLY. See BOUNDED. The deviation is the identity on the
    safe side of each bounded parameter and everywhere for the unbounded ones, so the constraint
    applies precisely to the divergence and to nothing else.

    tanh, not clamp: clamp has zero gradient outside its range, so a network that walks out of
    the box can never walk back. tanh saturates smoothly and keeps a gradient everywhere. The
    two branches meet at zero with matching value and slope, so the map is C1 and leaves the
    zero-initialisation exact.

    THE BOUND MUST NOT BIND ON THE ESTIMATES, or it is a constraint on the answer rather than on
    the divergence, and the comparison against the unconstrained MAP fit would be measuring the
    box. `saturation` reports how close the predictions come, PER PARAMETER, because a scalar
    summary is what made run A's failure take a cohort run to discover. It is a fixed
    constant across seeds and widths, so it cannot differentiate the configurations compared.
    """

    def __init__(self, d_in, width=256, depth=4, bound=DEVIATION_BOUND):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]
            d = width
        layers += [torch.nn.Linear(d, 4)]
        self.f = torch.nn.Sequential(*layers)
        self.bound = float(bound)
        # Start at the prior: the last layer is zeroed so an untrained network predicts x0 exactly.
        # Without this the initial T1 is exp(0) = 1 ms against a true 1400 and training diverges.
        # tanh(0) = 0, so the bound leaves this exact.
        torch.nn.init.zeros_(self.f[-1].weight)
        torch.nn.init.zeros_(self.f[-1].bias)

    def _limit(self, dev):
        """Identity except against each bounded parameter's divergent direction."""
        out = [dev[:, k] for k in range(dev.shape[1])]
        for k, sign in BOUNDED.items():
            z = sign * out[k]                       # z > 0 is the dangerous side
            out[k] = sign * torch.where(z <= 0, z, self.bound * torch.tanh(z / self.bound))
        return torch.stack(out, dim=1)

    def forward(self, inp, x0):
        return x0 + self._limit(self.f(inp))

    @torch.no_grad()
    def saturation(self, inp):
        """Per-parameter fraction of outputs within 10% of the bound, on the bounded side.

        Unbounded parameters report 0 by construction. Anything materially above 0 for a bounded
        one means the box is shaping the estimates and the run is not usable.
        """
        dev = self.f(inp)
        out = np.zeros(dev.shape[1])
        for k, sign in BOUNDED.items():
            out[k] = float(((sign * dev[:, k]) / self.bound > 0.9).double().mean())
        return out


def physics_loss_torch(x, y, rows, w, lam, x0):
    r = y - model_torch(x, rows)
    return (w * (r ** 2).sum(1) + lam * ((x - x0) ** 2).sum(1)).mean()


def train(exams, x0, w, lam, steps=8000, seed=0, width=256, lr=1e-3, batch=4096, scale=None,
          verbose=False, offset=None, clip=1.0, warmup=0.10, blowup=1e3,
          bound=DEVIATION_BOUND, checkpoints=(), on_checkpoint=None, schedule="cosine"):
    """`exams` is a list of (rows, Y (m,n), acq_feature_vector). Batches come from ONE exam.

    Sampling across exams would apply one exam's acquisition to another's measurements, which is
    wrong physics for most of the batch.

    `x0` is the prior mean IN THE LOSS. `offset` is the output parameterisation and must be held
    FIXED across the base and shifted runs; if it moves with the prior, the two networks learn
    different functions rather than the same function under different objectives, and the measured
    displacement contains the parameterisation change itself. Defaults to `x0` for a single run.
    The numpy version of this arm had exactly this confound and it produced ratios of -5 and -11.
    """
    if offset is None:
        offset = x0
    # Step counts at which to hand the caller the network mid-training. There is no dropout or
    # batch norm here, so a network evaluated at a checkpoint predicts exactly what it would
    # predict if training had stopped there; the caller can build a loss-versus-step curve from
    # ONE run instead of one run per step budget.
    marks = {int(c) for c in checkpoints if 0 < int(c) <= steps}
    torch.manual_seed(seed)
    dev = "cpu"
    packs = []
    for rows, Y, af in exams:
        packs.append((rows,
                      torch.tensor(Y.T, dtype=torch.float64, device=dev),
                      torch.tensor(af, dtype=torch.float64, device=dev)))
    if scale is None:
        allin = torch.cat([torch.cat([Yt, af.repeat(Yt.shape[0], 1)], 1) for _, Yt, af in packs])
        scale = (allin.mean(0, keepdim=True), allin.std(0, keepdim=True) + 1e-9)
    mu, sd = scale
    net = Net(mu.shape[1], width=width, bound=bound).double().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # Linear warmup then cosine decay. Real residuals are far larger than synthetic ones and the
    # loss carries a factor w = 1/sigma^2 = 400, so the first steps see enormous gradients; on the
    # cohort, three seeds trained and the fourth diverged at step 3760 without this.
    # `warmup` is a FRACTION of the run below 1 and an ABSOLUTE step count at or above it. The
    # absolute form exists for `schedule="constant"`: a fractional warmup still makes the
    # trajectory depend on the total step count, which is exactly what that schedule is there to
    # avoid.
    n_warm = max(1, int(warmup) if warmup >= 1 else int(warmup * steps))
    if schedule == "cosine":
        shape = lambda t: ((t + 1) / n_warm if t < n_warm
                           else 0.5 * (1 + np.cos(np.pi * (t - n_warm) / max(1, steps - n_warm))))
    elif schedule in ("constant", "inv"):
        # CHECKPOINTS ARE ONLY TRUNCATED RUNS UNDER A SCHEDULE THAT IS A FUNCTION OF t ALONE.
        # Cosine decay is built from the total step count, so step k of a long run has seen a
        # different learning-rate history from a run of k steps and predicts something no
        # truncated run would -- which would make a loss-versus-step curve sampled from one run a
        # comparison between networks that do not exist.
        #
        # `constant` satisfies that and nothing else, and it is not enough: with no annealing the
        # network never settles. Run D ran n=500 at a flat 1e-3 for 160k steps and finished
        # at held-out losses of 325 and 8.2e4, against 18.06 for the same data under cosine at
        # 40k. Removing the decay removed work cosine was doing.
        #
        # `inv` is the one to use: linear warmup then a (n_warm/t)**DECAY_POWER decay, which
        # depends only on t and so keeps the checkpoint property, while still annealing -- to
        # 1.25% of peak by 160k with a 2000-step warmup, and to a held-out loss matching cosine's
        # on a fixture. `constant` is kept only because a test pins the contrast between a
        # schedule that anneals and one that does not.
        pw = DECAY_POWER if schedule == "inv" else 0.0
        shape = lambda t: (min(1.0, (t + 1) / n_warm)
                           * (n_warm / max(t + 1, n_warm)) ** pw)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, shape)
    x0t = torch.tensor(x0.T, dtype=torch.float64, device=dev)      # prior mean, in the loss
    offt = torch.tensor(offset.T, dtype=torch.float64, device=dev)  # parameterisation, held fixed
    g = torch.Generator().manual_seed(seed)
    for t in range(steps):
        rows, Yt, af = packs[int(torch.randint(len(packs), (1,), generator=g))]
        idx = torch.randint(Yt.shape[0], (min(batch, Yt.shape[0]),), generator=g)
        yb = Yt[idx]
        inp = (torch.cat([yb, af.repeat(yb.shape[0], 1)], 1) - mu) / sd
        loss = physics_loss_torch(net(inp, offt), yb, rows, w, lam, x0t)
        if t == 0:
            # The loss at initialisation, where the network outputs the prior exactly. Anything
            # far above this is divergence, and the run must not be allowed to finish and report
            # a number. A finite check is NOT enough: a previous cohort run returned losses of
            # 1e117 and 1e208, which are finite, so five diverged networks were silently
            # summarised as a result.
            loss0 = float(loss.detach())
        opt.zero_grad(); loss.backward()
        # The shifted run diverged to NaN at 40k steps without this. Clipping costs nothing on
        # the runs that were already stable and makes the long ones usable.
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step(); sched.step()
        lv = float(loss.detach())
        if not np.isfinite(lv) or (t > n_warm and lv > blowup * max(loss0, 1e-12)):
            raise SystemExit(
                f"training diverged at step {t} (seed {seed}, lr {lr}, clip {clip}): "
                f"loss {lv:.4g} against {loss0:.4g} at initialisation. "
                f"Lower --lr or --clip.")
        if on_checkpoint is not None and (t + 1) in marks:
            on_checkpoint(net, scale, t + 1)
        if verbose and t % max(1, steps // 6) == 0:
            print(f"    step {t:>6}  loss {loss.item():.5g}", flush=True)
    return net, scale


def _inputs(Y, af, scale):
    mu, sd = scale
    Yt = torch.tensor(Y.T, dtype=torch.float64)
    aft = torch.tensor(af, dtype=torch.float64)
    return (torch.cat([Yt, aft.repeat(Yt.shape[0], 1)], 1) - mu) / sd


@torch.no_grad()
def predict(net, Y, af, scale, x0):
    return net(_inputs(Y, af, scale), torch.tensor(x0.T, dtype=torch.float64)).numpy().T


def saturation_on(net, Y, af, scale):
    """How close this exam's predictions come to the output bound; 0 means the bound is inert.

    The bound is there to stop the forward model's two exponential terms from running away. If it
    is also shaping the estimates then any comparison against the unconstrained MAP fit is partly
    measuring the box, so every run that relies on the bound has to report this rather than
    assume it. On fixtures spanning the held-out losses these networks actually reach, it sits
    between 0.0005 and 0.002 for the bounded parameters even under a normalisation offset large
    enough to put PD's saturation at 0.32.

    Returns a length-4 array, one entry per parameter, not a scalar: run A reported the max
    over parameters, which said only that something was pinned and not what, and identifying it
    cost a second cohort run.
    """
    return net.saturation(_inputs(Y, af, scale))


PARAM_NAMES = ("PD", "T1", "T2", "D")
