# Fisher information over-predicts prior sensitivity in quantitative MRI

Code, tests and analysis arms for the ML4PS 2026 workshop paper.

A maximum a posteriori fit resolves what the measurements cannot determine by adding a prior,
so part of every recovered parameter comes from the prior. Fisher information is the standard
way to predict how large that part is. These arms measure it directly, by moving the prior's
centre and refitting, and compare the measurement against the Fisher prediction and against the
exact curvature of the fitting problem.

## Layout

    physharm/physics/      Fisher information and the leak matrix
    physharm/analysis/     the analysis arms: prior shift, synthetic, real, traveling, self-supervised
    physharm/data/         manifests, splits, caching, intensity normalisation, unit guards
    physharm/models/       the MAP estimator and the network used by the self-supervised arm
    experiments/           drivers that produce the artifacts the paper cites
    audit/                 standalone checks on the cohort and the cache
    tests/                 unit tests, including the correct-model noiseless fixture
    paper/                 the paper source and its figure generator

## Running

    PYTHONPATH=. python -m physharm.analysis.prior_shift --help
    PYTHONPATH=. python experiments/cohort_shift_limit.py --help
    PYTHONPATH=. python -m pytest tests/

The clinical arm reads a derived cache built from imaging that cannot be redistributed, so
those drivers take `--cache-dir` and `--manifest-dir` and will not run without them. The
synthetic arm, the unit tests and the closed-form checks run with no data at all, and they
cover the algebra every claim on the clinical cohort rests on.

## Measuring the leak

The quantity is a derivative. Move the prior's nominal value for one parameter by a known
amount, refit the same data, and divide the resulting displacement by the prediction. A
predictor that is correct returns 1. Differencing centrally rather than one-sidedly removes an
O(delta) error that is visible at the shift sizes used here.
