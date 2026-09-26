"""Distill the algebraic eddy-viscosity correction into a `.cfdnn` network.

This is NOT a better closure, and is not meant to be one. It trains an MLP on
the incumbent the design note says any network must beat,

    beta = A * (S*)^B,   A = 1.6945, B = -0.2778   (TURB_ALG_BETA_A / _B)

so the whole Python -> .cfdnn -> C pipeline can be checked against a known
answer: feature order (ln S*, ln Re_t, ln nu_t/nu), the sign and meaning of the
output (a multiplier on nu_t, not on k), input normalization folded into the
first layer, and the byte format. If the pipeline is right, the learned closure
reproduces NS_NUT_CORRECTION_S_STAR to within the fit error; if any link is
wrong it does not, and nothing else in the tree would notice.

The network must learn to IGNORE ln Re_t and ln nu_t/nu, which is a useful
property to verify in its own right: a closure trained on real data should be
able to discount an uninformative feature.

Usage (numpy only, no PyTorch; ~10 s):

    python tools/cfdnn/distill_algebraic.py \\
        --out build/beta_s_star_distilled.cfdnn \\
        --c-header tests/nn/cfdnn_python_golden.h
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfdnn  # noqa: E402

ALG_A = 1.6945
ALG_B = -0.2778

# Feature box the network is trained on, in the closure's own units. Wide
# enough to cover the channel runs (S* ~ 0.3..10, Re_t ~ 1..1e4) and the unit
# test's shear field (S* ~ 1, Re_t ~ 1e3, nu_t/nu ~ 90). Outside it the tanh
# layers saturate, so the prediction flattens rather than extrapolating --
# and the solver's [0.1, 10] clamp bounds it regardless.
LN_S_STAR = (-4.0, 5.0)
LN_RE_T = (-2.0, 12.0)
LN_NUT_NU = (-5.0, 10.0)

HIDDEN = 16
MODEL_NAME = "beta-s-star-distilled"


def target(ln_s_star: np.ndarray) -> np.ndarray:
    return ALG_A * np.exp(ALG_B * ln_s_star)


def sample(rng: np.random.Generator, n: int) -> np.ndarray:
    lo = np.array([LN_S_STAR[0], LN_RE_T[0], LN_NUT_NU[0]])
    hi = np.array([LN_S_STAR[1], LN_RE_T[1], LN_NUT_NU[1]])
    return lo + (hi - lo) * rng.random((n, 3))


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train(seed: int, steps: int) -> tuple[list[cfdnn.Dense], np.ndarray, np.ndarray]:
    """3 -> H tanh -> H tanh -> 1 softplus, Adam on relative squared error.

    Softplus on the output keeps beta positive by construction, which is the
    property the solver's lower clamp is a backstop for, not a substitute.
    """
    rng = np.random.default_rng(seed)
    x_train = sample(rng, 8192)
    y_train = target(x_train[:, 0])[:, None]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    xn = (x_train - mean) / std

    def init(nin, nout):
        return rng.normal(0.0, np.sqrt(1.0 / nin), (nout, nin)), np.zeros(nout)

    params = [*init(3, HIDDEN), *init(HIDDEN, HIDDEN), *init(HIDDEN, 1)]
    params[5][:] = np.log(np.expm1(1.0))  # start the output at beta = 1
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    batch = 512
    for step in range(1, steps + 1):
        lr = 3e-3 * 0.5 * (1.0 + np.cos(np.pi * step / steps)) + 1e-5
        idx = rng.integers(0, len(xn), batch)
        x, y = xn[idx], y_train[idx]
        w1, c1, w2, c2, w3, c3 = params

        z1 = x @ w1.T + c1
        h1 = np.tanh(z1)
        z2 = h1 @ w2.T + c2
        h2 = np.tanh(z2)
        z3 = h2 @ w3.T + c3
        out = softplus(z3)

        # loss = mean(((out - y) / y)^2)
        g_out = 2.0 * (out - y) / (y * y) / batch
        g_z3 = g_out * sigmoid(z3)
        g_w3 = g_z3.T @ h2
        g_c3 = g_z3.sum(axis=0)
        g_z2 = (g_z3 @ w3) * (1.0 - h2 * h2)
        g_w2 = g_z2.T @ h1
        g_c2 = g_z2.sum(axis=0)
        g_z1 = (g_z2 @ w2) * (1.0 - h1 * h1)
        g_w1 = g_z1.T @ x
        g_c1 = g_z1.sum(axis=0)

        for i, g in enumerate((g_w1, g_c1, g_w2, g_c2, g_w3, g_c3)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mh = m[i] / (1 - b1 ** step)
            vh = v[i] / (1 - b2 ** step)
            params[i] -= lr * mh / (np.sqrt(vh) + eps)

    w1, c1, w2, c2, w3, c3 = params
    layers = [
        cfdnn.Dense(w1, c1, "tanh"),
        cfdnn.Dense(w2, c2, "tanh"),
        cfdnn.Dense(w3, c3, "softplus"),
    ]
    return layers, mean, std


def probes() -> np.ndarray:
    """Fixed feature vectors the C test evaluates, spanning the training box.

    Includes pairs that differ only in the two features the target ignores, so
    the C test can see the network discounting them.
    """
    rows = []
    for ln_s in (-3.0, -1.0, 0.0, 1.2, 2.5, 4.0):
        rows.append((ln_s, 6.9, 4.5))
    rows.append((0.0, 0.0, -2.0))
    rows.append((0.0, 11.0, 9.0))
    return np.array(rows, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Distill the algebraic nu_t correction into a .cfdnn model.")
    ap.add_argument("--out", required=True, help="output .cfdnn path")
    ap.add_argument("--c-header", help="also write the model as a C header for the test suite")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args()

    layers, mean, std = train(args.seed, args.steps)
    layers = cfdnn.fold_input_normalization(layers, mean, std)

    data = cfdnn.to_bytes(layers, MODEL_NAME)
    reread, name = cfdnn.from_bytes(data)
    assert name == MODEL_NAME

    # Fit quality, measured on the folded, float32-rounded model as written --
    # not on the float64 training weights -- so the number describes the file.
    x_val = sample(np.random.default_rng(args.seed + 1), 20000)
    rel = cfdnn.predict(reread, x_val)[:, 0] / target(x_val[:, 0]) - 1.0
    print(f"relative error over the training box: max {np.abs(rel).max():.4%}, "
          f"rms {np.sqrt(np.mean(rel ** 2)):.4%}")

    with open(args.out, "wb") as fp:
        fp.write(data)
    print(f"wrote {args.out} ({len(data)} bytes)")

    if args.c_header:
        p = probes()
        expect = cfdnn.predict(reread, p)[:, 0]
        text = "#ifndef CFD_TESTS_NN_PYTHON_GOLDEN_H\n#define CFD_TESTS_NN_PYTHON_GOLDEN_H\n\n"
        text += cfdnn.to_c_header(data, "k_python_golden",
                                 "tools/cfdnn/distill_algebraic.py "
                                 f"--seed {args.seed} --steps {args.steps}")
        text += "\n/* Probe features (ln S*, ln Re_t, ln nu_t/nu) and the model's output\n"
        text += " * at each, from cfdnn.predict() in float64 on the stored float32\n"
        text += " * weights. The C kernels accumulate in float32, so compare with a\n"
        text += " * relative tolerance, not for equality. */\n"
        text += f"#define PYTHON_GOLDEN_PROBES {len(p)}\n"
        text += f"#define PYTHON_GOLDEN_MAX_REL_ERR {np.abs(rel).max():.6e}\n\n"
        text += "static const double k_python_golden_probes[PYTHON_GOLDEN_PROBES][3] = {\n"
        # float(): numpy 2 reprs scalars as np.float64(...), which is not C.
        text += "".join("    {" + ", ".join(repr(float(c)) for c in r) + "},\n" for r in p)
        text += "};\n\nstatic const double k_python_golden_expect[PYTHON_GOLDEN_PROBES] = {\n"
        text += "".join(f"    {float(e)!r},\n" for e in expect)
        text += "};\n\n#endif /* CFD_TESTS_NN_PYTHON_GOLDEN_H */\n"
        with open(args.c_header, "w", newline="\n") as fp:
            fp.write(text)
        print(f"wrote {args.c_header}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
