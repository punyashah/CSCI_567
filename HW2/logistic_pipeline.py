"""CSCI 567 HW2, Problems 1-3: logistic regression, L2 and L1 regularization.

Fill in every function marked TODO. Allowed packages: numpy, matplotlib only.
Run from this directory:  python logistic_pipeline.py
Do not change the seed, split fractions, step size, iteration count, or the
lambda grid -- your numbers must be comparable to ours.

Data: data/wdbc.csv (Breast Cancer Wisconsin, Diagnostic; UCI ML Repository).
Each row is "label,f1,...,f30" with label 1 = malignant, 0 = benign.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/wdbc.csv"
ETA, ITERS = 0.5, 2000
LAMBDAS = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]

FEATURE_NAMES = [f"{s} {n}" for s in ("mean", "SE", "worst") for n in (
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave points", "symmetry", "fractal dimension")]


def load_and_split(path):
    """Load the csv, shuffle with seed 567, split 60/20/20 train/val/test."""
    raw = np.loadtxt(path, delimiter=",")
    rng = np.random.default_rng(567)
    raw = raw[rng.permutation(len(raw))]
    y, X = raw[:, 0], raw[:, 1:]
    n = len(y)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    return (X[:n_tr], y[:n_tr], X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va],
            X[n_tr + n_va:], y[n_tr + n_va:])


# ============================================================
# Problem 1: logistic regression and the ML pipeline
# ============================================================

def min_max_scale(X_train, *others):
    """Min-max scale each feature to [0,1] using the TRAINING set's min/max,
    then apply the same affine map to every other array passed in.
    If a feature is constant on the training set, leave it unscaled (divide
    by 1 instead of 0). Return (scaled X_train, scaled others...).
    """
    # TODO: implement
    xmin = X_train.min(axis=0)
    xmax = X_train.max(axis=0)
    rng = xmax - xmin
    rng = np.where(rng == 0, 1, rng)
    scaled_train = (X_train - xmin) / rng
    scaled_others = [((X - xmin) / rng) for X in others]
    return (scaled_train, *scaled_others)


def add_bias(X):
    """Return X with a constant-1 column appended as the LAST column (the bias
    fold-in from Lecture 02), so the bias is always the last coordinate of w.
    Every penalty below must skip that coordinate. Call this AFTER scaling."""
    # TODO: implement
    return np.hstack([X, np.ones((X.shape[0], 1))])


def sigmoid(z):
    """Numerically safe sigmoid. Never compute np.exp of a large positive
    number: use 1/(1+e^{-z}) where z >= 0 and e^{z}/(1+e^{z}) where z < 0.
    """
    # TODO: implement
    e = np.exp(-(np.abs(z)))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def logistic_loss(w, X, y):
    """Mean logistic loss with labels y in {0,1}.

    Compute it stably via log(1+e^{a}) = max(a,0) + log1p(e^{-|a|}),
    where a = -z for y=1 and a = z for y=0, z = Xw.
    """
    # TODO: implement
    z = X @ w
    a = np.where(y == 1, -z, z)
    return np.mean(np.maximum(a, 0) + np.log1p(np.exp(-(np.abs(a)))))


def gradient(w, X, y):
    """(1/n) X^T (sigmoid(Xw) - y)  -- derived in Lecture 02."""
    # TODO: implement
    z = X @ w
    n = X.shape[0]
    return (1/n) * (X.T @ (sigmoid(z) - y))


def train_gd(X, y, eta, iters):
    """Full-batch GD from w = 0. Return (w, loss history of length iters+1)."""
    # TODO: implement
    w = np.zeros(X.shape[1])
    history = [logistic_loss(w, X, y)]
    for _ in range(iters):
        w = w - eta * gradient(w, X, y)
        history.append(logistic_loss(w, X, y))
    return w, np.array(history)


def predict_labels(w, X, threshold=0.5):
    """Predict 1 where sigmoid(Xw) >= threshold, else 0."""
    # TODO: implement
    return (sigmoid(X @ w) >= threshold).astype(float)


def error_rate(yhat, y):
    # TODO: implement
    return np.mean(yhat != y)


# ============================================================
# Problem 2: L2 regularization
# ============================================================

def l2_objective(w, X, y, lam):
    """logistic_loss(w) + (lam/2) * sum_{j != bias} w_j^2."""
    # TODO: implement
    return logistic_loss(w, X, y) + (lam/2) * np.sum(w[:-1]**2)


def l2_gradient(w, X, y, lam):
    """Gradient of l2_objective. Do NOT penalize the bias coordinate."""
    # TODO: implement
    reg = w * lam
    reg[-1] = 0
    return gradient(w, X, y) + reg

def train_gd_l2(X, y, lam, eta, iters):
    """Full-batch GD on l2_objective from w = 0. Return w."""
    # TODO: implement
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        w = w - eta * l2_gradient(w, X, y, lam)
    return w


# ============================================================
# Problem 3: L1 regularization via proximal gradient (ISTA)
# ============================================================

def soft_threshold(v, tau):
    """Elementwise: sign(v) * max(|v| - tau, 0)."""
    # TODO: implement
    raise NotImplementedError


def train_ista_l1(X, y, lam, eta, iters):
    """Proximal gradient descent on  logistic_loss(w) + lam * ||w_{-bias}||_1:
        v <- w - eta * gradient(w)          (plain gradient step on the loss)
        w <- soft_threshold(v, eta * lam)   (except the bias, which keeps v[-1])
    from w = 0. Return w."""
    # TODO: implement
    raise NotImplementedError


# ============================================================
# Reporting (do not edit below this line)
# ============================================================

def main():
    Xtr, ytr, Xva, yva, Xte, yte = load_and_split(DATA)
    print(f"split sizes: train={len(ytr)} val={len(yva)} test={len(yte)}")

    # ---- 1.1: train on min-max scaled features ----
    Str, Sva, Ste = min_max_scale(Xtr, Xva, Xte)
    Btr, Bva, Bte = add_bias(Str), add_bias(Sva), add_bias(Ste)
    w, hist = train_gd(Btr, ytr, ETA, ITERS)
    print("== 1.1 ==")
    print(f"final train loss = {hist[-1]:.4f}")
    print(f"train error rate = {error_rate(predict_labels(w, Btr), ytr):.4f}")
    print(f"val error rate   = {error_rate(predict_labels(w, Bva), yva):.4f}")
    print(f"||w|| (no bias)  = {np.linalg.norm(w[:-1]):.3f}")

    # ---- 1.2: identical training on RAW features ----
    Rtr, Rva = add_bias(Xtr), add_bias(Xva)
    w_raw, hist_raw = train_gd(Rtr, ytr, ETA, ITERS)
    print("== 1.2 ==")
    print(f"final train loss = {hist_raw[-1]:.4f}")
    print(f"val error rate   = {error_rate(predict_labels(w_raw, Rva), yva):.4f}")
    plt.figure(figsize=(6, 4))
    plt.plot(hist, label="min-max scaled")
    plt.plot(hist_raw, label="raw features")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("training loss (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logistic_loss_curves.pdf")
    plt.close()

    # ---- 2: L2 sweep ----
    print("== 2 (L2 sweep) ==")
    print(f"{'lambda':>8} {'objective':>10} {'loss':>8} {'train err':>10} {'val err':>8} {'||w||':>8}")
    l2_rows = []
    for lam in LAMBDAS:
        w2 = train_gd_l2(Btr, ytr, lam, ETA, ITERS)
        row = (lam, l2_objective(w2, Btr, ytr, lam), logistic_loss(w2, Btr, ytr),
               error_rate(predict_labels(w2, Btr), ytr),
               error_rate(predict_labels(w2, Bva), yva), np.linalg.norm(w2[:-1]), w2)
        l2_rows.append(row)
        print(f"{lam:8.4f} {row[1]:10.4f} {row[2]:8.4f} {row[3]:10.4f} {row[4]:8.4f} {row[5]:8.3f}")

    # ---- 3: L1 sweep ----
    print("== 3 (L1 sweep, ISTA) ==")
    print(f"{'lambda':>8} {'loss':>8} {'train err':>10} {'val err':>8} {'||w||':>8} {'#nonzero':>9}")
    l1_rows = []
    for lam in LAMBDAS:
        w1 = train_ista_l1(Btr, ytr, lam, ETA, ITERS)
        nnz = int(np.sum(w1[:-1] != 0))
        row = (lam, logistic_loss(w1, Btr, ytr), error_rate(predict_labels(w1, Btr), ytr),
               error_rate(predict_labels(w1, Bva), yva), np.linalg.norm(w1[:-1]), nnz, w1)
        l1_rows.append(row)
        print(f"{lam:8.4f} {row[1]:8.4f} {row[2]:10.4f} {row[3]:8.4f} {row[4]:8.3f} {row[5]:9d}")
    for lam, *_, w1 in l1_rows:
        if lam in (1e-2, 3e-2):
            surv = [f"{FEATURE_NAMES[j]} ({w1[j]:+.2f})" for j in range(30) if w1[j] != 0]
            print(f"lambda={lam}: nonzero features: " + (", ".join(surv) or "none"))
    print(f"L2 at lambda=1e-2: #exact zeros = {int(np.sum(l2_rows[4][6][:-1] == 0))}")

    lams_plot = [lam if lam > 0 else 3e-5 for lam in LAMBDAS]  # log axis; leftmost = lambda 0
    plt.figure(figsize=(6, 4))
    plt.plot(lams_plot, [r[5] for r in l1_rows], "o-", label="L1: # nonzero weights")
    plt.plot(lams_plot, [int(np.sum(r[6][:-1] != 0)) for r in l2_rows], "s--", label="L2: # nonzero weights")
    plt.xscale("log")
    plt.xlabel(r"$\lambda$ (leftmost point is $\lambda=0$)")
    plt.ylabel("number of nonzero weights (of 30)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sparsity.pdf")
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.plot(lams_plot, [r[4] for r in l2_rows], "s--", label="L2 val error")
    plt.plot(lams_plot, [r[3] for r in l1_rows], "o-", label="L1 val error")
    plt.plot(lams_plot, [r[3] for r in l2_rows], "s:", color="gray", label="L2 train error")
    plt.plot(lams_plot, [r[2] for r in l1_rows], "o:", color="gray", label="L1 train error")
    plt.xscale("log")
    plt.xlabel(r"$\lambda$ (leftmost point is $\lambda=0$)")
    plt.ylabel("error rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reg_errors.pdf")
    plt.close()

    # ---- final: touch the test set exactly once ----
    best_l2 = min(l2_rows, key=lambda r: (r[4], r[0]))
    best_l1 = min(l1_rows, key=lambda r: (r[3], -r[0]))
    print("== final test evaluation (once) ==")
    for name, wv in (("unregularized", w), (f"L2 lambda={best_l2[0]}", best_l2[6]),
                     (f"L1 lambda={best_l1[0]}", best_l1[6])):
        print(f"{name:>22}: val err {error_rate(predict_labels(wv, Bva), yva):.4f}, "
              f"test err {error_rate(predict_labels(wv, Bte), yte):.4f}")


if __name__ == "__main__":
    main()
