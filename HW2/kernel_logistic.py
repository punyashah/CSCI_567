"""CSCI 567 HW2, Problem 4: kernel logistic regression as a learned nearest-neighbour classifier.

Fill in every function marked TODO. Allowed packages: numpy, matplotlib only.
Run from this directory (after finishing logistic_pipeline.py, which this file
imports from):  python kernel_logistic.py
Do not change seeds, sizes, lambda, the iteration count, or the grids.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from logistic_pipeline import sigmoid, logistic_loss, error_rate

ITERS, LAM = 3000, 1e-3
GAMMAS = [0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0]
KS = [1, 5, 15]
N_TRAIN, N_VAL = 200, 500


def make_rings(n, rng, uneven=False):
    """Two concentric rings in R^2, equal class probabilities: label 0 at radius ~1,
    label 1 at radius ~2, radial noise std 0.3, angles uniform. With uneven=True,
    class 1 keeps the same radius but 85% of its points have angles concentrated
    near 0 (dense on the right, sparse on the left); class 0 stays uniform."""
    y = (rng.uniform(size=n) < 0.5).astype(float)
    r = np.where(y == 1, 2.0, 1.0) + rng.normal(0, 0.3, size=n)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    if uneven:
        conc = 0.7 * rng.normal(size=n)
        use_conc = (y == 1) & (rng.uniform(size=n) < 0.85)
        theta = np.where(use_conc, conc, theta)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1), y


def load_and_split(uneven=False):
    rng = np.random.default_rng(567)
    Xtr, ytr = make_rings(N_TRAIN, rng, uneven)
    Xva, yva = make_rings(N_VAL, rng, uneven)
    return Xtr, ytr, Xva, yva


# ---------------- kernel logistic regression ----------------

def rbf_kernel(A, B, gamma):
    """K[i,j] = exp(-gamma * ||A[i] - B[j]||^2) for A (m,d), B (n,d), with no Python
    loop: use ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b and clip tiny negatives to 0."""
    # TODO: implement
    raise NotImplementedError


def kernel_objective(alpha, K, y, lam):
    """(1/n) sum_i logistic loss of score (K alpha)_i vs y_i + (lam/2) alpha^T K alpha
    (Lecture 04's kernel logistic regression slide). Hint: logistic_loss(w, X, y)
    computes the mean loss of the scores X @ w."""
    # TODO: implement
    raise NotImplementedError


def kernel_gradient(alpha, K, y, lam):
    """K [ (1/n)(sigmoid(K alpha) - y) + lam * alpha ]  (Lecture 04)."""
    # TODO: implement
    n = K.shape[0]
    return K @ ((1/n) * (sigmoid(K @ alpha) - y) + lam * alpha)


def step_size(K, lam):
    """Provided: 1/L with L a bound on the gradient's Lipschitz constant."""
    s = np.linalg.norm(K, 2)
    return 1.0 / (s ** 2 / (4 * K.shape[0]) + lam * s)


def train_kernel_gd(K, y, lam, eta, iters):
    """Full-batch GD on kernel_objective from alpha = 0. Return alpha."""
    # TODO: implement
    raise NotImplementedError


def kernel_predict(alpha, Xtrain, Xnew, gamma):
    """Labels for Xnew: 1 where sigmoid(sum_i alpha_i kappa(x_new, x_i)) >= 0.5."""
    # TODO: implement
    raise NotImplementedError


# ---------------- the two unlearned neighbour heuristics ----------------

def knn_predict(Xtrain, ytrain, Xnew, k):
    """k-nearest-neighbour classifier: for each row of Xnew find the k training
    points with the smallest Euclidean distance and predict 1 iff more than half
    of them have label 1 (so ties go to 0). No Python loop over points needed:
    compute all squared distances at once, then np.argsort along axis 1."""
    # TODO: implement
    raise NotImplementedError


def weighted_vote_predict(Xtrain, ytrain, Xnew, gamma):
    """Kernel-weighted vote: score(x) = sum_i kappa(x, x_i) * (2 y_i - 1) with the
    RBF kernel; predict 1 iff the score is positive."""
    # TODO: implement
    raise NotImplementedError


# ============================================================
# Reporting (do not edit below this line)
# ============================================================

def plot_boundary(ax, Xtr, ytr, predict, title, sizes=None):
    g = np.linspace(-3, 3, 201)
    GX, GY = np.meshgrid(g, g)
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
    P = predict(grid).reshape(GX.shape)
    ax.contourf(GX, GY, P, levels=[-0.5, 0.5, 1.5], colors=["#cfe3f7", "#f7d6d6"], alpha=0.8)
    s0 = 8 if sizes is None else sizes[ytr == 0]
    s1 = 8 if sizes is None else sizes[ytr == 1]
    ax.scatter(*Xtr[ytr == 0].T, s=s0, c="tab:blue", label="y=0")
    ax.scatter(*Xtr[ytr == 1].T, s=s1, c="tab:red", label="y=1")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def sweep(Xtr, ytr, Xva, yva, tag):
    print(f"== {tag}: k-NN == " + ", ".join(
        f"k={k}: {error_rate(knn_predict(Xtr, ytr, Xva, k), yva):.3f}" for k in KS))
    print(f"== {tag}: gamma sweep (val error) ==")
    print(f"{'gamma':>7} {'weighted vote':>14} {'kernel logistic':>16}")
    alphas = {}
    for gamma in GAMMAS:
        K = rbf_kernel(Xtr, Xtr, gamma)
        alpha = train_kernel_gd(K, ytr, LAM, step_size(K, LAM), ITERS)
        alphas[gamma] = alpha
        print(f"{gamma:7g} {error_rate(weighted_vote_predict(Xtr, ytr, Xva, gamma), yva):14.3f} "
              f"{error_rate(kernel_predict(alpha, Xtr, Xva, gamma), yva):16.3f}")
    return alphas


def main():
    # ---------- 4.1 / 4.4: balanced rings ----------
    Xtr, ytr, Xva, yva = load_and_split(uneven=False)
    print(f"balanced rings: train={len(ytr)} val={len(yva)}, class-1 fraction {ytr.mean():.2f}")
    alphas = sweep(Xtr, ytr, Xva, yva, "balanced")
    agree = np.mean(weighted_vote_predict(Xtr, ytr, Xva, 1000.0) == knn_predict(Xtr, ytr, Xva, 1))
    agree_klr = np.mean(kernel_predict(alphas[1000.0], Xtr, Xva, 1000.0) == knn_predict(Xtr, ytr, Xva, 1))
    print(f"gamma=1000: weighted vote agrees with 1-NN on {agree:.3f} of val points; "
          f"kernel logistic on {agree_klr:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    plot_boundary(axes[0], Xtr, ytr, lambda G: weighted_vote_predict(Xtr, ytr, G, 1.0), r"weighted vote, $\gamma=1$")
    plot_boundary(axes[1], Xtr, ytr, lambda G: kernel_predict(alphas[1.0], Xtr, G, 1.0), r"kernel logistic, $\gamma=1$")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("balanced_boundaries.pdf")
    plt.close()

    # ---------- 4.5: uneven density ----------
    Xtr, ytr, Xva, yva = load_and_split(uneven=True)
    left_tr, left_va = Xtr[:, 0] < 0, Xva[:, 0] < 0
    print(f"\nuneven rings: class-1 fraction on the left half of train {ytr[left_tr].mean():.2f}, "
          f"right half {ytr[~left_tr].mean():.2f}")
    print("== uneven: k-NN (all / left half) == " + ", ".join(
        f"k={k}: {error_rate(knn_predict(Xtr, ytr, Xva, k), yva):.3f}/"
        f"{error_rate(knn_predict(Xtr, ytr, Xva[left_va], k), yva[left_va]):.3f}" for k in KS))
    print(f"{'gamma':>7} {'vote all':>9} {'vote left':>10} {'KLR all':>9} {'KLR left':>9}")
    alphas_u = {}
    for gamma in [0.3, 1.0, 3.0, 10.0]:
        K = rbf_kernel(Xtr, Xtr, gamma)
        alpha = train_kernel_gd(K, ytr, LAM, step_size(K, LAM), ITERS)
        alphas_u[gamma] = alpha
        v = weighted_vote_predict(Xtr, ytr, Xva, gamma)
        p = kernel_predict(alpha, Xtr, Xva, gamma)
        print(f"{gamma:7g} {error_rate(v, yva):9.3f} {error_rate(v[left_va], yva[left_va]):10.3f} "
              f"{error_rate(p, yva):9.3f} {error_rate(p[left_va], yva[left_va]):9.3f}")
    gamma = 1.0
    alpha = alphas_u[gamma]
    K = rbf_kernel(Xtr, Xtr, gamma)
    beta = (ytr - sigmoid(K @ alpha)) / (len(ytr) * LAM)      # the residual weights of 4.2
    print(f"identity check at gamma={gamma}: predictions from beta = (y - p)/(n lam) agree with "
          f"those from alpha on {np.mean(kernel_predict(beta, Xtr, Xva, gamma) == kernel_predict(alpha, Xtr, Xva, gamma)):.3f} "
          f"of val points; max |beta| = {np.abs(beta).max():.3f} vs cap 1/(n lam) = {1/(len(ytr)*LAM):.1f}; "
          f"sign(beta) == 2y-1 on {np.mean(np.sign(beta) == 2*ytr-1):.3f}")
    for cls in (1, 0):
        m = ytr == cls
        print(f"  mean |beta| for class {cls}: left half {np.abs(beta[m & left_tr]).mean():.3f}, "
              f"right half {np.abs(beta[m & ~left_tr]).mean():.3f}")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    plot_boundary(axes[0], Xtr, ytr, lambda G: weighted_vote_predict(Xtr, ytr, G, gamma), r"weighted vote, $\gamma=1$")
    plot_boundary(axes[1], Xtr, ytr, lambda G: kernel_predict(alpha, Xtr, G, gamma), r"kernel logistic, $\gamma=1$")
    plot_boundary(axes[2], Xtr, ytr, lambda G: kernel_predict(alpha, Xtr, G, gamma),
                  r"same model; marker area $\propto |\alpha_i|$", sizes=4 + 60 * np.abs(beta) / np.abs(beta).max())
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("uneven_boundaries.pdf")
    plt.close()


if __name__ == "__main__":
    main()
