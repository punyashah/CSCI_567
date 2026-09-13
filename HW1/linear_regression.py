"""CSCI 567 HW1, Problem 5: linear regression -- closed form, GD, SGD.

Fill in every function marked TODO. Allowed packages: numpy, matplotlib only.
Run:  python linear_regression.py
Keep the seeds exactly as given so your reported numbers are comparable.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, N = 100, 1000


def generate_data(rng):
    """Draw (X, y) from the linear model described in the problem statement."""
    X = rng.normal(0, 1, size=(N, D))
    w_true = rng.normal(0, 1, size=D)
    y = X @ w_true + rng.normal(0, 0.5, size=N)
    return X, y, w_true


def emp_risk(w, X, y):
    """Empirical risk (1/2n) * sum_i (w^T x_i - y_i)^2."""
    n = X.shape[0]
    tot = 0
    for i in range(n):
        tot += (np.transpose(w) @ X[i] - y[i]) ** 2
    return (1 / (2 * n)) * tot
    


def closed_form(X, y):
    """Return the least-squares solution of X^T X w = X^T y.

    Use np.linalg.solve rather than explicitly inverting X^T X.
    """
    return np.linalg.solve(np.transpose(X) @ X, np.transpose(X) @ y)


def full_gradient(w, X, y):
    """Gradient of emp_risk at w: (1/n) X^T (X w - y)."""
    n = X.shape[0]
    return (1 / n) * (np.transpose(X) @ ((X @ w) - y))


def gradient_descent(X, y, eta, iters):
    """Run GD from w = 0 for `iters` steps.

    Return (final w, array of emp_risk values of length iters+1,
    including the starting point).
    """
    w = np.zeros(D)
    hist = [emp_risk(w, X, y)]
    for _ in range(iters):
        w = w - eta * full_gradient(w, X, y)
        hist.append(emp_risk(w, X, y))
    return (w, np.array(hist))


def sgd(X, y, eta, iters, rng):
    """Run SGD from w = 0 for `iters` steps.

    At each step draw i = rng.integers(len(y)) and take a step on the
    single-example loss (1/2)(w^T x_i - y_i)^2. Return (final w, array of
    emp_risk values of length iters+1). Recording emp_risk on the FULL
    training set at every step is for plotting only -- the update itself
    must touch one example.
    """
    w = np.zeros(D)
    hist = [emp_risk(w, X, y)]
    for _ in range(iters):
        i = rng.integers(len(y))
        w = w - eta * (X[i] * ((np.transpose(w) @ X[i]) - y[i]))
        hist.append(emp_risk(w, X, y))
    return (w, np.array(hist))


def main():
    rng = np.random.default_rng(567)
    X, y, w_true = generate_data(rng)
    # test data: same model, fresh draw (do not reorder these rng calls)
    Xte = rng.normal(0, 1, size=(N, D))
    yte = Xte @ w_true + rng.normal(0, 0.5, size=N)

    # ---- 5.1 ----
    w_ls = closed_form(X, y)
    print("== 5.1 ==")
    print(f"train EmpRisk(w_LS) = {emp_risk(w_ls, X, y):.6f}")
    print(f"train EmpRisk(0)    = {emp_risk(np.zeros(D), X, y):.4f}")
    print(f"test  EmpRisk(w_LS) = {emp_risk(w_ls, Xte, yte):.6f}")

    # ---- 5.2 ----
    print("== 5.2 ==")
    plt.figure(figsize=(6, 4))
    for eta in [0.005, 0.5, 1.3]:
        w, hist = gradient_descent(X, y, eta, 20)
        print(f"GD  eta={eta:<6} final EmpRisk = {hist[-1]:.6g}")
        plt.plot(hist, marker="o", ms=3, label=f"eta={eta}")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("empirical risk (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gd_curves.pdf")
    plt.close()

    # ---- 5.3 ----
    print("== 5.3 ==")
    plt.figure(figsize=(6, 4))
    for eta in [0.001, 0.01, 0.02]:
        w, hist = sgd(X, y, eta, 1000, np.random.default_rng(0))
        print(f"SGD eta={eta:<6} final EmpRisk = {hist[-1]:.6g}")
        plt.plot(hist, lw=1, label=f"eta={eta}")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("empirical risk (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sgd_curves.pdf")
    plt.close()


if __name__ == "__main__":
    main()
