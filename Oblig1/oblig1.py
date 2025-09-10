import numpy as np
import matplotlib.pyplot as plt

# ---------- DATASET ----------

n = 30
x = np.linspace(-2, 2, n)
eps = 1
np.random.seed(1)
r = np.random.rand(n) * eps
y_1 = x * (np.cos(r + 0.5 * x**3) + np.sin(0.5 * x**3))
y_2 = 4 * x**5 - 5 * x**4 - 20 * x**3 + 10 * x**2 + 40 * x + 10 + r

# ---------- FUNCTIONS ----------


def polynomial_features(x, p, intercept=False):
    n = len(x)
    if intercept:
        X = np.zeros((n, p + 1))
        X[:, 0] = 1
        for i in range(1, p + 1):
            X[:, i] = x**i
        return X
    else:
        X = np.zeros((n, p))
        for i in range(p):
            X[:, i] = x ** (i + 1)
        return X


def substitution(R, b, type):
    n = R.shape[0]
    x = np.zeros(n)

    if type == "Backward":
        for i in range(n - 1, -1, -1):
            if np.isclose(R[i, i], 0):
                raise ValueError(
                    f"Matrix is singular or nearly singular at diagonal index {i}"
                )
            x[i] = (b[i] - (R[i, i + 1 :] @ x[i + 1 :])) / R[i, i]

    elif type == "Forward":
        for i in range(n):
            if np.isclose(R[i, i], 0):
                raise ValueError(
                    f"Matrix is singular or nearly singular at diagonal index {i}"
                )
            x[i] = (b[i] - (R[i, :i] @ x[:i])) / R[i, i]

    return x


def fit_polynomial(x, y, degree):
    """Return predictions using both QR and Cholesky for a given dataset and degree."""
    A = polynomial_features(x, degree, intercept=True)

    # QR
    Q, R = np.linalg.qr(A)
    b = Q.T @ y
    theta_qr = substitution(R, b, "Backward")
    y_pred_qr = A @ theta_qr

    # Cholesky
    AtA = A.T @ A
    Aty = A.T @ y
    L = np.linalg.cholesky(AtA)
    z = substitution(L, Aty, "Forward")
    theta_chol = substitution(L.T, z, "Backward")
    y_pred_chol = A @ theta_chol

    return y_pred_qr, y_pred_chol


def plot_comparison(x, y, y_qr, y_chol, degree, dataset_name, savepath):
    """Plot true solution vs QR vs Cholesky for given dataset and degree."""
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, "o", label=f"Data ({dataset_name})")
    plt.plot(x, y_qr, "-", label=f"QR fit deg={degree}")
    plt.plot(x, y_chol, "--", label=f"Cholesky fit deg={degree}")
    plt.title(f"{dataset_name} Polynomial Fit (deg={degree})", fontsize=16)
    plt.xlabel("x", fontsize=16)
    plt.ylabel(dataset_name, fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


# ---------- FIT + PLOT ----------

# Dataset 1
y1_qr_2, y1_chol_2 = fit_polynomial(x, y_1, 2)
plot_comparison(x, y_1, y1_qr_2, y1_chol_2, 2, "y1", "oblig1/y1_deg2.pdf")

y1_qr_7, y1_chol_7 = fit_polynomial(x, y_1, 7)
plot_comparison(x, y_1, y1_qr_7, y1_chol_7, 7, "y1", "oblig1/y1_deg7.pdf")

# Dataset 2
y2_qr_2, y2_chol_2 = fit_polynomial(x, y_2, 2)
plot_comparison(x, y_2, y2_qr_2, y2_chol_2, 2, "y2", "oblig1/y2_deg2.pdf")

y2_qr_7, y2_chol_7 = fit_polynomial(x, y_2, 7)
plot_comparison(x, y_2, y2_qr_7, y2_chol_7, 7, "y2", "oblig1/y2_deg7.pdf")
