"""
Problem 2 solution script for the Duffing system

    x1' = x2
    x2' = mu * x1 - x1**3

The file is organized in VS Code/Jupyter style cells (``# %%``) so the work
flow for each question can be run independently.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sympy as sp

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "Problem_2"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# %%
# Helpers for the Duffing vector field and its linearization
def f(x1: np.ndarray, x2: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Vector field components for x1' and x2'."""
    x1_arr = np.asarray(x1)
    x2_arr = np.asarray(x2)
    return x2_arr, mu * x1_arr - x1_arr**3


def jacobian(x1: float, mu: float) -> np.ndarray:
    """Jacobian evaluated at (x1, 0)."""
    return np.array([[0.0, 1.0], [mu - 3 * x1**2, 0.0]])


def equilibrium_points(mu: float) -> list[tuple[float, float]]:
    """Critical points of the system for a given mu."""
    pts = [(0.0, 0.0)]
    if mu > 0:
        root = np.sqrt(mu)
        pts.extend([(root, 0.0), (-root, 0.0)])
    return pts


def classify_equilibrium(x1: float, mu: float, tol: float = 1e-10) -> str:
    """Classify (x1, 0) using eigenvalues of the linearization."""
    a = mu - 3 * x1**2
    if abs(a) < tol:
        return "nonhyperbolic (double zero eigenvalues)"
    if a > 0:
        eig = np.sqrt(a)
        return f"saddle (real eigenvalues ±{eig:.3g})"
    eig = np.sqrt(-a)
    return f"center (imaginary eigenvalues ±{eig:.3g}i)"


# %%
# (a) Bifurcation diagram: equilibrium branches as functions of mu
mus = np.linspace(-1.0, 1.2, 400)
x1_zero = np.zeros_like(mus)
x1_pitchfork = np.sqrt(np.clip(mus, 0.0, None))

fig, ax = plt.subplots()
sns.set_theme(style="whitegrid")
ax.plot(mus[mus < 0], x1_zero[mus < 0], "C1-", label=r"$x_1 = 0$ (stable $\mu<0$)")
ax.plot(mus[mus >= 0], x1_zero[mus >= 0], "C1--", label=r"$x_1 = 0$ (saddle $\mu>0$)")
ax.plot(mus, x1_pitchfork, "C2-", label=r"$x_1 = +\sqrt{\mu}$")
ax.plot(mus, -x1_pitchfork, "C2-", label=r"$x_1 = -\sqrt{\mu}$")
ax.axvline(0.0, color="0.5", linewidth=1)
ax.set(
    xlabel=r"$\mu$",
    ylabel=r"$x_1$ equilibrium coordinate",
    title="Bifurcation diagram for the Duffing system",
    xlim=(-1.0, 1.1),
    ylim=(-1.5, 1.5),
)
ax.legend(loc="upper left")
ax.grid(True, linestyle=":")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "problem2_bifurcation.pdf")
plt.close(fig)


# %%
# (b) Bifurcation type
for mu_test in (-0.5, 0.0, 0.5):
    for pt in equilibrium_points(mu_test):
        print(
            f"mu={mu_test:+.2f}, equilibrium {pt}: {classify_equilibrium(pt[0], mu_test)}"
        )

print(
    "\nObservation: the symmetric pair of equilibria ±sqrt(mu) is created for mu>0, "
    "and the origin changes stability at mu=0. This is a supercritical pitchfork bifurcation."
)


# %%
# (c) Hyperbolicity check for mu = 0
J0 = jacobian(0.0, mu=0.0)
eigs0 = np.linalg.eigvals(J0)
print("Jacobian at (0,0) for mu=0:\n", J0)
print("Eigenvalues:", eigs0)
print("The equilibrium is nonhyperbolic because both eigenvalues are 0.")


# %%
# (d) Liapunov function for mu = 0
def lyapunov(x1: np.ndarray, x2: np.ndarray, mu: float = 0.0) -> np.ndarray:
    """
    Energy-like conserved quantity. For general mu the expression
    V(x1, x2) = 0.5*x2**2 - 0.5*mu*x1**2 + 0.25*x1**4
    satisfies dV/dt = 0. For mu = 0 it is positive definite and proves stability.
    """
    return 0.5 * x2**2 - 0.5 * mu * x1**2 + 0.25 * x1**4


def lyapunov_derivative(x1: float, x2: float, mu: float = 0.0) -> float:
    """Time derivative of V along trajectories."""
    # gradient of V dotted with f
    dVdx1 = -mu * x1 + x1**3
    dVdx2 = x2
    dx1, dx2 = f(np.array([x1]), np.array([x2]), mu)
    return float(dVdx1 * dx1.item() + dVdx2 * dx2.item())


print("Liapunov function for mu=0: V = 0.5*x2^2 + 0.25*x1^4")
print("dV/dt for mu=0:", lyapunov_derivative(0.3, -0.4, mu=0.0))
print(
    "dV/dt vanishes identically for mu=0, so V is conserved -> stability (not asymptotic)."
)


# %%
# (e) Contour plot of V together with the vector field for mu = 0
mu0 = 0.0
grid_lim = 2.5
xs = np.linspace(-grid_lim, grid_lim, 61)
X1, X2 = np.meshgrid(xs, xs)
DX1, DX2 = f(X1, X2, mu0)
V = lyapunov(X1, X2, mu0)

fig, ax = plt.subplots(figsize=(6, 5))
cs = ax.contour(X1, X2, V, levels=15, cmap="viridis")
ax.clabel(cs, inline=True, fontsize=8)
ax.streamplot(X1, X2, DX1, DX2, color="k", density=1.2, linewidth=0.7, arrowsize=0.9)
ax.set(
    title=r"Vector field vs. level sets of $V$ for $\mu=0$",
    xlabel=r"$x_1$",
    ylabel=r"$x_2$",
    xlim=(-grid_lim, grid_lim),
    ylim=(-grid_lim, grid_lim),
)
ax.set_aspect("equal", "box")
ax.grid(True, linestyle=":")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "problem2_lyapunov_contours.pdf")
plt.close(fig)

print(
    "Level sets of V coincide with trajectories for mu=0 because V is conserved. "
    "The closed contours illustrate neutral (center-type) stability."
)


# %%
# (f) Stability insight from Perko Theorem 2 (p.151): with V positive definite and dV/dt=0,
# the origin is stable but not asymptotically stable (trajectories stay on closed level sets).
print(
    "Perko Thm. 2 (p.151) applies with V positive definite and Vdot=0, "
    "so the equilibrium at (0,0) for mu=0 is stable (Lyapunov) but not asymptotically stable."
)


# %%
# Lightweight self-checks colocated with subproblem steps
# Equilibria list sanity
assert equilibrium_points(-0.5) == [(0.0, 0.0)]
mu_pos = 0.5
eq_pos = equilibrium_points(mu_pos)
expected_root = np.sqrt(mu_pos)
assert len(eq_pos) == 3
assert any(np.isclose(pt[0], expected_root) and np.isclose(pt[1], 0.0) for pt in eq_pos)
assert any(
    np.isclose(pt[0], -expected_root) and np.isclose(pt[1], 0.0) for pt in eq_pos
)
print("Equilibria self-checks passed.")

# Classification checks
assert "center" in classify_equilibrium(0.0, mu=-0.5)
assert "saddle" in classify_equilibrium(0.0, mu=0.5)
assert "center" in classify_equilibrium(expected_root, mu=mu_pos)
print("Classification self-checks passed.")

# Lyapunov derivative is zero for mu=0
rng = np.random.default_rng(0)
for _ in range(5):
    x1_val, x2_val = rng.uniform(-1.5, 1.5, size=2)
    assert abs(lyapunov_derivative(x1_val, x2_val, mu=0.0)) < 1e-12
print("Lyapunov derivative self-checks passed.")

# Homoclinic expression satisfies V=0 for mu>0 and |x1| small enough
mu_h = 1.0
x1_sample = 1.0
x2_sample = abs(x1_sample) * np.sqrt(mu_h - 0.5 * x1_sample**2)
assert (
    abs(lyapunov(np.array([x1_sample]), np.array([x2_sample]), mu=mu_h).item()) < 1e-12
)
print("Homoclinic self-checks passed.")


# %%
# Symbolic verification with sympy (hard requirement)
x1s, x2s, mus = sp.symbols("x1 x2 mu", real=True)
Vsym = (
    sp.Rational(1, 2) * x2s**2
    - sp.Rational(1, 2) * mus * x1s**2
    + sp.Rational(1, 4) * x1s**4
)
dx1s = x2s
dx2s = mus * x1s - x1s**3
Vdot_sym = sp.simplify(sp.diff(Vsym, x1s) * dx1s + sp.diff(Vsym, x2s) * dx2s)
assert sp.simplify(Vdot_sym) == 0

# Homoclinic substitution
h_sym = sp.Abs(x1s) * sp.sqrt(mus - sp.Rational(1, 2) * x1s**2)
V_on_sep = sp.simplify(Vsym.subs({x2s: h_sym}))
assert sp.simplify(V_on_sep) == 0

print("Sympy verification passed (Vdot=0 symbolically; separatrix satisfies V=0).")
