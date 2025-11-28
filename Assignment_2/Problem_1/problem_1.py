"""
Problem 1 solution script for the Duffing system.

The system:
    x1' = x2
    x2' = mu * x1 - x1**3

Cells are organized with `# %%` for notebook-like execution.
Figures are saved to Assignment_2/figures/Problem_1/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sympy as sp
from scipy.integrate import solve_ivp

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "Problem_1"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


# %%
# Vector field helpers
def f(x1: np.ndarray, x2: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray]:
    x1_arr = np.asarray(x1)
    x2_arr = np.asarray(x2)
    return x2_arr, mu * x1_arr - x1_arr**3


def H(x1: np.ndarray, x2: np.ndarray, mu: float) -> np.ndarray:
    """Hamiltonian / energy function (conserved)."""
    return 0.5 * x2**2 - 0.5 * mu * x1**2 + 0.25 * x1**4


def equilibrium_points(mu: float) -> list[tuple[float, float]]:
    pts = [(0.0, 0.0)]
    if mu > 0:
        root = float(np.sqrt(mu))
        pts.extend([(root, 0.0), (-root, 0.0)])
    return pts


def jacobian(x1: float, mu: float) -> np.ndarray:
    return np.array([[0.0, 1.0], [mu - 3 * x1**2, 0.0]])


def classify_equilibrium(x1: float, mu: float, tol: float = 1e-12) -> str:
    a = mu - 3 * x1**2
    if abs(a) < tol:
        return "nonhyperbolic"
    if a > 0:
        ev = np.sqrt(a)
        return f"saddle (±{ev:.3g})"
    ev = np.sqrt(-a)
    return f"center (±{ev:.3g}i)"


# %%
# Analytical homoclinic branch for mu>0
def h_branch(x1: np.ndarray, mu: float) -> np.ndarray:
    inside = mu - 0.5 * x1**2
    inside = np.maximum(inside, 0.0)
    return np.abs(x1) * np.sqrt(inside)


# %%
# Phase portrait plotting
def plot_phase_portraits(mus=(-1.0, 1.0)) -> None:
    fig, axes = plt.subplots(1, len(mus), figsize=(12, 5))
    if len(mus) == 1:
        axes = [axes]
    for ax, mu in zip(axes, mus):
        xlim = ylim = (-3, 3)
        nx = ny = 28
        X, Y = np.meshgrid(np.linspace(*xlim, nx), np.linspace(*ylim, ny))
        U, V = f(X, Y, mu)
        M = np.hypot(U, V)
        M[M == 0] = 1.0
        ax.quiver(X, Y, U / M, V / M, color="0.25", scale=22, width=0.0035, alpha=0.9)

        # equilibria
        for ex, ey in equilibrium_points(mu):
            ax.plot(ex, ey, "o", color="#D95F02", ms=7, zorder=5)
            ax.text(ex + 0.08, ey + 0.08, f"({ex:.2g},0)", color="#D95F02", fontsize=9)

        # energy level curves
        xg = np.linspace(*xlim, 350)
        yg = np.linspace(*ylim, 350)
        XX, YY = np.meshgrid(xg, yg)
        HH = H(XX, YY, mu)
        levels = np.linspace(np.percentile(HH, 5), np.percentile(HH, 85), 10)
        ax.contour(
            XX, YY, HH, levels=levels, colors="#1f78b4", linewidths=1.0, alpha=0.8
        )
        ax.contour(
            XX,
            YY,
            HH,
            levels=[0.0],
            colors="#084594",
            linewidths=2.0,
            linestyles="--",
            alpha=0.9,
        )

        # trajectories
        inits = [
            (0.7, 0.2),
            (1.05, 0.45),
            (1.4, 0.1),
            (0.5, -0.3),
            (-1.1, 0.3),
            (-2.0, -0.4),
        ]
        t_span = 80.0 if mu <= 0 else 40.0
        t_eval = np.linspace(0, t_span, 2000)
        for z0 in inits:
            sol_f = solve_ivp(
                lambda t, z: f(z[0], z[1], mu),
                [0, t_span],
                z0,
                t_eval=t_eval,
                rtol=1e-9,
                atol=1e-12,
            )
            ax.plot(sol_f.y[0], sol_f.y[1], color="#1f78b4", lw=0.9, alpha=0.9)
            sol_b = solve_ivp(
                lambda t, z: f(z[0], z[1], mu),
                [0, -t_span],
                z0,
                t_eval=np.linspace(0, -t_span, 2000),
                rtol=1e-9,
                atol=1e-12,
            )
            ax.plot(sol_b.y[0], sol_b.y[1], color="#1f78b4", lw=0.9, alpha=0.9)

        # homoclinic for mu>0
        if mu > 0:
            xs = np.linspace(1e-6, np.sqrt(2 * mu) - 1e-6, 600)
            hb = h_branch(xs, mu)
            ax.plot(
                xs,
                hb,
                color="#e41a1c",
                lw=2.1,
                alpha=0.95,
                label=r"$x_2 = \pm |x_1|\sqrt{\mu - \tfrac12 x_1^2}$",
            )
            ax.plot(xs, -hb, color="#e41a1c", lw=2.1, alpha=0.95)

        ax.set(
            title=rf"$\mu={mu}$",
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            xlim=xlim,
            ylim=ylim,
        )
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.15, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "problem1_phase_portraits.pdf")
    plt.close(fig)


# %%
# Homoclinic plot with analytic branch overlay
def plot_homoclinic(mu: float = 1.0) -> None:
    xlim = ylim = (-3, 3)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    # vector field
    nx = ny = 25
    X, Y = np.meshgrid(np.linspace(*xlim, nx), np.linspace(*ylim, ny))
    U, V = f(X, Y, mu)
    M = np.hypot(U, V)
    M[M == 0] = 1.0
    ax.quiver(X, Y, U / M, V / M, color="0.25", scale=22, width=0.003, alpha=0.85)

    # equilibria
    for ex, ey in equilibrium_points(mu):
        ax.plot(ex, ey, "s", color="#D95F02", ms=7)
        ax.text(ex + 0.08, ey + 0.08, f"({ex:.2g},0)", color="#D95F02", fontsize=9)

    # analytic h branch
    xs = np.linspace(1e-6, np.sqrt(2 * mu) - 1e-6, 500)
    hb = h_branch(xs, mu)
    ax.plot(xs, hb, color="#e41a1c", lw=2.2, label=r"Analytic $h(x_1)$ (Q1)")
    ax.plot(
        xs, -hb, color="#e41a1c", lw=2.2, alpha=0.8, label=r"Analytic $-h(x_1)$ (Q4)"
    )

    # Hamiltonian zero-level
    xs_grid = np.linspace(*xlim, 400)
    ys_grid = np.linspace(*ylim, 400)
    XX, YY = np.meshgrid(xs_grid, ys_grid)
    HH = H(XX, YY, mu)
    cs = ax.contour(
        XX, YY, HH, levels=[0.0], colors="#377eb8", linewidths=1.6, linestyles="--"
    )
    if len(cs.allsegs[0]) > 0:
        ax.plot([], [], color="#377eb8", ls="--", label=r"$H=0$ (homoclinic)")

    ax.set(
        title=rf"Homoclinic orbit vs. vector field, $\mu={mu}$",
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        xlim=xlim,
        ylim=ylim,
    )
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "problem1_homoclinic.pdf")
    plt.close(fig)


# %%
# Self-checks
def run_self_checks() -> None:
    # equilibrium structure
    assert equilibrium_points(-0.5) == [(0.0, 0.0)]
    mu_pos = 0.5
    eqs = equilibrium_points(mu_pos)
    assert len(eqs) == 3
    root = np.sqrt(mu_pos)
    assert any(np.isclose(pt[0], root) for pt in eqs)
    assert any(np.isclose(pt[0], -root) for pt in eqs)

    # classification
    assert "center" in classify_equilibrium(0.0, mu=-1.0)
    assert "saddle" in classify_equilibrium(0.0, mu=1.0)

    # Lyapunov/Hamiltonian conservation along a trajectory
    z0 = np.array([1.0, 0.5])
    t_eval = np.linspace(0, 20.0, 400)
    sol = solve_ivp(
        lambda t, z: f(z[0], z[1], mu_pos),
        [0, 20.0],
        z0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
    )
    H_vals = H(sol.y[0], sol.y[1], mu_pos)
    assert np.max(np.abs(H_vals - H_vals[0])) < 1e-6

    # homoclinic satisfies H=0
    x_sample = 1.0
    h_s = h_branch(np.array([x_sample]), mu=1.0)
    assert abs(H(x_sample, h_s, mu=1.0)) < 1e-12

    # symbolic Vdot = 0
    x1s, x2s, mus = sp.symbols("x1 x2 mu", real=True)
    Vsym = (
        sp.Rational(1, 2) * x2s**2
        - sp.Rational(1, 2) * mus * x1s**2
        + sp.Rational(1, 4) * x1s**4
    )
    dx1s = x2s
    dx2s = mus * x1s - x1s**3
    Vdot = sp.simplify(sp.diff(Vsym, x1s) * dx1s + sp.diff(Vsym, x2s) * dx2s)
    assert sp.simplify(Vdot) == 0

    print("Problem 1 self-checks passed.")


# %%
if __name__ == "__main__":
    run_self_checks()
    plot_phase_portraits()
    plot_homoclinic(mu=1.0)
    print(f"Figures written to {FIGURES_DIR}")
