"""
En aquest apartat es resol l'equació de la calor adimensional  dT/dt = d2T/dx2 + 1  amb el mètode
d'Euler explícit, concretament amb la derivada temporal per la dreta.
"""

import numpy as np
import matplotlib.pyplot as plt

from parametres import N, t_a
from sol_analitica import T_analitica, error
from estil_figures import desa


def resol(N, dt, t_a):
    dx = 1 / (N - 1)

    n_passos = round(t_a / dt)
    dt = t_a / n_passos
    gamma = dt / dx**2

    x = np.linspace(0, 1, N)
    T_vella = np.zeros(N)

    for pas in range(n_passos):
        T_nova = np.zeros(N)
        for j in range(1, N - 1):
            T_nova[j] = T_vella[j] + gamma * (T_vella[j + 1] - 2 * T_vella[j] + T_vella[j - 1]) + dt
        T_vella = T_nova

    return x, T_vella


if __name__ == "__main__":
    dx = 1 / (N - 1)

    x, T_051 = resol(N, 0.51 * dx**2, t_a)
    x, T_049 = resol(N, 0.49 * dx**2, t_a)
    x, T_025 = resol(N, 0.25 * dx**2, t_a)

    T_ana = T_analitica(x, t_a)

    err_051 = error(T_051, T_ana)
    err_049 = error(T_049, T_ana)
    err_025 = error(T_025, T_ana)

    print("error màxim 0.51:", err_051.max())
    print("error màxim 0.49:", err_049.max())
    print("error màxim 0.25:", err_025.max())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.8, 2.4))

    ax1.plot(x, T_ana, "k-", label=r"$\hat{T}_\mathrm{ana}$")
    ax1.plot(x, T_049, "s", color="C0", markevery=5, ms=3, label=r"$\gamma\approx 0.49$")
    ax1.plot(x, T_025, "o", color="C2", markevery=5, ms=3, label=r"$\gamma = 0.25$")
    ax1.set_xlabel(r"$\hat{x}$")
    ax1.set_ylabel(r"$\hat{T}(\hat{x}, \hat{t}_a)$")
    ax1.legend(loc="lower center")
    ax1.text(0.04, 0.93, "(a)", transform=ax1.transAxes, ha="left", va="top")

    ax2.plot(x, T_ana, "k-", label=r"$\hat{T}_\mathrm{ana}$")
    ax2.plot(x, T_051, "^", color="C3", markevery=5, ms=3, label=r"$\gamma\approx 0.51$")
    ax2.set_xlabel(r"$\hat{x}$")
    ax2.set_ylabel(r"$\hat{T}(\hat{x}, \hat{t}_a)$")
    ax2.set_ylim(-3.6, 2.6)
    ax2.legend(loc="lower center", ncol=2, columnspacing=1.0, handletextpad=0.4)
    ax2.text(0.04, 0.93, "(b)", transform=ax2.transAxes, ha="left", va="top")

    ax3.semilogy(x[1:-1], err_051[1:-1], color="C3", label=r"$\gamma\approx 0.51$")
    ax3.semilogy(x[1:-1], err_049[1:-1], color="C0", label=r"$\gamma\approx 0.49$")
    ax3.semilogy(x[1:-1], err_025[1:-1], color="C2", label=r"$\gamma = 0.25$")
    ax3.set_xlabel(r"$\hat{x}$")
    ax3.set_ylabel(r"$|\hat{T}_\mathrm{num}-\hat{T}_\mathrm{ana}|$")
    ax3.set_ylim(1e-7, 1e2)
    ax3.legend(loc="center right")
    ax3.text(0.04, 0.93, "(c)", transform=ax3.transAxes, ha="left", va="top")

    desa(fig, "explicit_solucio_error")