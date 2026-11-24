

import numpy as np
from gauss_seidel import gauss_seidel

"""
Apliquem el mètode de Crank-Nicolson a l'equació de la calor ja adimensionalitzada:

    dT/dt = d2T/dx2 + 1, x en [0, 1], T = 0 a les fronteres i a t = 0

A cada pas de temps es resol el sistema tridiagonal

    -(g/2) T_{j-1}^{nou} + (1+g) T_j^{nou} - (g/2) T_{j+1}^{nou}
    = (g/2) T_{j-1}^{vell} + (1-g) T_j^{vell} + (g/2) T_{j+1}^{vell} + dt

amb gamma= g = dt/dx^2, fent servir el Gauss-Seidel definit a un arxiu anterior.
"""

def resol_cn(N, dt, t_a):
    """
    Resol l'equació de la calor amb Crank-Nicolson fins a t = t_a.

    N: nombre de punts de la malla espacial (inclou les dues fronteres).
    dt: pas de temps adimensional.
    t_a: instant final adimensional.

    Ens dona:
    x: array de N posicions, de 0 a 1.
    T: array de N temperatures adimensionals a t = t_a.
    """
    # Discretitzem
    dx = 1.0 / (N - 1)
    n_passos = round(t_a / dt)
    dt = t_a / n_passos              # ajust perquè acabi exactament a t_a
    gamma = dt / dx**2

    x = np.linspace(0, 1, N)
    T = np.zeros(N)                  # condició inicial: T = 0 a tot arreu

    # Matriu del sistema d'equacions, sobre la que farem Gauss-Seidel
    n = N - 2                        # incògnites: només els punts interiors
    a = np.full(n, -gamma / 2)       # subdiagonal   (veí de l'esquerra)
    b = np.full(n, 1 + gamma)        # diagonal
    c = np.full(n, -gamma / 2)       # superdiagonal (veí de la dreta)

    # Fem avançar el temps
    for i in range(n_passos):
        # Costat dret: tot amb la T del pas anterior (coneguda).
        # T[:-2] = T_{j-1},  T[1:-1] = T_j,  T[2:] = T_{j+1}
        d = (gamma / 2) * T[:-2] + (1 - gamma) * T[1:-1] + (gamma / 2) * T[2:] + dt

        # Resolem el sistema. Punt de partida: la T del pas anterior.
        T_interior, n_iter = gauss_seidel(a, b, c, d, x0=T[1:-1])
        T[1:-1] = T_interior         # les fronteres T[0] i T[-1] es queden a 0

    return x, T

# Tot el que hi ha dins d'aquest if només s'executa quan corres aquest
# fitxer directament. Si un altre fitxer importa resol_cn(), no s'executa.
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from parametres import N, t_a
    from sol_analitica import T_analitica, error
    from estil_figures import desa

    dx = 1.0 / (N - 1)

    #Els dos casos que demana l'enunciat.
    casos = [
        (1.0, r"$\Delta\hat{t} = (\Delta\hat{x})^2$", "-"),
        (0.5, r"$\Delta\hat{t} = 0.5\,(\Delta\hat{x})^2$", "--"),
    ]

    # Solució analítica als mateixos punts i al mateix instant
    x = np.linspace(0, 1, N)
    T_ana = T_analitica(x, t_a)

    fig_sol, ax_sol = plt.subplots()     # figura 1: solucions
    fig_err, ax_err = plt.subplots()     # figura 2: errors

    # L'analítica es dibuixa amb punts per distingir-la de les numèriques
    ax_sol.plot(x[::5], T_ana[::5], "ko", markersize=3, label="Analítica")

    for factor, etiqueta, linia in casos:
        dt = factor * dx**2
        x, T_cn = resol_cn(N, dt, t_a)
        err = error(T_cn, T_ana)

        print(f"dt = {factor} dx^2:  error màxim = {err.max():.2e}")

        ax_sol.plot(x, T_cn, linia, label=etiqueta)
        # A les fronteres l'error és exactament 0 i log(0) no existeix: les ometem
        ax_err.semilogy(x[1:-1], err[1:-1], linia, label=etiqueta)

    ax_sol.set_xlabel(r"$\hat{x}$")
    ax_sol.set_ylabel(r"$\hat{T}(\hat{x}, \hat{t}_a)$")
    ax_sol.legend(loc="lower center")

    ax_err.set_xlabel(r"$\hat{x}$")
    ax_err.set_ylabel(r"$|\hat{T}_\mathrm{CN} - \hat{T}_\mathrm{ana}|$")
    ax_err.legend(loc="lower center")

    desa(fig_sol, "cn_solucio")
    desa(fig_err, "cn_error")
    plt.show()