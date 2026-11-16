

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 101
    t_a = 0.025
    dx = 1.0 / (N - 1)

    # Els dos casos que demana l'enunciat
    casos = {
        r"$\Delta\hat{t} = (\Delta\hat{x})^2$": dx**2,
        r"$\Delta\hat{t} = 0.5(\Delta\hat{x})^2$": 0.5 * dx**2,
    }

    # Si el company ja ha fet analitica.py, també dibuixem l'error.
    try:
        from analitica import T_analitica
        hi_ha_analitica = True
    except ImportError:
        hi_ha_analitica = False
        print("Encara no hi ha analitica.py: només es dibuixa la solució.")

    if hi_ha_analitica:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, ax1 = plt.subplots(figsize=(5, 4))

    for etiqueta, dt in casos.items():
        x, T = resol_cn(N, dt, t_a)
        print(f"{etiqueta}: T màxima = {T.max():.6f} a x = {x[T.argmax()]:.2f}")
        ax1.plot(x, T, label=etiqueta)

        if hi_ha_analitica:
            T_ana = T_analitica(x, t_a)
            ax2.plot(x, np.abs(T - T_ana), label=etiqueta)

    ax1.set_xlabel(r"$\hat{x}$")
    ax1.set_ylabel(r"$\hat{T}$")
    ax1.legend()

    if hi_ha_analitica:
        ax2.set_xlabel(r"$\hat{x}$")
        ax2.set_ylabel(r"$|\hat{T}_{\mathrm{CN}} - \hat{T}_{\mathrm{an}}|$")
        ax2.legend()

    plt.tight_layout()
    plt.show()