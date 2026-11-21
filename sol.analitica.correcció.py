# S'ha resolt analíticament utilitzant l'annex del guió. La solució analítica conté un sumatori
# infinit que ve d'expandir en sèrie de Fourier de sinus la font i la condició inicial; es trunca
# aquest sumatori al terme n_termes-èssim.
#
# Equació adimensional (informació addicional del professor, Eq. 7):
#     dT/dt = d2T/dx2 + 1,   x en [0, 1],   T = 0 a les fronteres i a t = 0
# Comparant amb l'annex: q(x, t) = 1 i beta = 0, perquè T_hat = (T - Tc)/T0 val 0 quan T = Tc.

import numpy as np


def T_analitica(x, t, n_termes=1001):
    """
    Solució analítica truncada:

        T(x, t) = sum_{n senar} 4/(n pi) * (1 - exp(-n^2 pi^2 t)) / (n^2 pi^2) * sin(n pi x)

    x        : array de posicions (adimensional).
    t        : instant (adimensional), un sol número.
    n_termes : on es trunca la sèrie. Amb 1000 termes l'error de truncament és ~1e-9,
               molt per sota de l'error dels mètodes numèrics (~1e-6).
    """
    x = np.asarray(x, dtype=float)
    T = np.zeros_like(x)

    # Només els n senars: per als parells, b_n = 2(1 - (-1)^n)/(n pi) = 0
    for n in range(1, n_termes + 1, 2):
        b_n = 4 / (n * np.pi)
        terme_temporal = (1 - np.exp(-(n**2) * np.pi**2 * t)) / (n**2 * np.pi**2)
        T += b_n * terme_temporal * np.sin(n * np.pi * x)

    return T


def error(T_num, T_ana):
    """Error absolut punt a punt entre una solució numèrica i l'analítica."""
    return np.abs(T_num - T_ana)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 101)
    t_a = 0.025

    # --- Comprovacions ---
    T = T_analitica(x, t_a)
    print("Fronteres (han de ser 0):", T[0], T[-1])
    print("t = 0 (ha de ser 0):", np.abs(T_analitica(x, 0.0)).max())

    T_est = T_analitica(x, 50.0)                  # t molt gran
    print("Diferència amb l'estacionari x(1-x)/2:", np.abs(T_est - x * (1 - x) / 2).max())

    # Convergència de la sèrie: quants termes cal sumar?
    referencia = T_analitica(x, t_a, n_termes=20001)
    for n_t in [11, 101, 1001, 10001]:
        dif = np.abs(T_analitica(x, t_a, n_termes=n_t) - referencia).max()
        print(f"n_termes = {n_t:6d}: error de truncament = {dif:.1e}")

    # --- Gràfica 3D de l'evolució (la de la companya/company) ---
    t = np.linspace(0, t_a, 100)
    X, T_mesh = np.meshgrid(x, t)
    F = np.array([T_analitica(x, ti) for ti in t])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, T_mesh, F, cmap="viridis")
    ax.set_xlabel(r"$\hat{x}$")
    ax.set_ylabel(r"$\hat{t}$")
    ax.set_zlabel(r"$\hat{T}$")
    plt.show()