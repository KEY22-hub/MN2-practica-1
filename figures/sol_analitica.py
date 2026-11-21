
"""
S'ha resolt analíticament l'equació adimensionalitzada utilitzant l'annex del guió. La solució analítica conté un sumatori infinit que ve d'expandir en serie de fourier 
la funció solució i les seves derivades, es trunca aquest sumatori al terme n-èssim.

equació adimensionalitzada i condicions imposades:
    dT/dt = d2T/dx2 + 1,   x en [0, 1],   T = 0 a les fronteres i a t = 0

Solució proposada al guió:
    T(x, t) = sum_{n senar} 4/(n pi) * (1 - exp(-n^2 pi^2 t)) / (n^2 pi^2) * sin(n pi x)
"""




import numpy as np

from parametres import (
    N,
    T_hat_limit,
    T_hat_mort,
    a_celsius,
    a_metres,
    a_segons,
    t_a,
    x_malalt_f,
    x_malalt_i,
)


#definim la funció temperatura que depen del punt x del teixit, l'instant t i el truncament que es fa a la serie infinita
def T_analitica(x, t, n_termes=1001):  
    
    x = np.asarray(x, dtype=float)
    T = np.zeros_like(x)

    # Només els n senars, els parells s'anul·len tots a la suma de fourier
    for n in range(1, n_termes + 1, 2):
        b_n = 4 / (n * np.pi)
        terme_temporal = (1 - np.exp(-(n**2) * np.pi**2 * t)) / (n**2 * np.pi**2)
        T += b_n * terme_temporal * np.sin(n * np.pi * x)

    return T


#definim una funció error per comparar la solució analítica amb el mètodes numerics que s'empraran per aproximar-la
def error(T_num, T_ana):
    #Error absolut punt a punt entre una solució numèrica i l'analítica.#
    return np.abs(T_num - T_ana)


# En aquest apartat s'evalua l'error comès per truncament a la solució analítica
if __name__ == "__main__":            
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 101)
    t_a = 0.025


    """
    Per evaluar l'error de truncament s'utilitza el fet que la solució convergeix a x(1-x)/2 quan t tendeix a infinit.
    A la solució truncada se li assigna un t suficientment gran i s'evalua la diferencia màxima entre aquesta solució trunada i x(1-x)/2. Triant un t prou gran es pot 
    considerar que l'error l'origina exclusivament el truncament.
    A continuació es grafica aquest error en funció del numero de termes de fourier emprats
    """
    t_gran = 50.0
    T_est = x * (1.0 - x) / 2.0

    llista_n = [1, 3, 5, 11, 21, 51, 101, 251, 501, 1001]

    llista_error = [
      error(T_analitica(x, t_gran, n_termes=n_t), T_est).max()
      for n_t in llista_n
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
    llista_n,
    llista_error,
    marker="o",
    label=r"$\max_{\hat{x}} |\hat{T}_{n}(\hat{x}, \hat{t}_{\text{gran}}) - \hat{T}_{\text{est}}(\hat{x})|$",
    )



ax.set_xlabel("Nombre de termes de Fourier")
ax.set_ylabel("Error absolut màxim")
ax.set_title(
    f"error de truncament (t = {t_gran})"
)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()




# Gràfica 3D de l'evolució temporal de la temperatura en cada punt (x) del teixit
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
    
