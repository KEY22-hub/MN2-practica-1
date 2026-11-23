import numpy as np
from gauss_seidel import gauss_seidel
from parametres import N, t_a


def iteracions_mitjanes(metode, gamma, des_de_zero):
    # Nombre mitjà d'iteracions de Gauss-Seidel per pas de temps.
    dx = 1 / (N - 1)
    n_passos = round(t_a / (gamma * dx**2))
    dt = t_a / n_passos
    g = dt / dx**2

    T = np.zeros(N)
    if metode == "implicit":
        a = np.full(N - 2, -g)
        b = np.full(N - 2, 1 + 2 * g) # Definim "la matriu tridiagonal" del sistema amb la que fariem Gauss-Seidel 
    else:  # El mateix per Crank-Nicolson
        a = np.full(N - 2, -g / 2)
        b = np.full(N - 2, 1 + g)

    iteracions = [] # Llista on guardarem el nombre d'iteracions de cada pas per calcular després el nombre mitjà d'iteracions.
    for _ in range(n_passos):
        if metode == "implicit":
            d = T[1:-1] + dt
        else:
            d = (g / 2) * T[:-2] + (1 - g) * T[1:-1] + (g / 2) * T[2:] + dt
        x0 = np.zeros(N - 2) if des_de_zero else T[1:-1]
        T[1:-1], n_iter = gauss_seidel(a, b, a, d, x0)
        iteracions.append(n_iter)
    return np.mean(iteracions)


if __name__ == "__main__":
    for metode in ["implicit", "cn"]:
        for gamma in [0.5, 1.0]:
            prev = iteracions_mitjanes(metode, gamma, des_de_zero=False)
            zero = iteracions_mitjanes(metode, gamma, des_de_zero=True)
            print(f"{metode} gamma = {gamma}:  pas anterior = {prev:.1f}   zero = {zero:.1f}")