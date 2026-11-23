"""
Mètode de Gauss-Seidel per a sistemes lineals TRIDIAGONALS.

Resol A x = d, definim parts de A:
    a[j] : subdiagonal   (element A[j, j-1]); a[0] no es fa servir
    b[j] : diagonal      (element A[j, j])
    c[j] : superdiagonal (element A[j, j+1]); c[-1] no es fa servir

"""

import warnings
import numpy as np


def gauss_seidel(a, b, c, d, x0, tol=1e-10, max_iter=10000):
    """
    Els paràmetres que fem servir són:

    a, b, c : arrays de mida n amb la subdiagonal, la diagonal i la superdiagonal.
    d       : array de mida n, terme independent.
    x0      : array de mida n, punt de partida (a la pràctica: T del pas de temps anterior).
    tol     : tolerància, criteri d'aturada, max_j |x_j^(k+1) - x_j^(k)| < tol.
    max_iter: nombre màxim d'iteracions.

    El que volem que retorni el codi:
    x      : array de mida n amb la solució.
    n_iter : nombre d'iteracions fetes.
    """
    n = len(b)
    x = np.array(x0, dtype=float)  # còpia: no modifiquem el x0 original

    for k in range(1, max_iter + 1):
        canvi_max = 0.0

        for j in range(n):
            suma = d[j]
            if j > 0:
                suma -= a[j] * x[j - 1]   # x[j-1] ja és el valor NOU (és la idea de Gauss-Seidel)
            if j < n - 1:
                suma -= c[j] * x[j + 1]   # x[j+1] encara és el valor VELL
            x_nou = suma / b[j]

            canvi_max = max(canvi_max, abs(x_nou - x[j]))
            x[j] = x_nou

        if canvi_max < tol:
            return x, k

    warnings.warn(f"Gauss-Seidel no ha convergit en {max_iter} iteracions "
                  f"(últim canvi = {canvi_max:.2e})")
    return x, max_iter


def multiplica_tridiagonal(a, b, c, x): # La vam definir per fer proves, no es fa servir
    """Calcula A x per a una matriu tridiagonal (només per fer proves)."""
    n = len(b)
    y = b * x
    y[1:] += a[1:] * x[:-1]
    y[:-1] += c[:-1] * x[1:]
    return y
