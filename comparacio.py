import numpy as np
import matplotlib.pyplot as plt

from parametres import N, t_a
from sol_analitica import T_analitica, error
from estil_figures import desa
from euler_explicit import resol_exp
from euler_imp import resol_imp
from crank_nicolson import resol_cn

# Error punt a punt amb gamma = 0.5 per als tres mètodes:

dx = 1 / (N - 1)

# Solució analítica als N punts de la malla i a t = t_a. És la mateixa per a
# tots els càlculs, així que la calculem un sol cop.
x = np.linspace(0, 1, N)
T_ana = T_analitica(x, t_a)

# Els tres mètodes amb el nom per a la llegenda, el color i el marcador (com surtiran als gràfics).
# Així els dos bucles de sota poden recórrer els tres mètodes de la mateixa manera.
metodes = [
    ("Euler explícit", resol_exp, "C3", "^"),
    ("Euler implícit", resol_imp, "C0", "s"),
    ("Crank-Nicolson", resol_cn, "C2", "o"),
]

# Figura 1: error punt a punt dels tres mètodes amb el mateix dt.
# gamma = 0.5 és el pas més gran amb què l'explícit encara és estable.

gamma_comu = 0.5
dt_comu = gamma_comu * dx**2

fig1, ax1 = plt.subplots()

for nom, resol, color, marcador in metodes:
    _, T_num = resol(N, dt_comu, t_a)
    err = error(T_num, T_ana)
    print(f"gamma = {gamma_comu}  {nom}  error màxim = {err.max():.2e}")

    # A les fronteres l'error és exactament 0 i log(0) no existeix, les hem d'ometre
    ax1.semilogy(x[1:-1], err[1:-1], color=color, label=nom)

ax1.set_xlabel(r"$\hat{x}$")
ax1.set_ylabel(r"$|\hat{T}_\mathrm{num} - \hat{T}_\mathrm{ana}|$")
ax1.legend(loc="lower center")

desa(fig1, "comparacio_error")


# Bucle sobre una llista de gammes per l'error màxim de cada mètode
# (l'explícit només si gamma <= 0.5)

# Figura 2: error màxim de cada mètode en funció del pas de temps.
# Per a cada gamma de la llista, resolem amb els tres mètodes i ens quedem
# amb l'error més gran de tota la malla.

gammes = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

fig2, ax2 = plt.subplots()

for nom, resol, color, marcador in metodes:
    dts = []
    errors_max = []

    for gamma in gammes:
        # L'explícit és inestable per sobre de gamma = 0.5 així que no el calculem
        if resol is resol_exp and gamma > 0.5:
            continue # Li estem dient: si en aquesta iteració del bucle et toca fer euler explicit 
                     # i la gamma es més gran que 0.5, saltat aquesta iteració i segueix amb lo següent

        dt = gamma * dx**2
        _, T_num = resol(N, dt, t_a)

        dts.append(dt)
        errors_max.append(error(T_num, T_ana).max())

    ax2.loglog(dts, errors_max, "-", color=color, marker=marcador, ms=4, label=nom) # Fem servir escala logarítmica a ambdós eixos per evitar que es vegin les línes colapsades/molt juntes

ax2.set_xlabel(r"$\Delta\hat{t}$")
ax2.set_ylabel(r"$\max_j |\hat{T}_\mathrm{num} - \hat{T}_\mathrm{ana}|$")
ax2.legend(loc="upper left")

desa(fig2, "comparacio_convergencia")

plt.show()