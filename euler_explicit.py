"""
En aquest apartat es resol l'equació de la calor adimensional  dT/dt = d2T/dx2 + 1  amb el mètode
d'Euler explícit, concretament amb la derivada temporal per la dreta.
"""

import numpy as np
import matplotlib.pyplot as plt


from parametres import N, t_a
from sol_analitica import T_analitica, error
from estil_figures import desa


# Aplicació del mètode d'Euler explícit. Rep el nombre de punts (N), el pas de temps (dt) i el temps
# final (t_a).
def resol_exp(N, dt, t_a):

    # Distància entre dos punts veïns. Hi ha N punts entre 0 i 1, per tant
    # N - 1 espais entre ells.
    dx = 1 / (N - 1)

    # Nombre de passos de temps per arribar a t_a. El quocient pot no ser
    # enter, i el nombre de passos ho ha de ser,
    # per això s'arrodoneix.
    n_passos = round(t_a / dt)

    # Recalculem dt perquè n_passos passos arribin exactament a t_a.
    dt = t_a / n_passos

    # gamma = dt / dx^2. És el número que decideix si el mètode és estable
    # (ha de ser com a màxim 0.5).
    gamma = dt / dx**2

    # Llista de les N posicions de 0 a 1, equiespaiades.
    x = np.linspace(0, 1, N)

    # Condició inicial: tots els punts a temperatura 0, que en variables
    # adimensionals vol dir temperatura corporal.
    # T_vella guarda la temperatura del pas de temps anterior.
    T_vella = np.zeros(N)

    # Aquest bucle fa un pas de temps a cada volta. 
    for pas in range(n_passos):

        # Llista nova per a les temperatures del pas que calculem ara.
        # Comença tota a 0. 
        T_nova = np.zeros(N)

        # Aquest bucle recorre els punts interiors, del 1 al N - 2. El 0 i
        # el N - 1 són les fronteres i no es calculen.
        for j in range(1, N - 1):

            # Fórmula d'Euler explícit (Eq. 9). 
            T_nova[j] = T_vella[j] + gamma * (T_vella[j + 1] - 2 * T_vella[j] + T_vella[j - 1]) + dt

        # Quan s'han calculat tots els punts, la llista nova passa a ser la
        # vella per fer el pas següent. Per això fem servir dues llistes:
        # cada valor nou ha de dependre només dels valors vells.
        T_vella = T_nova

    # Retornem les posicions i la temperatura final (a t = t_a).
    return x, T_vella


# Tot el que hi ha dins d'aquest if només s'executa quan corres aquest
# fitxer directament. Si un altre fitxer importa resol(), no s'executa.
if __name__ == "__main__":

    # Distància entre punts veïns, per calcular els dt dels tres casos.
    dx = 1 / (N - 1)

    # Els tres casos de l'enunciat: dt = 0.51 dx^2, 0.49 dx^2 i 0.25 dx^2.
    # Cada crida retorna x (les posicions) i la temperatura a t = t_a.
    # La x és la mateixa als tres casos, per això es torna a escriure.
    x, T_051 = resol_exp(N, 0.51 * dx**2, t_a)
    x, T_049 = resol_exp(N, 0.49 * dx**2, t_a)
    x, T_025 = resol_exp(N, 0.25 * dx**2, t_a)

    # Solució exacta als mateixos punts x i al mateix temps t_a.
    T_ana = T_analitica(x, t_a)

    # Error de cada cas: diferència en valor absolut, punt a punt, entre
    # la solució numèrica i l'analítica.
    err_051 = error(T_051, T_ana)
    err_049 = error(T_049, T_ana)
    err_025 = error(T_025, T_ana)

    # Mostrem per pantalla l'error més gran de cada cas, per comprovar
    # sense haver de mirar la gràfica.
    print("error màxim 0.51:", err_051.max())
    print("error màxim 0.49:", err_049.max())
    print("error màxim 0.25:", err_025.max())

    # Creem una figura amb 3 gràfiques una al costat de l'altra. fig és tota
    # la figura i ax1, ax2, ax3 són les tres gràfiques. 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.8, 2.4))

    # Gràfica (a): els dos casos estables sobre la solució exacta.
    ax1.plot(x, T_ana, "k-", label=r"$\hat{T}_\mathrm{ana}$")
    ax1.plot(x, T_049, "s", color="C0", markevery=5, ms=3, label=r"$\gamma\approx 0.49$")
    ax1.plot(x, T_025, "o", color="C2", markevery=5, ms=3, label=r"$\gamma = 0.25$")
    ax1.set_xlabel(r"$\hat{x}$")
    ax1.set_ylabel(r"$\hat{T}(\hat{x}, \hat{t}_a)$")
    ax1.legend(loc="lower center")

    ax1.text(0.04, 0.93, "(a)", transform=ax1.transAxes, ha="left", va="top")

    # Gràfica (b): el cas inestable a part. Surt amb valors d'uns +-2,
    # que aplanarien la gràfica (a), per això no hi va al costat.
    # set_ylim fixa els límits de l'eix vertical perquè la llegenda,
    # a baix, no tapi cap punt.
    ax2.plot(x, T_ana, "k-", label=r"$\hat{T}_\mathrm{ana}$")
    ax2.plot(x, T_051, "^", color="C3", markevery=5, ms=3, label=r"$\gamma\approx 0.51$")
    ax2.set_xlabel(r"$\hat{x}$")
    ax2.set_ylabel(r"$\hat{T}(\hat{x}, \hat{t}_a)$")
    ax2.set_ylim(-3.6, 2.6)
    ax2.legend(loc="lower center", ncol=2, columnspacing=1.0, handletextpad=0.4)
    ax2.text(0.04, 0.93, "(b)", transform=ax2.transAxes, ha="left", va="top")

    # Gràfica (c): els tres errors amb l'eix vertical en escala logarítmica
    # (semilogy), perquè van de 1e-7 a 2 i en escala normal no es veurien.
    # Es dibuixa de x[1:-1] (del segon punt al penúltim) perquè a les
    # fronteres l'error és exactament 0 i el 0 no es pot dibuixar en
    # escala logarítmica.
    ax3.semilogy(x[1:-1], err_051[1:-1], color="C3", label=r"$\gamma\approx 0.51$")
    ax3.semilogy(x[1:-1], err_049[1:-1], color="C0", label=r"$\gamma\approx 0.49$")
    ax3.semilogy(x[1:-1], err_025[1:-1], color="C2", label=r"$\gamma = 0.25$")
    ax3.set_xlabel(r"$\hat{x}$")
    ax3.set_ylabel(r"$|\hat{T}_\mathrm{num}-\hat{T}_\mathrm{ana}|$")
    ax3.set_ylim(1e-7, 1e2)
    ax3.legend(loc="center right")
    ax3.text(0.04, 0.93, "(c)", transform=ax3.transAxes, ha="left", va="top")

    # Desa la figura com a figures/explicit_solucio_error.pdf
    desa(fig, "explicit_solucio_error")

    plt.show()