
#aplicació del mètode d'Euler implicit per aproximar numèricament la solució
#la edp que volem resoldre és df/dt = d^2f/dx^2 + 1
#aproximem les derivades com df/dt=(f_(m+1) - f_(m))/(dt)  i   d^2f/dx^2 = (f(i-1)-2f(i)+f(i+1))/(dx^2)
#on els index m es refereixen als passos temporals i els index i als espacials i reordenem per poder evaluar la derivada espacial en el pas de temps següent per cada iteració



import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


from parametres import N, t_a
from sol_analitica import T_analitica, error
from estil_figures import desa


# definim Rep el nombre de punts (N), el pas de temps (dt) i el temps
# final (t_a).
def resol(N, dt, t_a):

    # Distància entre punts. Hi ha N punts entre 0 i 1, per tant
    # N - 1 espais entre ells.
    dx = 1 / (N - 1)

    # Nombre de passos de temps per arribar a t_a. El quocient pot no ser
    # enter, i el nombre de passos ho ha de ser,
    # per això s'arrodoneix.
    n_passos = round(t_a / dt)

    # Recalculem dt perquè n_passos passos arribin exactament a t_a.
    dt = t_a / n_passos

    # r = dt / dx^2. A l'Euler explícit s'ha de mantenir per sota de 0.5, però el mètode implícit
    # és incondicionalment estable per a qualsevol valor positiu d'r.
    r = dt / dx**2

    # Llista de les N posicions de 0 a 1, equiespaiades.
    x = np.linspace(0, 1, N)

    # Condició inicial: tots els punts a temperatura adimensional 0 (temperatura corporal)
    # T_vella guarda la temperatura del pas de temps anterior.
    T_vella = np.zeros(N)

    # Nombre de punts interiors (del 1 al N - 2). Els extrems 0 i N - 1
    # queden fixats a 0 per les condicions de contorn.
    M = N - 2

    # Definim la matriu tridiagonal a partir de l'equació: 
    #   -r * T_{j-1}^{k+1} + (1 + 2*r) * T_j^{k+1} - r * T_{j+1}^{k+1} = T_j^k + dt
    # que ve d'aplicar l'aproximació en les derivades de la edp i reordenar.
    ab = np.zeros((3, M))
    ab[0, 1:] = -r     #supradiagonal
    ab[1, :] = 1 + 2 * r    #diagonal pura
    ab[2, :-1] = -r   #subdiagonal

    # Aquest bucle fa que s'avanci un pas temporal a cada iteració.
    for pas in range(n_passos):

        # Termes independents per als punts interiors a partir del pas anterior.
        rhs = T_vella[1:-1] + dt

        # Resolem el sistema lineal tridiagonal i guardem el nou perfil de temperatures.
        T_nova = np.zeros(N)
        T_nova[1:-1] = solve_banded((1, 1), ab, rhs)
        T_vella = T_nova

    # Retornem les posicions i la temperatura final (a t = t_a).
    return x, T_vella



if __name__ == "__main__":

    # Distància entre punts veïns, per calcular els dt dels tres casos.
    dx = 1 / (N - 1)

    # Tres valors de dt / dx^2 per comparar el comportament del mètode implícit
    # (incloent-hi passos temporals més grans on l'explícit seria inestable).
    x, T_025 = resol(N, 0.25 * dx**2, t_a)
    x, T_050 = resol(N, 0.50 * dx**2, t_a)
    x, T_200 = resol(N, 2.00 * dx**2, t_a)

    # Solució exacta als mateixos punts x i al mateix temps t_a.
    T_ana = T_analitica(x, t_a)

    # Error de cada cas: diferència en valor absolut, punt a punt, entre
    # la solució numèrica i l'analítica.
    err_025 = error(T_025, T_ana)
    err_050 = error(T_050, T_ana)
    err_200 = error(T_200, T_ana)

    # Mostrem per pantalla l'error més gran de cada cas, per comprovar
    # sense haver de mirar la gràfica.
    print("error màxim 0.25:", err_025.max())
    print("error màxim 0.50:", err_050.max())
    print("error màxim 2.00:", err_200.max())

    # Creem una figura amb 3 gràfiques una al costat de l'altra. fig és tota
    # la figura i ax1, ax2, ax3 són les tres gràfiques.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.8, 2.4))

    # Gràfica (a): casos amb dt / dx^2 petit i moderat sobre la solució exacta.
    ax1.plot(x, T_ana, "k-", label=r"$\hat{T}_\mathrm{ana}$")
    ax1.plot(x, T_050, "s", color="C0", markevery=5, ms=3, label=r"$\gamma = 0.50$")
    ax1.plot(x, T_025, "o", color="C2", markevery=5, ms=3, label=r"$\gamma = 0.25$")
    ax1.set_xlabel(r"$\hat{x}$")
    ax1.set_ylabel(r"$\hat{T}(\hat{x}, \hat{t}_a)$")
    ax1.legend(loc="lower center")
    ax1.text(0.04, 0.93, "(a)", transform=ax1.transAxes, ha="left", va="top")

    # Gràfica (b): cas amb dt / dx^2 gran a part, per observar-ne el perfil
    # i l'efecte d'un pas temporal més llarg sobre la solució exacta.
    ax2.plot(x, T_ana, "k-", label=r"$\hat{T}_\mathrm{ana}$")
    ax2.plot(x, T_200, "^", color="C3", markevery=5, ms=3, label=r"$\gamma = 2.00$")
    ax2.set_xlabel(r"$\hat{x}$")
    ax2.set_ylabel(r"$\hat{T}(\hat{x}, \hat{t}_a)$")
    ax2.legend(loc="lower center", ncol=2, columnspacing=1.0, handletextpad=0.4)
    ax2.text(0.04, 0.93, "(b)", transform=ax2.transAxes, ha="left", va="top")

    # Gràfica (c): els tres errors amb l'eix vertical en escala logarítmica
    # (semilogy) als punts interiors x[1:-1] (a les fronteres l'error és 0).
    ax3.semilogy(x[1:-1], err_200[1:-1], color="C3", label=r"$\gamma = 2.00$")
    ax3.semilogy(x[1:-1], err_050[1:-1], color="C0", label=r"$\gamma = 0.50$")
    ax3.semilogy(x[1:-1], err_025[1:-1], color="C2", label=r"$\gamma = 0.25$")
    ax3.set_xlabel(r"$\hat{x}$")
    ax3.set_ylabel(r"$|\hat{T}_\mathrm{num}-\hat{T}_\mathrm{ana}|$")
    ax3.set_ylim(1e-7, 1e2)
    ax3.legend(loc="center right")
    ax3.text(0.04, 0.93, "(c)", transform=ax3.transAxes, ha="left", va="top")

    # Desa la figura com a figures/implicit_solucio_error.pdf
    desa(fig, "implicit_solucio_error")

    plt.show()
