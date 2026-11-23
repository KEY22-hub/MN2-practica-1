"""
Solució del problema: durant quant de temps cal aplicar els 40 V?

Fem servir Crank-Nicolson amb la font encesa durant els primers n_on passos
i apagada després. Continuem simulant després d'apagar, perquè la calor del
centre es continua difonent cap al teixit sa.

Condicions:
    - el teixit sa no ha d'arribar mai a 50 ºC
    - cap punt ha d'arribar mai a 80 ºC
Com més temps escalfem, més temps passa la zona malalta per sobre de 50 ºC, 
que és el es necessita pel tractament, així que l'òptim és el temps d'aplicació 
més llarg que compleix les condicions.
"""

import numpy as np
from gauss_seidel import gauss_seidel
from parametres import (T_hat_mort, T_hat_limit, x_malalt_i, x_malalt_f,
                        a_celsius, a_segons, L)

N = 201        # amb N = 201, x = 0.375 i 0.625 són punts de la malla
GAMMA = 1.0
dx = 1 / (N - 1)
dt = GAMMA * dx**2

x = np.linspace(0, 1, N)
# El petit marge (1e-9) evita problemes d'arrodoniment a les vores de la zona malalta.
# Els ordinadors guarden els decimals amb petites imprecisions, 
# Pot passar que el punt que hauria de valer exactament 0.375 es guardi com a 
# 0.37499999999999994. Llavors x >= 0.375 donaria False, i un punt que és malalt 
# el comptaríem com a sa.
malalt = (x >= x_malalt_i - 1e-9) & (x <= x_malalt_f + 1e-9)
sa = ~malalt #Inverteix la condició de malalt, és a dir, es com el complementari, tot el que no es malalt.


def simula(n_on, aturar_si_sa_supera=False):
    """
    Crank-Nicolson amb la font encesa durant els n_on primers passos i fins que 
    un cop la font apagada tota la paret ha tornat per sota de 50 ºC.
    (no passa res interessant després). Si aturar_si_sa_supera és True, para també en el 
    moment que el teixit sa arriba a 50 °C (només volem saber si el cas és vàlid, 
    i així anem més ràpid).

    Retorna els instants t (adimensionals) i la matriu H, amb una fila per
    instant i una columna per punt de la malla.
    """
    n = N - 2 # Les fronteres no són incògnites
    a = np.full(n, -GAMMA / 2)
    b = np.full(n, 1 + GAMMA) # Definim "la matriu tridiagonal" del sistema

    T = np.zeros(N)
    historia = [T.copy()] # .copy es com fer una foto a la llista T a cada instant
    i = 0
    while True:
        font = dt if i < n_on else 0.0 # la font només actua mentre està encesa
        d = (GAMMA / 2) * T[:-2] + (1 - GAMMA) * T[1:-1] + (GAMMA / 2) * T[2:] + font
        T[1:-1], _ = gauss_seidel(a, b, a, d, x0=T[1:-1])
        i += 1
        historia.append(T.copy())

        if aturar_si_sa_supera and T[sa].max() >= T_hat_mort:
            break 
        # La condició de aturar_si_sa_supera és la nostra manera de activar o desactivar 
        # la condició de T[sa].max >= T_Hat_mort. Si la definim True fem que aquesta condició
        # pugui entrar en joc, cosa que ens serveix per filtrar els casos vàlids.
        # Si la deixem en False aquesta condició no es complirà mai i la simulació segueix.
        if i >= n_on and T.max() < T_hat_mort:
            break # Aquesta condició es compleix sempre, diu que si la font és apagada i 
                  # tot el teixit es troba per sota 50 ºC, que parem la simulació.

    t = np.arange(len(historia)) * dt # Tots els instants de temps en unitats de dt.
                                      # Si s'han fet 1000 pasos, sera t = [0, dt, 2dt..., 999dt]
    return t, np.array(historia) # El np.array(historia) es una matriu que té per files els
                                 # instants de temps i per columnes els punts del teixit


def es_valid(n_on): # Només retorna la matriu H=np.array(historia) si el cas es vàlid
    #True si, encenent la font n_on passos, es compleixen les dues condicions.
    _, H = simula(n_on, aturar_si_sa_supera=True)
    return H[:, sa].max() < T_hat_mort and H.max() < T_hat_limit


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from estil_figures import desa

    # Cerquem el temps òptim per bisecció sobre el nombre de passos
    # low: un nombre de passos vàlid (0 passos sempre ho és)
    # high: un nombre de passos que segur que no és vàlid
    low = 0
    high = round(0.03 / dt) # Un valor de passos que fent proves hem vist que no compleix les condicions
    while high - low > 1: # Quan high - low = 1, voldrà dir que son números de passos consecutius (per exemple, 700 i 701 passos, ja no hi ha punt mig)
        mig = (low + high) // 2 # Divisió entera //, no podem tenir 2.5 passos
        if es_valid(mig): # Mirem si encenent la font fins al nombre de passos que valgui mig es valid
            low = mig
        else:
            high = mig
    n_opt = low
    t_off = n_opt * dt

    # Simulació completa amb el temps òptim
    t, H = simula(n_opt)
    dins = (H[:, malalt] >= T_hat_mort) & (H[:, malalt] < T_hat_limit) # Regió de teixit malalt que reb tractament (entre 50 i 80 ºC)
    j_centre = N // 2
    j_sa = np.where(sa & (x < 0.5))[0][-1] # punt sa més proper a la zona malalta

    print(f"t_off = {t_off:.5f} (adimensional) = {a_segons(t_off):.1f} s")
    print(f"Màxim al teixit sa:   {a_celsius(H[:, sa].max()):.2f} °C")
    print(f"Màxim a tota la paret: {a_celsius(H.max()):.2f} °C")
    print(f"Temps que el centre passa entre 50 i 80 °C: "
          f"{a_segons(dins[:, malalt.sum() // 2].sum() * dt):.1f} s")
    print(f"Fracció de la zona malalta que arriba a 50 °C: {dins.any(axis=0).mean():.2f}")

    # --- Figura 1: perfils de temperatura a diversos instants ---
    fig1, ax1 = plt.subplots()
    for frac in [0.5, 0.8, 1.0]: #calculem 3 casos, un on la meitat del teixit reb el tractament, un on el 80% i un altre on el 100%, el cas optim
        k = round(frac * n_opt)
        ax1.plot(L * 100 * x, a_celsius(H[k]), label=f"$t = {a_segons(t[k]):.0f}$ s")
    ax1.axhline(50, color="k", ls="--", lw=0.8)
    ax1.axvspan(L * 100 * x_malalt_i, L * 100 * x_malalt_f, color="0.85", zorder=0)
    ax1.set_xlabel(r"$x$ (cm)")
    ax1.set_ylabel(r"$T$ ($^\circ$C)")
    ax1.legend(loc="lower center")
    desa(fig1, "problema_perfils")

    # --- Figura 2: evolució al centre i al punt sa més proper ---
    fig2, ax2 = plt.subplots()
    ax2.plot(a_segons(t), a_celsius(H[:, j_centre]), label=r"centre ($x = 1$ cm)")
    ax2.plot(a_segons(t), a_celsius(H[:, j_sa]), "--",
             label=f"teixit sa ($x = {L * 100 * x[j_sa]:.2f}$ cm)")
    ax2.axhline(50, color="k", ls=":", lw=0.8)
    ax2.axvline(a_segons(t_off), color="0.5", lw=0.8)
    ax2.set_xlabel(r"$t$ (s)")
    ax2.set_ylabel(r"$T$ ($^\circ$C)")
    ax2.legend(loc="lower right")
    desa(fig2, "problema_evolucio")

    plt.show()