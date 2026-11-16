
"""
Totes les constants fisiques i les escales de la practica s'importen d'aqui, per posar-les a un altre fitxer, cal fer:
    from parametres import kappa, N, t_a

Unitats: SI (m, s, kg, K, W, S, V), excepte les temperatures, en graus Celsius.
"""
import math


# Propietats termoelèctriques mitjanes del teixit cardiac 

cv = 3698.0        # calor especific       [J/(kg K)]
rho = 1081.0       # densitat              [kg/m^3]
kappa = 0.56       # conductivitat termica [W/(m K)]
sigma = 0.472      # conductivitat electrica [S/m]


# Dades del problema 

L = 0.02               # gruix de la paret entre ventricles [m]
gruix_malalt = 0.005   # gruix de la zona malalta [m]
d = L                  # distancia entre electrodes (plaques) [m]

# Potencials elèctics 

V0 = 40.0                       # senyal donada [V]
V_ef = V0 / math.sqrt(2)        # potencial efectiu [V]

# Potencia per unitat de volum per efecte Joule: P = sigma * E^2, E = V_ef / d

Pext = sigma * (V_ef / d) ** 2  # [W/m^3]


# Temperatures (condicions del tractament)

Tc = 36.5          # temperatura corporal (inicial i de frontera) [C]
T_mort = 50.0      # a partir d'aqui hi ha mort cel.lular [C]
T_limit = 80.0     # no s'ha d'arribar-hi (risc de trombosi) [C]


# Adimensionalització per fer la normalització 
#   x_hat = x / L,  t_hat = t / t0,  T_hat = (T - Tc) / T0

alpha = kappa / (cv * rho)      # difusivitat termica [m^2/s]
t0 = L ** 2 / alpha             # temps caracteristic de difusio [s]
T0 = Pext * L ** 2 / kappa      # escala de temperatura [K]

# Zona malalta centrada, en coordenada adimensional x_hat (això serveix per l'evaluació del problema )
x_malalt_i = 0.5 - gruix_malalt / (2 * L)   # 0.375 
x_malalt_f = 0.5 + gruix_malalt / (2 * L)    # 0.625


# Condicions de temperatura en variables adimensionals
T_hat_mort = (T_mort - Tc) / T0
T_hat_limit = (T_limit - Tc) / T0


# Discretització (per passar d'una funció continua de la temperatura a una discreta)

N = 101                  # punts de mallat espacial
dx = 1.0 / (N - 1)       # amplada del mallat espacial (adimensional)
t_a = 0.025              # temps d'avaluacio (adimensional)


# Conversio adimensional -> fisic (o al reves)

# Les dues primeres serveixen per les grafiques del problema, per poser-ho unitats)
def a_celsius(T_hat):
    """Temperatura adimensional -> graus Celsius."""
    return Tc + T0 * T_hat


def a_adimensional(T_celsius):
    """Graus Celsius -> temperatura adimensional."""
    return (T_celsius - Tc) / T0



def a_segons(t_hat):
    """Temps adimensional -> segons."""
    return t0 * t_hat


def a_metres(x_hat):
    """Posicio adimensional -> metres."""
    return L * x_hat


if __name__ == "__main__":
    print(f"V_ef          = {V_ef:.3f} V")
    print(f"Pext          = {Pext:.4e} W/m^3")
    print(f"alpha         = {alpha:.4e} m^2/s")
    print(f"t0            = {t0:.1f} s  ({t0 / 60:.1f} min)")
    print(f"T0            = {T0:.2f} K")
    print(f"T_hat_mort    = {T_hat_mort:.5f}")
    print(f"T_hat_limit   = {T_hat_limit:.5f}")
    print(f"zona malalta  = [{x_malalt_i}, {x_malalt_f}]")
    print(f"dx            = {dx}")
    print(f"t_a           = {t_a}  ->  {a_segons(t_a):.1f} s")
    print(f"max estacionari = {a_celsius(1 / 8):.1f} C")