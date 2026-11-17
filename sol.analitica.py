#discretització espaciotemporal

#la edp que volem resoldre és df/dt = d^2f/dx^2 + p
#aproximem les derivades com df/dt=(f_(m+1) - f_(m))/(dt)  i   d^2f/dx^2 = (f(i-1)-2f(i)+f(i+1))/(dx^2)
#on els index m es refereixen als passos temporals i els index i als espacials.

#definim r com (reordenant l'edp separant els passos de temps de manera que tinguem 
# la funció futura en funció de la passada es veu que és convenient definir-ho així):


#Creem la matriu A que descriu com evoluciona cada punt del teixit a cada pas temporal

import matplotlib.pyplot as plt
import numpy as np

# Paràmetres del problema
beta = 1.0  # Temperatura inicial i de frontera adimensional
N = 40  # Nombre de termes on trunquem la sèrie de Fourier
T_final = 0.025  # Temps adimensional que demana el guió (t_a = 0.025)

# Paràmetres físics per calcular q_0
sigma = 0.472  # Conductivitat elèctrica [S/m]
kappa = 0.56  # Conductivitat tèrmica [W/(m·K)]
V_pic = 40.0  # Voltatge de pic [V]
Tc = 309.65  # Temperatura corporal [K] (36.5 °C + 273.15)

# Terme font adimensional constant q_0
q0 = (sigma * V_pic**2) / (2 * kappa * Tc)

# defineixo el mallat espaciotemporal sobre el que evaluarà la funció el codi
x = np.linspace(0, 1, 101)
t = np.linspace(0, T_final, 100)
X, T_mesh = np.meshgrid(x, t)  #Creo matrius amb tots els elements de x i t per no haver d'escriure dos bucles un dins de l'altre



f = np.ones_like(X) * beta  #matriu tota amb valors beta. Es tracta de l'estat natural del teixit cardíac i a aquest estat s'aplica la font de calor i la consequent evolució temporal

#bucle que suma les contribucions del sumatori fins al terme N-èssim per obtenir la temperatura a cada punt espaciotemporal
for n in range(1, N + 1, 2):  # el 2 es referiex a la longitud dels passos (només agafa les n parelles perque les senars al fer la integral s'anul·len totes)
  b_n = (4 * q0) / (n * np.pi)
  terme_temporal = (1 - np.exp(-(n**2) * np.pi**2 * T_mesh)) / (n**2 * np.pi**2)
  f += b_n * terme_temporal * np.sin(n * np.pi * X) 

# Gràfica
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, T_mesh, f, cmap="viridis")

ax.set_xlabel("x (adimensional)")
ax.set_ylabel("t (adimensional)")
ax.set_zlabel("f(x,t)")
ax.set_title("Solució analítica truncada de l'EDP")

plt.show()


