import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad, solve_ivp

#paràmetres

b = 1
N = 40
T = 1
c = 0.472 #conductivitat elèctrica
V = 40 #diferencia de potencial
L = 1  #espessor paret cardíaca normalitzat

x = np.linspace(0,1,100)
t = np.linspace(0,T,100)

#q(x,t) és el terme que representa la font de calor (el calor que aporta la diferencia de potencial)
# q= c · |E|**2 on c és la conductivitat elèctrica. Sabent que E=V/h  (h longitud de la paret cardíaca) i que V_ef = V/Sqrt[2]
#s'obté: q = c·V^2/(2h)

def p(x, t):
    return c*V**2/(2*L**2)

#defineixo els coeficients de fourier de q(x,t) mitjançant la funció quad() (de scipy.integrate) que calcula integrals definides numericamenr
def p_n(n, t):
    integral, error = quad(
        lambda x: 2 * p(x, t) * np.sin(n*np.pi*x), 0, 1
    )
    return integral

#resolució de la EDP
#defineixo un sistema que té com a variables el temps i el vector "a" comformat per tots els N coeficients de fourier de la solució en l'instant t

def sistema(t, a):
    da = np.zeros(N) 

    for n in range(1, N+1):
        da[n-1] = (-n**2 * np.pi**2 * a[n-1] + p_n(n, t))

    return da      #s'està resolvent l'equació de coeficients de fourier a_n'(t) = -n**2 * pi**2 * a_n(t) + p_n(t) resultat d'expandir en serie de fourier 
                   #les derivades de la funció que es preten resoldre. El vector da conté totes les derivades temporals del coeficient a_n(t) a l'instant t


#consdicions inicials:

a0 = np.zeros(N)

sol = solve_ivp(sistema, (0, T), a0, t_eval=t, method="RK45", rtol = 1e-7, atol=1e-9)  #resolem el sistema per obtenir tots els coeficients de fourier en cad instant de temps

#contrucció de la funció temperatura

f = np.zeros((len(t)), (len(x)))

for j in range(len(t)):         #bucle que itera per tots els instants de temps
    f[j, :] = b                 # inicialitzem la funció sempre fent que a tots els punts del teixit la temperatura inicial sigui b

    for n in range(1, N+1):     #un altre bucle que per cada instant de temps actualitza el valor dels N coeficients de fourier
        f[j, :] = f[j, :] + (sol.y[n-1, j] * np.sin(n*np.pi*x))  #suma el resultat nou als anteriors


#representació gràfica de la solució

X, T_mesh = np.meshgrid(x, t)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, T_mesh, f, cmap="viridis")

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("f(x,t)")
ax.set_title("Solucio de l'EDP")

plt.show()
