import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt


#---------------------------------------------------------
# Parámetros del problema
#---------------------------------------------------------
alpha = 0.1
w0 = 1.0
w = 2.0
T = 2*pi/w            # periodo de la fuerza (muestreo estroboscópico)

#---------------------------------------------------------
# Condición inicial (theta, theta_punto) = (1, 1)
#---------------------------------------------------------
x0, v0 = 1.0, 1.0

#---------------------------------------------------------
# Parámetro de bifurcación: gamma en [0, 2.25]
#---------------------------------------------------------
gamma_min = 0.0
gamma_max = 2.25
dgamma = 0.0025
gamma_values = np.arange(gamma_min, gamma_max + dgamma, dgamma)
n_orbits = len(gamma_values)

#---------------------------------------------------------
# Estado inicial (todas las órbitas en paralelo)
#---------------------------------------------------------
x = np.full(n_orbits, x0)
v = np.full(n_orbits, v0)
y = np.concatenate([x, v])

#---------------------------------------------------------
# Parámetros numéricos
#---------------------------------------------------------
Trans = 250           # periodos descartados (transitorio)
Nkeep = 100           # periodos guardados
steps_per_T = 200
dt = T/steps_per_T

#---------------------------------------------------------
# Dinámica: theta'' = -alpha*theta' - w0^2 sin(theta) + gamma cos(w t) sin(theta)
#---------------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -alpha*v - w0**2*sin(x) + gamma_values*cos(w*t)*sin(x)
    return np.concatenate([dx, dv])

#---------------------------------------------------------
# Runge-Kutta de cuarto orden
#---------------------------------------------------------
def rk4(f, t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

#---------------------------------------------------------
# Almacenamiento estroboscópico
#---------------------------------------------------------
v_strobe = np.empty((Nkeep, n_orbits))
save_index = 0

#---------------------------------------------------------
# Integración
#---------------------------------------------------------
total_steps = (Trans + Nkeep)*steps_per_T
for step in range(total_steps):
    t = step*dt
    y = rk4(dyn, t, y, dt)
    if (step + 1) % steps_per_T == 0:
        if (step + 1)//steps_per_T > Trans:
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

#---------------------------------------------------------
# Diagrama de bifurcación: |theta_punto| vs gamma
#---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
G = np.tile(gamma_values, Nkeep)          # mismo orden que v_strobe.ravel()
ax.scatter(G, np.abs(v_strobe).ravel(), s=0.5, color='blue',
           linewidths=0, rasterized=True)

ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$|\dot{\theta}|$', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim(gamma_min, gamma_max)
ax.set_ylim(0, 3.5)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.show()
