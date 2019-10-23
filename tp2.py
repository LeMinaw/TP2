# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from math import pi, sin, cos

def __EXO__(number):
    print(f"\n{12*'='} EXO N°{number} {12*'='}\n")


__EXO__(1.1)

a = np.array((0, 0))
b = a # B est une référence à A
b[0] = 1
print(f"Passage par référence : {a}") # A est altéré par les modifications de B

a = np.array((0, 0))
b = np.copy(a) # B est une copie de A (passage par assignement)
b[0] = 1
print(f"Passage par assignement : {a}") # A n'est altéré par les modifications de B


__EXO__(1.2)

array = np.array((
    (1, 2, 3, 4),
    (3, 4, 5, 6),
    (7, 8, 9, 10),
    (11, 12, 13, 14),
))
print(f"Seconde ligne :\n{array[1]}")
print(f"Troisième colonne :\n{array[:,2]}")

array[0:2, 0:2] = np.array((
    (3, 3),
    (3, 3)
))
print(f"Array modifié :\n{array}")


__EXO__(1.3)

arrays = np.split(array, 2)
print(f"Arrays divisés :\n{arrays}")
array = np.concatenate(arrays)
print(f"Array concaténé :\n{array}")

arrays = np.split(array, 2, axis=1)
print(f"Arrays divisés :\n{arrays}")
array = np.concatenate(arrays, axis=1)
print(f"Array concaténé :\n{array}")


__EXO__(2.1)

x_step = .1
x = np.arange(0, pi/2, x_step)

vsin = np.vectorize(sin)
vcos = np.vectorize(cos)
sin_x = vsin(x)
cos_x = vcos(x)

sin_x_deriv = []
for i in range(1, len(x)-1):
    deriv = (sin_x[i+1] - sin_x[i-1]) / (2*x_step)
    sin_x_deriv.append(deriv)

delta = cos_x[1:-1] - sin_x_deriv

print(f"Erreurs sur l'estimtion numérique des dérivées :\n{delta}")
# La précision dépend de x_step, puisque la dérivée est atteinte
# à la limite du taux d'acroissement.


__EXO__(2.2)

sin_x_poly = np.poly1d(np.polyfit(x, sin_x, 2))
cos_x_poly = np.poly1d(np.polyfit(x, cos_x, 2))

plt.plot(x, sin_x, label="sin(x)")
plt.plot(x, cos_x, label="cos(x)")
plt.plot(x, sin_x_poly(x), label="Approx. sin(x)")
plt.plot(x, cos_x_poly(x), label="Approx. cos(x)")
plt.title("Fonctions trigo et approx. au degré 2")
plt.xlabel("θ")
plt.legend()
plt.show()
plt.close()


__EXO__(2.3)

a = np.matrix((
    (1, 2),
    (3, 4)
))
b = np.matrix((
    (1, -3),
    (-5, -7)
))
a = a + np.transpose(a)
b = b + np.transpose(b)

c = a *b
print(f"Produit :\n{c}")
print(f"Valeurs propres :\n{la.eigvals(c)}")


__EXO__(2.4)

x_min, x_max = -pi/2, pi/2
x_samples = 1000

x_step = (x_max - x_min) / x_samples
x = np.arange(x_min, x_max, x_step)

vsin = np.vectorize(sin)
vcos = np.vectorize(cos)

plt.plot(x, vsin(x), label="sin(x)")
plt.plot(x, vcos(x), label="cos(x)")
plt.title("Fonctions trigo")
plt.xlabel("θ")
plt.legend()
plt.show()
plt.savefig("trigo.png")
plt.close()


__EXO__(2.5)

from scipy import integrate
from math import inf

integ_1 = integrate.quad(lambda x: 1+x**2, 0, 3.5)[0]
integ_2 = integrate.quad(lambda x: np.exp(-x**2), 0, inf)[0]
print(f"Première intégrale : {integ_1}, seconde intégrale : {integ_2}.")


__EXO__(2.6)

from scipy import optimize

func = lambda x: (x+4) * (x+1) * (x-1) * (x-3)
res = optimize.minimize_scalar(func, bracket=(-10**8, 10**8))
print(f"Minimum : y={res['fun']} atteint en x={res['x']}.")


__EXO__(2.7)

def riemann_med_sum(func, start, end, samples):
    step = (end-start) / samples
    x = np.arange(start, end, step)

    vfunc = np.vectorize(func)
    func_x = vfunc(x + step/2) # x(i)+x(i-1) / 2 = x(i) + dx/2
    areas = step * func_x
    return np.sum(areas)

integ_riemann = riemann_med_sum(np.exp, 0, pi/2, 1000)
integ_scipy = integrate.quad(np.exp, 0, pi/2)[0]
print(f"Test : pour la fonction exp, entre 0 et pi/2, on trouve {integ_riemann}, "
      f"soit une erreur de {integ_riemann-integ_scipy} pour 1000 itérations.")

def compute_riemann_rank(func, start, end, eps=.1):
    integ_scipy = integrate.quad(func, start, end)[0]
    i = 1
    while abs(integ_scipy - riemann_med_sum(func, start, end, i)) > eps:
        i += 1
    return i

eps = .1
i = compute_riemann_rank(cos, 0, pi/2, eps)
print(f"Test : pour la fonction cos, entre 0 et pi/2, {i} itérations sont nécéssaires "
      f"pour atteindre la précision {eps}.")


__EXO__(3)

var = np.array((0, 0, 0.3, 0)) # l, l', θ, θ'
time = np.linspace(0, 25, 1000)
l_0 = 1
k = 1.2
m = .035
g = 9.81

def pendulum(var, time, l_0, k, m, g):
    l, d_l, theta, d_theta = var
    
    new_l = d_l
    new_theta = d_theta
    new_d_l = (l_0+l) * theta**2 - k/m * l + g * cos(theta)
    new_d_theta = -1 / (l_0+l) * (2 * d_l * d_theta + g * sin(theta))
    
    return new_l, new_d_l, new_theta, new_d_theta

solutions = integrate.odeint(pendulum, var, time, args=(l_0, k, m, g))

# Affichage premier graphique
fig, ax_1 = plt.subplots()
ax_2 = ax_1.twinx()
plt.title("Évolution des variables")
ax_1.set_xlabel("t[s]")
ax_1.set_ylabel("L[m], L'[m/s]")
ax_2.set_ylabel("θ[rad], θ'[rad/s]")
ax_1.plot(time, solutions[:, 0], label="L",  color='darkred')
ax_1.plot(time, solutions[:, 1], label="L'", color='indianred')
ax_2.plot(time, solutions[:, 2], label="θ",  color='dodgerblue')
ax_2.plot(time, solutions[:, 3], label="θ'", color='skyblue')
ax_1.legend()
ax_2.legend()
plt.show()
plt.close()

# Affichage second graphique
l = solutions[:, 0]
theta = solutions[:, 2]

def proj_x(l, theta):
    return (l+l_0) * sin(theta)

def proj_y(l, theta):
    return -(l+l_0) * cos(theta)

x = np.vectorize(proj_x)(l, theta)
y = np.vectorize(proj_y)(l, theta)

plt.title("Trajectoire du pendule")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.plot(x, y)
plt.plot((0, x[0]),  (0, y[0]),  color='k', linewidth=2, label="Départ pendule", linestyle='dashed')
plt.plot((0, x[-1]), (0, y[-1]), color='k', linewidth=2, label="Arrivée pendule")
plt.axis('equal')
plt.legend()
plt.show()
plt.close()
