import matplotlib.pyplot as plt
import numpy as np

a, b = 0, 1
N = 1000

dx = (b-a)/N
x = np.linspace(a, b+dx, N+2)

dt = 0.001
T = 20
t = 0

rho = 0.01*np.sin(2*np.pi*x)
u = rho/2

while t < T:
    rho_half = np.zeros(N+2)
    for i in range(len(rho_half)-1):
        rho_half[i] = (rho[i] + rho[i+1])/2 - (2*dt/dx)*(rho[i+1]*u[i+1] - rho[i]*u[i])
    rho_half[-1] = rho[-1]

    rho = np.zeros(N+2)
    for i in range(1, len(rho)-1):
        rho[i] = rho_half[i] - (dt/dx)*(rho_half[i+1] - rho_half[i-1])

    t += dt

    if t % 2 < dt:
        plt.plot(x, rho)
        plt.show()