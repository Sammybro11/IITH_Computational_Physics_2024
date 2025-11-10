import numpy as np
u = 0.1
dt = 0.001
N = 1000
a, b = 0, 1

x_full = np.linspace(a, b, N+2, dtype=np.float64)
x = x_full[1:-1]
dx = x[1]-x[0]

P = np.diag(np.ones(N-1)*(u*dt/(2*dx)), -1) + np.diag(np.ones(N), 0) + np.diag(np.ones(N-1)*(-u*dt/(2*dx)), 1)
P[0, N-1] = -1
P[N-1, 0] = 1

import math
import matplotlib.pyplot as plt

rho0 = np.exp(-1.0*((x-0.5)/0.15)**2)

t = 0.
rho = rho0
soln=[]
while t < 10:
    rho1 = P@rho
    rho = np.copy(rho1)
    if math.isclose(t % 1, 0, abs_tol=1e-5):
        soln.append(rho1)
        plt.plot(x, rho)
        plt.title(f"Solution @ t = {t}")
        plt.show()
        print(t)
    t += dt
    #print(f'Yo@{t}')