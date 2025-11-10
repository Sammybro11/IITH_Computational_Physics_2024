#%%
import numpy as np

#%% md
# Consider a potential well,
# $$ V(x) = \begin{cases} 0 & x < 0 \\ -V_0 & 0 < x < a \\ 0 &  x > a \\ \end{cases} $$
# 
# With transcendental equation $$ f(z) = \tan(z) - \frac{2 z \sqrt{1 -z^2}}{2z^2 - 1}$$
#%%
def f(z):
    return np.tan(z) - 2 * z * np.sqrt((1 - z**2)) / (2 * (z**2) - 1)

def df(z):
    return (f(z + 1e-10) - f(z - 1e-10) ) / (2e-10)
#%%
def newton(guess):
    z = guess
    for i in range(100):
        fz = f(z)
        dfz = df(z)
        z_new = z - fz / dfz

        if abs(z_new - z) < 1e-9:
            break
        print(f'z = {z:.9f}')
        z = z_new

    return z_new
#%% md
# Since $$\frac{2 z \sqrt{1 -z^2}}{2z^2 - 1} \to \pm \infty \quad \text{as} \quad z \to \frac{1}{\sqrt{2}} \approx 0.70711 $$
# 
# We will get different solutions for newton method on both sides of $\frac{1}{\sqrt{2}}$, so we will multiple samples on both sides and observe.
#%%
guess_values = np.linspace(0.65, 0.9, 26)
for guess in guess_values:
    print(f'Initial guess: {guess:.8f}\n')
    print(f'Obtained z: {newton(guess):.9f}\n')

#%% md
# We can see that $z$ converges to $0$ and $0.90036722$ for values of guess below and above $\frac{1}{\sqrt{2}}$ respectively, so now we find our final value of $f$
#%%
Energy = (newton(0.75)**2) - 1
print(f'Final Energy: {Energy:.7f}\n')
#%% md
# Energy eigenvalue is $-0.1893389$