#import "@preview/boxed-sheet:0.1.0": *
#import "@preview/muchpdf:0.1.2": muchpdf

#set text(font: (
    "Times New Roman",
    "Fira Sans",
))

#let author = "Samyak Rai"
#let title = "Computational Physics CheatSheet"

#show: cheatsheet.with(
    title: title,
    authors: author,
    write-title: true,
    title-align: left,
    title-number: true,
    title-delta: 2pt,
    scaling-size: false,
    font-size: 5.5pt,
    line-skip: 5.5pt,
    x-margin: 10pt,
    y-margin: 30pt,
    num-columns: 4,
    column-gutter: 2pt,
    numbered-units: false
)

= Root Finding
#concept-block(body: [
    #inline("Bisection Method")
    ```python
    def bisection(f, a, b, tol=1e-8, maxiter=100):
        fa, fb = f(a), f(b)
        if fa == 0: return a
        if fb == 0: return b
        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        for _ in range(maxiter):
            c = 0.5*(a+b)
            fc = f(c)
        if abs(b-a) < tol or abs(fc) < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        return 0.5*(a+b)
        ```
    #inline("Newton Raphson Method")
    ```python
    def newton(f, df, x0, tol=1e-10, maxiter=50):
        x = x0
        for i in range(maxiter):
            fx = f(x)
            dfx = df(x)
            if abs(dfx) < 1e-16:
                raise ZeroDivisionError("Derivative too small")
            dx = fx/dfx
            x = x - dx
            if abs(dx) < tol:
                return x
        return x

    def newton_fd(f, x0, tol=1e-10, maxiter=50, h=1e-6):
        def df(x):
            return (f(x+h)-f(x-h))/(2*h)
        return newton(f, df, x0, tol, maxiter)
    ```
])

= Numerical Integration Techniques
#concept-block(body: [

    #inline("Trapezoidal Rule")
    ```python
    def trapezoid(f, a, b, n=1000):
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b-a)/n
        return (h/2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])
    ```
    #inline("Simpson Rule")
    ```python
    def simpson(f, n, a, b):
        if n % 2 == 1:
            raise ValueError("n must be even for Simpson's rule")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        intgr = h/3*(y[0]+y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]))
        return intgr
    ```
])

= Ordinary Differntial Equations
#concept-block(body: [
    #inline("Verlet Method")
    ```python
    def velocity_verlet(x0, v0, a_func, t):
        t = np.asarray(t)
        n = t.size
        x = np.empty((n,) + np.shape(x0))
        v = np.empty((n,) + np.shape(v0))
        x[0] = x0
        v[0] = v0
        for i in range(n-1):
            dt = t[i+1] - t[i]
            ai = a_func(x[i], t[i])
            x[i+1] = x[i] + v[i]*dt + 0.5*ai*dt*dt
            a_next = a_func(x[i+1], t[i+1])
            v[i+1] = v[i] + 0.5*(ai + a_next)*dt
        return x, v
    ```
    #inline("Runga Kutta 2 (Midpoint)")
    ```python
    def rk2(f, y0, t):
        y0 = np.asarray(y0, dtype=float)
        t = np.asarray(t)
        n = t.size
        y = np.empty((n,) + y0.shape)
        y[0] = y0
        for i in range(n-1):
            dt = t[i+1] - t[i]
            k1 = f(t[i], y[i])
            k2 = f(t[i] + 0.5*dt, y[i] + 0.5*dt*k1)
            y[i+1] = y[i] + dt*k2
        return y
    ```
    #inline("RK4")
    ```python

    def force(function, state):
        dxdt = state[1]
        dvxdt = function.x
        dydt = state[3]
        dvydt = function.y
        return np.array([dxdt, dvxdt, dydt, dvydt])

    def rk4_step(f, state, t, dt):
        k1 = f(state, t)
        k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(state + dt * k3, t + dt)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def Solver(f, state_initial, t0, t_final, dt):
        n_steps = int(np.ceil((t_final - t0) / dt))
        t_eval = np.linspace(t0, t_final, n_steps + 1)
        history = np.empty((n_steps + 1, len(state_initial)))
        history[0] = state_initial

        state = state_initial.copy()
        for n in range(n_steps):
            t = t_eval[n]
            state = rk4_step(f, state, t, dt)
            history[n + 1] = state

        return history, t_eval
    ```
    #inline("Predictor Corrector")
    Predictor--corrector methods combine an explicit #emph[predictor]
    (cheap, possibly less accurate) with an implicit or more accurate
    #emph[corrector] that uses the predicted value. A typical workflow is
    PECE (Predict, Evaluate, Correct, Evaluate) where the final Evaluate
    updates the derivative used in the next step.

    #strong[Simple Heun (PECE) --- one-step example.] \
    $ upright("Predictor:") quad & y_(n + 1)^(\( 0 \)) = y_n + h thin f \( t_n \, y_n \) \,\
    upright("Evaluate:") quad & f_(n + 1)^(\( 0 \)) = f \( t_(n + 1) \, y_(n + 1)^(\( 0 \)) \) \,\
    upright("Corrector:") quad & y_(n + 1) = y_n + h / 2 #scale(x: 120%, y: 120%)[\(] f \( t_n \, y_n \) + f_(n + 1)^(\( 0 \)) #scale(x: 120%, y: 120%)[\)] \,\
    upright("Re-evaluate:") quad & f_(n + 1)^(\( 1 \)) = f \( t_(n + 1) \, y_(n + 1) \) thin thin mono("(Optional)") $
    This is PECE: Predictor (Euler) $arrow.r$ Evaluate $arrow.r$ Correct
    (trapezoid) $arrow.r$ Evaluate.

    One may iterate the corrector (use the newly computed $y_(n + 1)$ to
    recompute $f_(n + 1)$ and reapply the Corrector formula) until the
    change is below a tolerance. \
])

= 1D Boundary Value Problems
#concept-block(body: [
    #inline("Newton Shooting")

    ```python
    def newton_shooting(func, x_start: float, x_end: float, y_start: float, y_end: float, num_points: int, max_iterations: int):
        slope= 1.0
        dx = (x_end - x_start) / num_points

        for _ in range(max_iterations):
            initial_state = [y_start, slope]
            x_values, y_values = Solver(func, initial_state, x_start, x_end, dx)

            error = y_values[-1] - y_end

            _, y_values_eps = Solver(func, [y_start, slope+ 1e-6], 
            x_start, x_end, dx)
            error_eps = y_values_eps[-1] - y_end

            error_derivative = ((error_eps - error) / 1e-6) + 1e-9
            slope -= error / error_derivative

            if abs(error) < 1e-6:
                print(f"Converged with boundary error: {error:.2e}")
                break

        return x_values, y_values
    ```

    #inline("Finte Diference Method")
    ```python
    def fd_bvp_linear(p, q, r, a, b, alpha, beta, m):
        h = (b-a)/m
        x = np.linspace(a, b, m+1)
        # interior unknowns y1..y_{m-1}
        A = np.zeros((m-1, m-1))
        B = np.zeros(m-1)
        for i in range(1, m):
            xi = x[i]
            ai = 1.0/h**2 - p(xi)/(2*h)
            bi = -2.0/h**2 + q(xi)
            ci = 1.0/h**2 + p(xi)/(2*h)
            idx = i-1
            if idx > 0:
                A[idx, idx-1] = ai
                A[idx, idx] = bi
            if idx < m-2:
                A[idx, idx+1] = ci
                B[idx] = r(xi)

        # incorporate boundary values
        B[0] -= (1.0/h**2 - p(x[1])/(2*h))*alpha
        B[-1] -= (1.0/h**2 + p(x[-2])/(2*h))*beta
        y_interior = np.linalg.solve(A, B)
        y = np.empty(m+1)
        y[0] = alpha
        y[-1] = beta
        y[1:-1] = y_interior
        return x, y
    ```
])
= 2D Partial Differntial Equations

#concept-block(body: [
    == Iterative Methods
    #inline("Jacobi Iteration")
    ```python
    def Jacobi(V, s, delta, max_iter=5000, tol=1e-6):
        for it in tqdm(range(max_iter), desc= "Iteration: "):
            V_new = V.copy()
            for n in range(1, N+1):
                for m in range(1, M+1):
                    V_new[n,m] = 0.25*(V[n+1,m] + V[n-1,m] + V[n,m+1] + V[n,m-1] - s[n,m]*delta*delta)
            if np.linalg.norm(V_new-V)/np.linalg.norm(V_new) < tol:
                print(f"Converged at iteration {it}")
                return V_new
            V = V_new
        return V
            ```
    #inline("Gauss-Seidel")
    ```python
    def GaussSeidel(V, s, delta, max_iter=5000, tol=1e-6):
        for it in tqdm(range(max_iter), desc= "Iteration: "):
            V_prev = V.copy()
            for n in range(1, N+1):
                for m in range(1, M+1):
                    V[n,m] = 0.25*(V[n+1,m] + V[n-1,m] + V[n,m+1] + V[n,m-1] - s[n,m]*delta*delta)
            if np.linalg.norm(V - V_prev)/np.linalg.norm(V) < tol:
                print(f"Converged at iteration {it}")
                return V
        return V
    ```
    Running the Algorithm for both cases,
    ```python
    N = 100
    M = 100
    Lx, Ly = 1.0, 1.0
    dx = Lx/(N+1)
    dy = Ly/(M+1)
    V = np.zeros((N+2, M+2))
    # setting up Boundary Conditions along y
    y = np.linspace(0, Ly, M+2)
    V[0, :] = np.sin(2*np.pi*y)
    # Constant Term
    s = (np.zeros((N+2, M+2)))

    V_sol = Solver(V, s, dx) # if dx and dy are different change things a bit
    ```
    == Matrix Formulation
    ```python
    # Define N, M and delta
    N = 100
    M = 100
    dx = 1.0 / (N + 1)
    dy = 1.0 / (M + 1)


    # Create 1D finite difference matrices
    def create_1d_laplacian(n):
        main_diag = np.full(n, -2)
        side_diag = np.full(n - 1, 1)
        return np.diag(main_diag) + np.diag(side_diag, 1) + np.diag(side_diag, -1)


    # 1D matrices
    A_x = create_1d_laplacian(N)
    A_y = create_1d_laplacian(M)
    I_N = np.eye(N)
    I_M = np.eye(M)

    # 2D Laplacian using Kronecker products
    D_x = np.kron(I_M, A_x)  # ∂²/∂x²
    D_y = np.kron(A_y, I_N)  # ∂²/∂y²

    L = D_x / (dx ** 2) + D_y / (dy ** 2)

    # Boundary condition vectors
    V_left = np.sin(2* np.pi * np.linspace(0,1,M))  # V(0,:)
    V_right = np.zeros(M)  # V(N+1,:)
    V_bottom = np.zeros(N)  # V(:,0)
    V_top = np.zeros(N)  # V(:,M+1)

    # Pre-scale by dx, dy once
    V_left /= dx ** 2
    V_right /= dx ** 2
    V_bottom /= dy ** 2
    V_top /= dy ** 2

    # Create source term
    x = np.linspace(dx, 1 - dx, N)
    y = np.linspace(dy, 1 - dy, M)
    X, Y = np.meshgrid(x, y, indexing="ij")

    f = 0 * X * Y
    f_vec = f.flatten()

    # Copy to boundary-corrected RHS
    f_bc = f_vec.copy()

    for j in range(M):
        for i in range(N):
            k = j * N + i  # Flattened index

            if i == 0:  # Left boundary
                f_bc[k] -= V_left[j]
            if i == N - 1:  # Right boundary
                f_bc[k] -= V_right[j]
            if j == 0:  # Bottom boundary
                f_bc[k] -= V_bottom[i]
            if j == M - 1:  # Top boundary
                f_bc[k] -= V_top[i]

    # Solve system
    u_vec = np.linalg.solve(L, f_bc)
    u_numerical = u_vec.reshape((N, M))

    # Plot
    plt.contourf(Y, X, u_numerical, levels=25, cmap="viridis")
    plt.colorbar()
    plt.title("Numerical Solution")
    plt.show()
    ```
])

= Time Dependent PDEs
#concept-block(body: [
Consider the continuity equation, 
$ (diff rho)/(diff t) + nabla dot ( rho arrow(v)) = 0 $
In one dimensional, 
$ (diff rho(x, t))/(diff t) + (diff rho u(x,t))/(diff rho) = 0 $
#inline("Forward Euler ( Explicit ) Scheme")
Taking $u(x,t) = u$ constant we get 
$ rho_j^(n+1) = rho_j^n - u (Delta t)/(2 Delta x)( rho_(j+1)^n - rho_(j-1)^n) $
This method is unstable as the eigenvalue for the $P$ Matrix is always more than 1, thus norm increases to infinity.
```python
def forward_euler(rho0, u, dx, dt, steps):
    rho = rho0.copy()
    N = len(rho)
    for _ in range(steps):
        rho_new = rho.copy()
        rho_new[1:-1] = rho[1:-1] - u*dt/(2*dx)*(rho[2:] - rho[:-2])
        rho = rho_new
    return rho
```
#inline("Implicit Euler Method")
$ rho_j^(n+1) = rho_j^n - u (Delta t)/(2 Delta x)( rho_(j+1)^(n+1) - rho_(j-1)^(n+1)) $
Thus we can write it as, $ bold(rho)^n = T bold(rho)^(n+1) + B bold(rho)^(n+1) $
$ T = mat(delim: "[", 1, a, 0, dots, 0 ; -a, 1, a, dots, 0; dots.v , dots.v ; 0, dots.c, 0 , -a , 1) quad B = mat(delim: "[", 0, 0, 0, dots, -a ; 0, 0, 0, dots, 0; dots.v , dots.v ; a, dots.c, 0 , 0 , 0) $
where $a = (u Delta t)/(2 Delta x)$

```python
def implicit_euler_periodic(rho0, u, dx, dt, steps):
    rho = rho0.copy()
    N = rho.size
    a = u * dt / (2.0 * dx)   # as defined in the notes
    # Build A = T + B (dense)
    A = np.eye(N, dtype=float)
    
    if N > 1:
        off = a * np.ones(N - 1, dtype=float)
        A += np.diag(off, 1)       # +a on superdiagonal
        A -= np.diag(off, -1)      # -a on subdiagonal
        # periodic wrap contributions (the B matrix)
        A[0, -1] = -a              # top-right corner
        A[-1, 0] =  a              # bottom-left corner
    for _ in range(steps):
        rho = np.linalg.solve(A, rho)

    return rho
```

#inline("Leapfrog Method")
By taking centre difference in not just space but also time, 
$ rho_j^(n+1) = rho_j^(n-1) - u (Delta t)/( Delta x)( rho_(j+1)^(n+1) - rho_(j-1)^(n+1)) $

```python
def leapfrog(rho_prev, rho_curr, u, dx, dt, steps):
    rho_nm1 = rho_prev.copy()
    rho_n = rho_curr.copy()
    for _ in range(steps):
        rho_np1 = rho_nm1 - u * dt / dx * (rho_n[2:] - rho_n[:-2]) / 2
        rho_nm1, rho_n = rho_n, np.concatenate(([0], rho_np1, [0]))
    return rho_n
```

#inline("Lax-Friedrich Method")
Current step and position is taken as an average of neighbouring positions,
$ bold(rho)^(n+1) = 1/2 mat(delim: "[", 0, 1- u (Delta t) / (Delta x), 0 , dots.c , 0; 1+ u (Delta t) / (Delta x), 0, 1- u (Delta t) / (Delta x), dots.c , 0 ; dots.v ; 0,  dots.c , 0 , 1+ u (Delta t) / (Delta x), 0 ) bold(rho)^n $

```python
def lax_friedrichs_T(N, u, dx, dt):
    c = u * dt / (2.0 * dx)
    upper = 0.5 - c   # weight for rho_{j+1}
    lower = 0.5 + c   # weight for rho_{j-1}
    T = np.zeros((N, N), dtype=float)
    for j in range(N):
        if j - 1 >= 0:
            T[j, j - 1] = lower
        if j + 1 < N:
            T[j, j + 1] = upper

    return T

def evolve_matrix(rho0, T, steps, periodic=False, u=None, dx=None, dt=None):
    rho = rho0.copy()
    N = rho.size

    if periodic:
        c = u * dt / (2.0 * dx)
        T[0, -1] = 0.5 - c    # contribution from rho_{N-1} into rho_0
        T[-1, 0] = 0.5 + c    # contribution from rho_0 into rho_{N-1}
        P = T + B
    else:
        P = T

    for _ in range(steps):
        rho = P @ rho

N = 200
x = np.linspace(0, 1, N, endpoint=False)
dx = x[1] - x[0]
u = 0.1
dt = 0.8 * dx / u
steps = 100

rho0 = np.exp(-((x - 0.5)**2) / 0.02)
T = lax_friedrichs(N, u, dx, dt)
rho_p  = evolve_matrix(rho0, T, steps, periodic=True, u=u, dx=dx, dt=dt)
```
#inline("Lax-Wendroff")

```python
def lax_wendroff_matrix(N, u, dx, dt):
    c = u * dt / (2 * dx)
    P = np.zeros((N, N))
    for j in range(N):
        P[j, (j - 1) % N] = c + c**2
        P[j, j] = 1 - 2 * c**2
        P[j, (j + 1) % N] = -c + c**2
    return P
```

== Non-Linear Equations
#inline("Richtmyer 2-step Lax Wendroff")
```python
import numpy as np
import matplotlib.pyplot as plt

N = 1000
a, b = 0, 1
x_full = np.linspace(a, b, N+2, dtype=np.float64)
x = x_full[1:-1]
dx = x[1] - x[0]

u = 0.01 * np.sin(2*np.pi*x)
t_max = 20.0
time = 0.0

fig, axes = plt.subplots(4, 5, sharex=True, sharey=True, figsize=(15,12))

snapshots = 20
snapshot_times = np.linspace(0, t_max, snapshots)
s = 0
for snap_t in snapshot_times:
    while time < snap_t:
        u_max = np.max(np.abs(u)) + 1e-9
        dt = 0.6 * dx / u_max
        dt_dx = dt / dx

        # p
        f = 0.5 * u**2
        u_half = 0.5*(u[1:] + u[:-1]) - 0.5*dt_dx*(f[1:] - f[:-1])

        f_half = 0.5 * u_half**2

        u[1:-1] = u[1:-1] - dt_dx * (f_half[1:] - f_half[:-1])

        # periodic wrap for ghost cells
        u[0]  = u[-2]
        u[-1] = u[1]

        time += dt

    r = s // 5
    c = s % 5
    ax = axes[r, c]
    ax.plot(x, u, color=f'#{8*s:06X}')
    ax.set_title(f"t ≈ {time:.2f}s")
    ax.grid(True)
    s += 1

plt.tight_layout()
plt.show()
```
#inline("Schrodinger Equation")
```python
def crank_nicolson(psi0, V, dx, dt, hbar=1, m=1, steps=1000):
    N = len(psi0)
    D = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
    P = np.diag(V)
    H = -D + P

    A = np.eye(N) + 1j * dt / (2) * H
    B = np.eye(N) - 1j * dt / (2) * H

    psi = psi0.copy()
    for _ in range(steps):
        psi = np.linalg.solve(A, B @ psi)
        psi /= np.linalg.norm(psi)  # normalization
    return psi

```
])

= Monte Carlo Methods
#concept-block(body: [
    == Integration
    #inline("Area of Circle")
    ```python
    N = 10000
    rng = np.random.default_rng(seed=67)
    x = rng.random(N)
    y = rng.random(N)

    inside = (x*x + y*y) < 1
    cum_inside = np.cumsum(inside)
    trials = np.arange(1, N+1)
    value_circle = 4 * (cum_inside / trials)
    print(value_circle[-1])
    ```
    #inline("Volume of N-dim Sphere")
    ```python
    N = 1000000
    dimension = 10

    def N_dim_Sphere(N, dimension):
        rng = np.random.default_rng(seed=70)
        points = rng.random((N, dimension))
        points = points*points
        r = np.sum(points, axis=1)

        inside = np.where(r <= 1, 1, 0)
        cum_inside_me = np.cumsum(inside)
        trials_ndim = np.arange(1, N+1)
        value_circle = (2**dimension) * (cum_inside_me/ trials_ndim)
        print(f"Monte Carlo Estimate: {value_circle[-1]:.5f}")
        true_value = (np.pi**(dimension/2)) / math.gamma(dimension/2 + 1)  # π^{d/2}/Γ(d/2+1)
        print(f"True Value: {true_value:.5f}")
    ```
    #inline("Area under Function")
    ```python
    def Monte_Carlo_Integration(f, x_lim: np.ndarray, y_lim: np.ndarray, N: int = 10000, seed: int = 42):
        np.random.seed(seed)
        points = np.random.random((N, 2))

        # Making Random Points go into range of x_lim and y_lim
        points[:, 0] *= x_lim[1] - x_lim[0]
        points[:, 1] *= y_lim[1] - y_lim[0]
        points[:, 0] += x_lim[0]
        points[:, 1] += y_lim[0]

        f_points = f(points[:, 0]) # Array of evaluated values on given random points

        positive_region = (f_points >= 0) & (points[:, 1] >= 0) & (points[:, 1] <= f_points)
        negative_region = (f_points < 0) & (points[:, 1] < 0) & (points[:, 1] >= f_points)

        point_contributions = positive_region.astype(int) - negative_region.astype(int)

        cumulative_contributions = np.cumsum(point_contributions)

        # Running estimate at each iteration
        box_area = (x_lim[1] - x_lim[0]) * (y_lim[1] - y_lim[0])
        iterations = np.arange(1, N + 1)
        convergence = (cumulative_contributions / iterations) * box_area
        return convergence[-1]

    limits_x = np.array([0, 3*np.pi/2])
    limits_y = np.array([np.exp(0), np.exp(3*np.pi/2)]) # Min and Max Value of the function in entire range
    N = 10000
    seed = 67
    print(f"Monte Carlo: {Monte_Carlo_Integration(function, limits_x, limits_y, N, seed):.5f}")
    ```
    == Casting Distributions
    To generate a function a distribution $f(y)$, then we need to create a mapping from $x |-> y$ such that 
    $ (dif y)/(dif x) = 1/(f(y)) $
    Then for a probability distribution $rho(x)$, we can find the integral by,
    $ integral f(x) dif x = lr( angle.l (f(x))/rho(x) angle.r ) $

    #inline("Closed Form Integrals")
    Given $f(y) = sin(y)$, we can say $sin(y)dif y = dif x => y = arccos(1 - x)$.

    ```python
    N = 1000000
    x = np.random.rand(N)             # Uniform [0, 1)
    y = np.arccos(1 - x)              # Inverse transform sampling
    plt.hist(y, bins=100, density=True, alpha=0.6, label='Sampled')
    plt.plot(np.linspace(0, np.pi/2, 100), np.sin(np.linspace(0, np.pi/2, 100))/1, 'r-', lw=2, label='sin(y)')
    plt.legend(); plt.show()
    ```
    #inline("Box Muller Transform")
    To generate the standard normal distribution, 
    $ f(y) = 1/(sigma sqrt(2 pi)) "exp" lr([ - 1/2 (y - mu)^2 / ( sigma^2)]) quad "where " mu = 0, thin sigma = 1 $
    There exists no closed form solution to the required integral to obtain the map $x |-> y$ so we convert to polar coordinates,
    $ f(x)f(y) dif x dif y |-> 1/(2 pi) "exp"( - r^2 / 2 ) r thin dif r dif theta $
    Now to generate the probability distribution above, 
    $ g(r) = "exp"( - r^2 /2 ) r => dif x = g(r)dif r => r = - sqrt(2 ln(x)) $
    $ theta = x times (2 pi) $
    Thus, we get 2 normal distributions 
    $ y_1 = r cos(theta) quad y_2 = r sin(theta) $
    #line(length: 100%)
    ```python
    N = 100000

    x_1 = np.random.rand(N)
    x_2 = np.random.rand(N)

    r = np.sqrt(-2*np.log(x_1))
    theta = 2*np.pi*x_2

    y_1 = r * np.cos(theta)
    y_2 = r * np.sin(theta)
    normal = lambda x: np.exp(-x**2/2)/(np.sqrt(2*np.pi))
    compare = np.linspace(-3, 3, N)

    counts1, bins1 = np.histogram(y_1, bins=50)
    deltax1 = bins1[1]-bins1[0]
    pdf1 = (counts1/N)/deltax1
    bin_mid1 = (bins1[:-1]+bins1[1:])/2

    counts2, bins2 = np.histogram(y_2, bins=50)
    deltax2 = bins2[1]-bins2[0]
    pdf2 = (counts2/N)/deltax2
    bin_mid2 = (bins2[:-1]+bins2[1:])/2

    plt.hist(y_1, bins=50, alpha=0.5, label='cos', color='r')
    plt.hist(y_2, bins=50, alpha=0.5, label='sin', color='g')
    plt.legend()
    plt.show()
    plt.clf()

    plt.plot(bin_mid1, pdf1, label='cos', color='r', alpha = 0.5)
    plt.plot(compare, normal(compare), label='normal', color='b')
    plt.plot(bin_mid2, pdf2, label='sin', color='g', alpha = 0.5)
    plt.legend()
    plt.show()
    plt.clf()
    ```
    == Random Walks
    #inline("Metropolis Hastings")
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize parameters
    n_samples = 100000  # total number of samples to generate
    samples = []  # list to store accepted samples
    x_current = 0.5  # start point in the middle of [0,1]
    sigma = 0.1  # proposal step size (standard deviation)

    # Run the Metropolis algorithm
    for i in range(n_samples):
        # Propose a new candidate from a normal distribution centered at current x
        x_proposed = x_current + np.random.normal(0, sigma)

        # Reflect if the proposed value goes out of [0,1] (to stay within bounds)
        if x_proposed < 0:
            x_proposed = -x_proposed
        elif x_proposed > 1:
            x_proposed = 2 - x_proposed

        # Compute acceptance ratio (unnormalized)
        p_current = - np.sin(x_current) * np.log(x_current)
        p_proposed = - np.sin(x_proposed) * np.log(x_proposed)

        # Avoid divide-by-zero errors
        if p_current == 0:
            acceptance_ratio = 1
        else:
            acceptance_ratio = p_proposed / p_current

        # Accept or reject based on Metropolis criterion
        if np.random.rand() < min(1, acceptance_ratio):
            x_current = x_proposed  # accept move

        # Store the current sample
        samples.append(x_current)

    # Convert to NumPy array for analysis
    samples = np.array(samples)

    # Plot the sampled distribution
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=100, density=True, alpha=0.6, color='skyblue', label='Sampled distribution')

    # Plot the true (unnormalized) function for comparison
    x = np.linspace(0.001, 0.999, 500)
    norm_const = np.trapezoid(-np.sin(x)*np.log(x), x)
    plt.plot(x, -np.sin(x)*np.log(x) / norm_const, 'r-', lw=2, label='True (normalized) function')
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.legend()
    plt.title('Sampling from f(x) = -sin(x) ln(x) using Metropolis algorithm')
    plt.show()
    ```
])
= Neural Networks
#concept-block(body: [
    #let pdfpath = "module10.pdf"
    #let data = read(pdfpath, encoding: none)
    #align(center, muchpdf(data, pages: (start: 7, end: 14), width: 78% ))
    #align(center, muchpdf(data, pages: (start: 15, end: 25), width: 78% ))
    #align(center, muchpdf(data, pages: (start: 28, end: 33), width: 78% ))
])
