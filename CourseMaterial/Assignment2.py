import numpy as np
import matplotlib.pyplot as plt
# state variable structure
# [ x , y , z , vx, vy, vz ]

def ElectricField(state: np.ndarray, time: float):
    E_x = 0.0
    E_y = 0.02
    E_z = 0.0

    return np.array([E_x, E_y, E_z])

def MagneticField(state: np.ndarray, time: float):
    M_x = 0.0
    M_y = 0.0
    M_z = (1 + 0.1*state[0] + 0.01 * time)
    return np.array([M_x, M_y, M_z])

def Lorentz(state: np.ndarray, time: float, charge: float = 1.0, mass: float = 1.0 ):
    dr_dt = state[3:]

    E = ElectricField(state, time)
    B = MagneticField(state, time)
    Force_B = np.cross(dr_dt, B)
    dv_dt = (charge/mass)*(E + Force_B)

    return np.concatenate([dr_dt, dv_dt])

def rk4_step(state: np.ndarray, time: float, dt: float = 0.01):
    k1 = Lorentz(state, time)
    k2 = Lorentz(state + 0.5 * dt * k1, time + 0.5 * dt)
    k3 = Lorentz(state + 0.5 * dt * k2, time + 0.5 * dt)
    k4 = Lorentz(state + dt * k3, time + dt)
    return state + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

def Solver(state_initial: np.ndarray, time_initial: float = 0.0, time_final: float = 100.0, N: int = 1000):
    dt = (time_final - time_initial) / N
    time_eval = np.linspace(time_initial, time_final, N+1)
    state = np.copy(state_initial)
    state_history = np.zeros((N+1, 6))
    state_history[0] = state_initial
    for i in range(1, N+1):
        state = rk4_step(state, time_eval[i-1].item(), dt)
        state_history[i] = state

    return state_history, time_eval

x_pos = input("Enter the x position: ")
y_pos = input("Enter the y position: ")
z_pos = input("Enter the z position: ")
x_vel = input("Enter the x velocity: ")
y_vel = input("Enter the y velocity: ")
z_vel = input("Enter the z velocity: ")
state_initial = np.array([x_pos, y_pos, z_pos, x_vel, y_vel, z_vel], dtype=float)
t_i, t_f = 0.0, 100.0
N = 5000

state_evolution, time_evolution = Solver(state_initial, t_i, t_f, N)

fig = plt.figure(figsize=(12, 8))
plt.scatter(state_evolution[:, 0], state_evolution[:, 1],
            c=time_evolution, s=4, cmap='viridis', label = "Position")
plt.colorbar(label='Time')
plt.xlabel('X - Axis', fontsize = 16)
plt.ylabel('Y - Axis', fontsize=16)
plt.title('Trajectory of Particle in XY Plane', fontsize=20)
plt.grid(True)
plt.legend()
plt.show()







