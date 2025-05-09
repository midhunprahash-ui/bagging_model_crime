import numpy as np
import matplotlib.pyplot as plt

# Constants
r_E = 1.0  # AU
r_M = 1.524  # AU
a = (r_E + r_M) / 2  # 1.262 AU
e = 1 - r_E / a  # ~0.2076
T = (a / r_E)**1.5  # ~1.417 years
t_transfer = T / 2  # ~0.7085 years
T_M = 1.88  # years
omega_E = 2 * np.pi
omega_M = 2 * np.pi / T_M
phi = np.pi - omega_M * t_transfer  # ~0.7746 rad

# Time steps
t = np.linspace(0, t_transfer, 100)

# Kepler's equation solver
def solve_kepler(M, e, tol=1e-6):
    E = M
    while True:
        delta = (M - (E - e * np.sin(E))) / (1 - e * np.cos(E))
        E += delta
        if abs(delta) < tol:
            break
    return E

# True anomaly
def true_anomaly(E, e):
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

# Positions
theta_E = omega_E * t
x_E = r_E * np.cos(theta_E)
y_E = r_E * np.sin(theta_E)

theta_M = phi + omega_M * t
x_M = r_M * np.cos(theta_M)
y_M = r_M * np.sin(theta_M)

M = 2 * np.pi / T * t
x_r, y_r = [], []
for m in M:
    E = solve_kepler(m, e)
    theta = true_anomaly(E, e)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x_r.append(r * np.cos(theta))
    y_r.append(r * np.sin(theta))
x_r, y_r = np.array(x_r), np.array(y_r)

# Plot
plt.figure(figsize=(8, 8))
plt.plot(x_E, y_E, 'b-', label='Earth orbit')
plt.plot(x_M, y_M, 'r-', label='Mars orbit')
plt.plot(x_r, y_r, 'g-', label='Rocket trajectory')
plt.plot(x_E[0], y_E[0], 'bo', label='Earth at launch')
plt.plot(x_M[0], y_M[0], 'rs', label='Mars at launch')
plt.plot(x_r[0], y_r[0], 'g^', label='Rocket at launch')
plt.plot(x_E[-1], y_E[-1], 'b*', label='Earth at arrival')
plt.plot(x_M[-1], y_M[-1], 'r*', label='Mars at arrival')
plt.plot(x_r[-1], y_r[-1], 'g*', label='Rocket at arrival')
plt.plot(0, 0, 'yo', label='Sun')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title('Rocket Trajectory from Earth to Mars')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()