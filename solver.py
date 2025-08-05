import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def pendulum(time, y):
    theta = y[0]
    omega = y[1]
    
    # System parameters 

    q = 0.5
    omega0 = 1.5
    F = 1.44
    freq = 2/3
    
    # EDO System
    dtheta_dt = omega
    domega_dt = F * np.cos(freq * time) - q * omega - omega0**2 * np.sin(theta)
    
    return [dtheta_dt, domega_dt]

if __name__ == '__main__':
    t_span = (0, 500) 
    t_eval = np.linspace(*t_span, 50000)
    y0 = [3.14, 0]
    
    sol = solve_ivp(pendulum, t_span, y0, t_eval=t_eval)
    
    t = sol.t
    theta = sol.y[0]
    omega = sol.y[1]
    
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(t, theta)
    plt.title("Posiçã angular θ(t)")
    plt.xlabel("Tempo")
    plt.ylabel("θ")

    plt.subplot(1, 2, 2)
    plt.plot(theta, omega, alpha=0.7)
    plt.title("Espaço de fase: ω(t) vs θ(t)")
    plt.xlabel("θ")
    plt.ylabel("ω")

    plt.tight_layout()
    plt.show()