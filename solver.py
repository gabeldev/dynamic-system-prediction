import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd


class ForcedPendulum:
    
    def __init__(self, q=0.5, omega0=1.5, F=1000, freq=2/3):
        """ 
        Parameters of the system:
         - q = damping coefficient
         - omega0 = natural frequency of pendulum
         - F = amplitude of the applied external force
         - freq = external force frequency        
        """
        self.q = q
        self.omega0 = omega0
        self.F = F
        self.freq = freq

        self.time_series = None
        self.theta_series = None
        self.omega_series = None

    def dynamics(self, t, y):
        theta, omega = y

        dtheta_dt = omega
        domega_dt = (self.F * np.cos(self.freq * t) -
                     self.q * omega - 
                     self.omega0**2 * np.sin(theta))
        
        return [dtheta_dt, domega_dt]
    
    def simulate(self, t_span=(0, 500), n_points=50000, inicial_conditions=[3.14, 0]):
        """
        Parameters:
         - t_span = time interval
         - n_points = number of points to evaluate the solution
         - initial_conditions = [theta_inicial, omega_inicial]
        """
        print(f"The system is being simulated with t={t_span[0]} to t={t_span[1]} with {n_points} points")

        # Time points that we want to evaluate the solution
        t_eval = np.linspace(*t_span, n_points)

        solution = solve_ivp(self.dynamics, t_span, inicial_conditions, t_eval=t_eval, method='RK45', rtol=1e-8)

        self.time_series = solution.t
        self.theta_series = solution.y[0]
        self.omega_series = solution.y[1]

        # Normalize theta (-pi to pi)
        self.theta_normalized = np.mod(self.theta_series + np.pi, 2*np.pi) - np.pi

        print(f"Simulation completed, generated points {len(self.time_series)}")
        return self
    
    def prepare_ml_data(self, sequence_length=50, train_split=0.8):
        states = np.column_stack([self.theta_series, self.omega_series])

        x = [] # input sequency
        y = [] # Target output

        for i in range(len(states) - sequence_length):
            x.append(states[i:i+sequence_length])
            y.append(states[i+sequence_length])

        x = np.array(x)
        y = np.array(y)

        split_idx = int(len(x) * train_split)

        data = {
            'x_train': x[:split_idx],
            'y_train': y[:split_idx],
            'x_test': x[split_idx:],
            'y_test': y[split_idx:],
            'sequence_length': sequence_length,
            'n_features': 2 # theta and omega
        }

        return data
    

if __name__ == "__main__":
    pendulum = ForcedPendulum()
    pendulum.simulate(t_span=(0, 500), n_points=50000)
    ml_data = pendulum.prepare_ml_data(sequence_length=50)

    np.savez('pendulum_data.npz', time=pendulum.time_series, theta=pendulum.theta_series, omega=pendulum.omega_series, **ml_data)