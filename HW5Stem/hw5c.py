import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def ode_system(t, X, A, Cd, ps, pa, V, beta, rho, Kvalve, m, y):
    """
    Define the system of ODEs for the hydraulic valve system.

    Parameters:
    t : float - Current time
    X : array - State variables [x, xdot, p1, p2]
    A, Cd, ps, pa, V, beta, rho, Kvalve, m, y : float - System parameters

    Returns:
    list - Derivatives [dx/dt, d(xdot)/dt, dp1/dt, dp2/dt]
    """
    x = X[0]  # Position
    xdot = X[1]  # Velocity
    p1 = X[2]  # Pressure 1
    p2 = X[3]  # Pressure 2

    # Derivatives
    dx_dt = xdot
    dxdot_dt = (p1 - p2) * A / m
    dp1_dt = (beta / (V * rho)) * (y * Kvalve * (ps - p1) - rho * A * xdot)
    dp2_dt = -(beta / (V * rho)) * (y * Kvalve * (p2 - pa) - rho * A * xdot)

    return [dx_dt, dxdot_dt, dp1_dt, dp2_dt]


def main():
    # Define time array
    t = np.linspace(0, 0.02, 200)

    # System parameters
    myargs = (4.909e-4,  # A (area)
              0.6,  # Cd (unused)
              1.4e7,  # ps (supply pressure, Pa)
              1.0e5,  # pa (ambient pressure, Pa)
              1.473e-4,  # V (volume)
              2.0e9,  # beta (bulk modulus)
              850.0,  # rho (density)
              2.0e-5,  # Kvalve (valve constant)
              30,  # m (mass)
              0.002)  # y (constant input)

    # Extract pa for initial conditions
    pa = myargs[3]

    # Initial conditions: [x, xdot, p1, p2]
    ic = [0, 0, pa, pa]

    # Solve the ODE system
    sln = solve_ivp(ode_system, [t[0], t[-1]], ic, args=myargs, t_eval=t, method='RK45')

    # Extract solution components
    xvals = sln.y[0]  # Position
    xdot = sln.y[1]  # Velocity
    p1 = sln.y[2]  # Pressure 1
    p2 = sln.y[3]  # Pressure 2

    # Create subplots
    plt.figure(figsize=(10, 8))

    # Plot 1: Velocity (xdot) vs Time
    plt.subplot(2, 1, 1)
    plt.plot(t, xdot, 'b-', label=r'$\dot{x}$')
    plt.title('Velocity Response of Hydraulic Valve System')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\dot{x}$ (m/s)')
    plt.grid(True)
    plt.legend(loc='best')

    # Plot 2: Pressures (p1 and p2) vs Time
    plt.subplot(2, 1, 2)
    plt.plot(t, p1, 'b-', label='$p_1$')
    plt.plot(t, p2, 'r-', label='$p_2$')
    plt.title('Pressure Response of Hydraulic Valve System')
    plt.xlabel('Time (s)')
    plt.ylabel('$p_1, p_2$ (Pa)')
    plt.grid(True)
    plt.legend(loc='best')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()