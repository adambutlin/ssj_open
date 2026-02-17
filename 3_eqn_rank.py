import numpy as np
import matplotlib.pyplot as plt


# Set number of periods
Q = 50

# Set number of scenarios
S = 3

# Set period in which shock/shift will occur
s = 5

# Create (S x Q) arrays to store simulated data
y = np.zeros((S, Q))  # Income/output
p = np.zeros((S, Q))  # Inflation rate
r = np.zeros((S, Q))  # Policy rate
rs = np.zeros((S, Q))  # Stabilizing interest rate

# Set constant parameter values
a1 = 0.3  # Sensitivity of inflation with respect to output gap
a2 = 0.7  # Sensitivity of output with respect to interest rate
b = 1     # Sensitivity of the central bank to inflation gap
a3 = (a1 * (1 / (b * a2) + a2)) ** (-1)

# Set parameter values for different scenarios
A = np.full((S, Q), 6)  # Autonomous spending
pt = np.full((S, Q), 2)  # Inflation target
ye = np.full((S, Q), 5)  # Potential output

A[0, s:Q] = 7  # Scenario 1: AD boost
pt[1, s:Q] = 3  # Scenario 2: Higher inflation target
ye[2, s:Q] = 5.5  # Scenario 3: Higher potential output

# Initialize endogenous variables at equilibrium values
y[:, 0] = ye[:, 0]
p[:, 0] = pt[:, 0]
rs[:, 0] = (A[:, 0] - ye[:, 0]) / a1
r[:, 0] = rs[:, 0]

# Simulate the model by looping over Q time periods for S different scenarios
for i in range(S):
    for t in range(1, Q):
        # (1) IS curve
        y[i, t] = A[i, t] - a1 * r[i, t - 1]
        # (2) Phillips Curve
        p[i, t] = p[i, t - 1] + a2 * (y[i, t] - ye[i, t])
        # (3) Stabilizing interest rate
        rs[i, t] = (A[i, t] - ye[i, t]) / a1
        # (4) Monetary policy rule, solved for r
        r[i, t] = rs[i, t] + a3 * (p[i, t] - pt[i, t])

#############################################################################
####  Impulse Response Functions
#############################################################################

Tmax = 15       # Set maximum period for plots

# Plot output under different scenarios
plt.figure(figsize=(8, 6))
plt.plot(y[0, :Tmax + 1], label="Scenario 1: aggregate demand boost",
         color='k', linestyle='solid', linewidth=2)
plt.plot(y[1, :Tmax + 1], label="Scenario 2: Rise inflation target",
         color='k', linestyle='dashed', linewidth=2)
plt.plot(y[2, :Tmax + 1], label="Scenario 3: Rise potential output",
         color='k', linestyle='dotted', linewidth=2)

plt.title("Output under Different Scenarios")
plt.xlabel("Time")
plt.ylabel("y")
plt.xlim(1, Tmax)
plt.ylim(np.min(y), np.max(y))
plt.legend()
plt.show()


# Plot policy rate
plt.figure(figsize=(8, 6))
plt.plot(r[0, :Tmax + 1], label="Scenario 1: aggregate demand boost",
         color='k', linestyle='solid', linewidth=2)
plt.plot(r[1, :Tmax + 1], label="Scenario 2: Rise inflation target",
         color='k', linestyle='dashed', linewidth=2)
plt.plot(r[2, :Tmax + 1], label="Scenario 3: Rise potential output",
         color='k', linestyle='dotted', linewidth=2)

plt.title("Policy Rate")
plt.xlabel("Time")
plt.ylabel("r")
plt.xlim(1, Tmax)
plt.ylim(np.min(r), np.max(r))
plt.legend()
plt.show()

# Plot inflation expectations
plt.figure(figsize=(8, 6))
plt.plot(p[0, :Tmax + 1], label="Scenario 1: aggregate demand boost",
         color='k', linestyle='solid', linewidth=2)
plt.plot(p[1, :Tmax + 1], label="Scenario 2: Rise inflation target",
         color='k', linestyle='dashed', linewidth=2)
plt.plot(p[2, :Tmax + 1], label="Scenario 3: Rise potential output",
         color='k', linestyle='dotted', linewidth=2)

plt.title("Inflation")
plt.xlabel("Time")
plt.ylabel("p")
plt.xlim(1, Tmax)
plt.ylim(np.min(p), np.max(p))
plt.legend()
plt.show()

#%%