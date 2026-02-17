import numpy as np
import matplotlib.pyplot as plt

# Number of scenarios
S = 6

# Arrays to store equilibrium solutions
Y_star = np.empty(S)
w_star = np.empty(S)
C_star = np.empty(S)
I_star = np.empty(S)
r_star = np.empty(S)
rn_star = np.empty(S)
N_star = np.empty(S)
P_star = np.empty(S)

# Exogenous parameter vectors (baseline + shocks)
M0 = np.zeros(S)
G0 = np.zeros(S)
A = np.zeros(S)
Yf = np.zeros(S)
chi = np.zeros(S)

# Baseline parameterisation
M0[:] = 5
G0[:] = 1
A[:] = 2
Yf[:] = 1
chi[:] = 0.6

# Shocks / alternative scenarios
M0[1] = 6     # monetary expansion
G0[2] = 2     # fiscal expansion
A[3]  = 2.5   # productivity boost
Yf[4] = 0.2   # lower expected future income
chi[5] = 0.8  # increased preference for money

# Constant parameters
a = 0.3        # capital elasticity
phi = 0.5      # parameter in disutility of labour (appears in N^(1+phi)/(1+phi))
beta = 0.95
K = 5          # exogenous capital stock
pe = 0.02      # expected inflation (nominal interest add-on)
Gf = 1         # future government spending (for small PV term)

# Initial guesses for endogenous variables (per scenario)
# We keep them scalars and update in-place for each scenario
for i in range(S):
    # sensible initial guesses
    N = 0.6
    C = 1.0
    I = 0.5
    Y = A[i] * (K**a) * N**(1-a)
    r = a * A[i] * (K**(a-1)) * N**(1-a)
    rn = r + pe
    P = max(1e-6, M0[i] * rn / (chi[i] * C))

    # iteration parameters
    max_iter = 1000
    tol = 1e-8

    for it in range(max_iter):
        N_old, C_old, P_old = N, C, P

        # 1) Output (Cobb-Douglas)
        Y = A[i] * (K**a) * N**(1-a)

        # 2) Real wage (marginal product of labour)
        w = (1 - a) * A[i] * (K**a) * N**(-a)

        # 3) Real interest rate (marginal product of capital)
        r = a * A[i] * (K**(a-1)) * N**(1-a)

        # 4) Nominal interest rate
        rn = r + pe
        rn = max(rn, 1e-8)  # avoid division by zero or negative rn

        # 5) Household budget / consumption:
        #    We assume a simple static budget: C = w*N + r*K - T + PV(future net)
        #    with lump-sum tax T = G0[i] (government spending financed lump-sum).
        #    Add the small present-value of (Yf - Gf) as you had previously.
        C = w * N + r * K - G0[i] + (Yf[i] - Gf) / (1 + r)

        # Guard against non-positive consumption
        C = max(C, 1e-8)

        # 6) Labour supply from FOC of utility:
        #    N^phi = w / C  => N = (w / C)^(1/phi)
        #    (this corrects the sign/power used previously)
        N = (w / C) ** (1.0 / phi)
        # keep N in a reasonable range
        N = np.clip(N, 1e-8, 1.0)

        # 7) Goods market: Investment is residual
        I = Y - C - G0[i]

        # Optional: if you want investment non-negative, clip:
        # I = max(I, 0.0)

        # 8) Price level from money-in-utility condition:
        #    Real money demand: M0/P = chi * C / rn  => P = M0 * rn / (chi * C)
        P = max(1e-8, M0[i] * rn / (chi[i] * C))

        # Convergence check (max change small)
        if max(abs(N - N_old), abs(C - C_old), abs(P - P_old)) < tol:
            #print(f"scenario {i+1} converged in {it+1} iters")
            break

    # Save results
    Y_star[i] = Y
    w_star[i] = w
    C_star[i] = C
    I_star[i] = I
    r_star[i] = r
    rn_star[i] = rn
    N_star[i] = N
    P_star[i] = P






# Plot results
scenario_names = ["1: Baseline", "2: Increase in M0", "3: Increase in G0", 
                  "4: Increase in A", "5: Decrease in Yf", "6: Increase in chi"]

scenario_short = ["1", "2", "3", "4", "5", "6"]

fig, axs = plt.subplots(2, 3, figsize=(12, 12))
axs = axs.flatten()

variables = [
    (Y_star, "Output (Y)"),
    (P_star, "Price Level (P)"),
    (C_star, "Consumption (C)"),
    (I_star, "Investment (I)"),
    (N_star, "Labour (N)"),
    (r_star, "Real Interest Rate (r)")
    
]

for i, (data, title) in enumerate(variables):
    axs[i].bar(scenario_short, data)
    axs[i].set_title(title)
    axs[i].tick_params(axis='x')

legend_text = (
    "1: Baseline   |   "
    "2: ↑ Money Supply   |   "
    "3: ↑ Government Spending   |   "
    "4: ↑ Productivity   |   "
    "5: ↓ Expected Income   |   "
    "6: ↑ Leisure Preference"
)

fig.text(0.5, 0.95, legend_text, ha='center', fontsize=9)

plt.show()