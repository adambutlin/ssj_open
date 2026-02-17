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
b1 = np.zeros(S)  # kept but not used in new FOCs (was old leisure-pref param)

# Baseline parameterisation
M0[:] = 5
G0[:] = 1
A[:] = 2
Yf[:] = 1
b1[:] = 0.4

# Shocks / alternative scenarios
M0[1] = 6     # monetary expansion
G0[2] = 2     # fiscal expansion
A[3]  = 2.5   # productivity boost
Yf[4] = 0.2   # lower expected future income
b1[5] = 0.8   # increased original leisure pref (kept for bookkeeping)

# Constant parameters
a = 0.3        # capital elasticity
phi = 0.5      # parameter in disutility of labour (appears in N^(1+phi)/(1+phi))
beta = 0.95
chi = 0.6      # preference weight on real money balances (replaces b3)
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
    P = max(1e-6, M0[i] * rn / (chi * C))

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
        P = max(1e-8, M0[i] * rn / (chi * C))

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
                  "4: Increase in A", "5: Decrease in Yf", "6: Increase in b1"]

plt.figure(figsize=(8,4))
plt.bar(scenario_names, Y_star)
plt.ylabel('Y')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Print a quick table of results
for i, name in enumerate(scenario_names):
    print(f"{name:20s}  Y={Y_star[i]:6.3f}, C={C_star[i]:6.3f}, I={I_star[i]:6.3f}, "
          f"w={w_star[i]:6.3f}, r={r_star[i]:6.3f}, N={N_star[i]:6.3f}, P={P_star[i]:6.3f}")
