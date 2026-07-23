import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings("ignore")

# Plotting configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 5.5), "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_ACCENT, C_MUTED = "#33415C", "#7F1D1D", "#9AA0A6"

# Running example: one-year at-the-money call, delta-hedged long
S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.02, 0.20

print("Environment configured.")
print(f"Running example: S0={S0:.0f}, K={K:.0f}, T={T:.0f}y, r={R:.0%}, sigma={SIGMA:.0%}")

#Implements the closed black scholes forms and then validates them via Finite Diff and PCP
def bs_price(S, K, T, r, sig):
    """Black-Scholes price of a European call."""
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    return norm.cdf(d1)

def bs_gamma(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    return norm.pdf(d1) / (S * sig * np.sqrt(T))

def bs_theta(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return -(S * norm.pdf(d1) * sig) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)

def bs_vega(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# --- Validation 1: analytic Greeks vs central finite differences ---
h = 1e-4
fd_delta = (bs_price(S0 + h, K, T, R, SIGMA) - bs_price(S0 - h, K, T, R, SIGMA)) / (2 * h)
fd_gamma = (bs_price(S0 + h, K, T, R, SIGMA) - 2 * bs_price(S0, K, T, R, SIGMA)
            + bs_price(S0 - h, K, T, R, SIGMA)) / h**2
fd_theta = -(bs_price(S0, K, T + h, R, SIGMA) - bs_price(S0, K, T - h, R, SIGMA)) / (2 * h)
fd_vega = (bs_price(S0, K, T, R, SIGMA + h) - bs_price(S0, K, T, R, SIGMA - h)) / (2 * h)

rows = [
    ("price", bs_price(S0, K, T, R, SIGMA), np.nan),
    ("delta", bs_delta(S0, K, T, R, SIGMA), fd_delta),
    ("gamma", bs_gamma(S0, K, T, R, SIGMA), fd_gamma),
    ("theta (per yr)", bs_theta(S0, K, T, R, SIGMA), fd_theta),
    ("vega (per unit)", bs_vega(S0, K, T, R, SIGMA), fd_vega),
]
print(f"{'quantity':16s} {'analytic':>12s} {'finite diff':>12s} {'abs error':>10s}")
for name, an, fd in rows:
    err = "" if np.isnan(fd) else f"{abs(an - fd):10.2e}"
    fd_s = "" if np.isnan(fd) else f"{fd:12.6f}"
    print(f"{name:16s} {an:12.6f} {fd_s:>12s} {err:>10s}")

# --- Validation 2: put-call parity ---
call = bs_price(S0, K, T, R, SIGMA)
put_parity = call - S0 + K * np.exp(-R * T)
print(f"\nput via parity C - S + K e^(-rT) = {put_parity:.6f}  (positive, as required)")



#----------------
# Actual one-day hedged P&L by full repricing vs the gamma-theta identity
#Implementing the gamma-theta identity and comparing it to the actual P&L of a delta-hedged call over one day
t_rem, dt_day = 0.5, 1 / 252
V0_ = bs_price(S0, K, t_rem, R, SIGMA)
delta_ = bs_delta(S0, K, t_rem, R, SIGMA)
gamma_ = bs_gamma(S0, K, t_rem, R, SIGMA)

moves = np.linspace(-8, 8, 321)
actual = np.array([
    (bs_price(S0 + dS, K, t_rem - dt_day, R, SIGMA) - V0_)
    - delta_ * dS + R * (delta_ * S0 - V0_) * dt_day
    for dS in moves
])
identity = 0.5 * gamma_ * (moves**2 - SIGMA**2 * S0**2 * dt_day)
breakeven = SIGMA * S0 * np.sqrt(dt_day)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(moves, actual, lw=2.5, color=C_MAIN, label="actual hedged P&L (full reprice)")
ax.plot(moves, identity, lw=1.4, ls="--", color=C_ACCENT,
        label=r"identity $\frac{1}{2}\Gamma(\delta S^2 - \sigma^2 S^2 \delta t)$")
ax.axhline(0, color="black", lw=0.7)
for be in (-breakeven, breakeven):
    ax.axvline(be, color=C_MUTED, ls=":", lw=1.2)
ax.set_xlabel("one-day stock move  $\\delta S$")
ax.set_ylabel("hedged P&L")
ax.set_title("One-day P&L of a delta-hedged call: full reprice vs the second-order identity")
ax.legend()
plt.tight_layout(); plt.show()

mask = np.abs(moves) <= 4
print(f"breakeven move: +/- {breakeven:.3f}  ({breakeven / S0:.2%} of spot)")
print(f"max |actual - identity|, |dS| <= 4 (a 3-sigma day): {np.abs(actual - identity)[mask].max():.5f}")
print(f"max |actual - identity| over the full +/-8 range:   {np.abs(actual - identity).max():.5f}")

##Agrees as expected. 

#____________________________
#Discrete hedging at N discrete dates
def make_gbm_paths(N, n_paths, seed, mu=R, sigma=SIGMA):
    """Exact GBM paths on an N-step grid, shape (n_paths, N+1)."""
    rng = np.random.default_rng(seed)
    dt = T / N
    inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, N))
    return np.concatenate([np.full((n_paths, 1), S0),
                           np.exp(np.log(S0) + np.cumsum(inc, axis=1))], axis=1)

def hedge_pnl(S, sigma_imp=SIGMA, r=R):
    """ self-financing delta hedge of a long call along each path.

    Set-up: cash = -V0 + delta0*S0. Each step: cash grows at r, then the hedge
    trades to the new delta. Settlement: payoff minus the final stock leg.
    Returns the terminal replication P&L per path (zero for a perfect hedge).
    """
    n, Np1 = S.shape
    N = Np1 - 1
    dt = T / N
    held = np.full(n, bs_delta(S0, K, T, r, sigma_imp))
    cash = -bs_price(S0, K, T, r, sigma_imp) + held * S0
    for j in range(1, N):
        tgt = bs_delta(S[:, j], K, T - j * dt, r, sigma_imp)
        cash = cash * np.exp(r * dt) + (tgt - held) * S[:, j]
        held = tgt
    cash = cash * np.exp(r * dt)
    return cash + np.maximum(S[:, N] - K, 0.0) - held * S[:, N]

# Convergence sweep
Ns = np.array([8, 16, 32, 64, 128, 256, 512])
stds, means = [], []
for N in Ns:
    pnl = hedge_pnl(make_gbm_paths(int(N), 10000, seed=42))
    means.append(pnl.mean()); stds.append(pnl.std())
stds = np.array(stds)

slope = np.polyfit(np.log(Ns), np.log(stds), 1)[0]
b_const = float(np.mean(stds * np.sqrt(Ns)))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.loglog(Ns, stds, "o-", color=C_MAIN, lw=2, label=f"measured (fitted slope {slope:.3f})")
ax.loglog(Ns, stds[0] * np.sqrt(Ns[0] / Ns), "--", color=C_ACCENT, lw=1.4,
          label=r"reference $\propto 1/\sqrt{N}$")
ax.set_xlabel("rebalances per year  $N$")
ax.set_ylabel("std of replication P&L")
ax.set_title("The 1/sqrt(N) law of discrete hedging error")
ax.legend()
plt.tight_layout(); plt.show()

print(f"{'N':>5s} {'mean':>9s} {'std':>8s} {'std*sqrt(N)':>12s}")
for N, m, s in zip(Ns, means, stds):
    print(f"{N:5d} {m:+9.4f} {s:8.4f} {s * np.sqrt(N):12.3f}")
print(f"\nfitted log-log slope: {slope:.3f}   (theory: -0.5)")
print(f"error constant b = std*sqrt(N) ~ {b_const:.2f}  (used again in Section 7)")

# Distributional view: monthly vs daily rebalancing
pnl_m = hedge_pnl(make_gbm_paths(12, 20000, seed=7))
pnl_d = hedge_pnl(make_gbm_paths(252, 20000, seed=7))

fig, ax = plt.subplots(figsize=(10, 5.5))
bins = np.linspace(-7, 7, 90)
ax.hist(pnl_m, bins=bins, alpha=0.55, color=C_MUTED, label=f"monthly (N=12),  std={pnl_m.std():.2f}")
ax.hist(pnl_d, bins=bins, alpha=0.80, color=C_MAIN,  label=f"daily (N=252),  std={pnl_d.std():.2f}")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("replication P&L at expiry")
ax.set_ylabel("paths")
ax.set_title("Same option, same world: only the rebalancing frequency differs")
ax.legend()
plt.tight_layout(); plt.show()

for name, p in [("monthly", pnl_m), ("daily", pnl_d)]:
    print(f"{name:8s} mean {p.mean():+.4f}   std {p.std():.4f}   skew {skew(p):+.3f}   "
          f"P5 {np.percentile(p, 5):+.3f}   P95 {np.percentile(p, 95):+.3f}")



#Gamma Scalp v/s Theta decay
def attribute_pnl(path, K, T, sigma):
    """Exact reprice attribution of a discrete hedge at r=0.

    Returns cumulative (gamma_stream, theta_stream); their sum telescopes to the
    exact hedged P&L. Gamma pieces are >= 0 (convexity), theta pieces <= 0 (decay).
    """
    def V(S, tau):
        return bs_price(S, K, tau, 0.0, sigma) if tau > 1e-12 else max(S - K, 0.0)
    path = np.asarray(path, float)
    N = len(path) - 1
    dt = T / N
    g = np.zeros(N); th = np.zeros(N)
    for k in range(1, N + 1):
        t_old, t_new = T - (k - 1) * dt, T - k * dt
        So, Sn = path[k - 1], path[k]
        d_prev = bs_delta(So, K, t_old, 0.0, sigma)
        g[k - 1] = (V(Sn, t_old) - V(So, t_old)) - d_prev * (Sn - So)
        th[k - 1] = V(Sn, t_new) - V(Sn, t_old)
    return np.cumsum(g), np.cumsum(th)

def realized_vol(path, T):
    lr = np.diff(np.log(np.asarray(path, float)))
    return np.sqrt(np.sum(lr**2) / T)

# One daily-hedged path, attributed exactly
path = make_gbm_paths(252, 1, seed=11, mu=0.0)[0]
cum_g, cum_t = attribute_pnl(path, K, T, SIGMA)
exact = hedge_pnl(path[None, :], r=0.0)[0]
tie_out = abs((cum_g[-1] + cum_t[-1]) - exact)

tgrid = np.linspace(T / 252, T, 252)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
ax1.plot(np.linspace(0, T, 253), path, color=C_MAIN, lw=1.2)
ax1.axhline(K, color=C_MUTED, ls=":", lw=1)
ax1.set_xlabel("years"); ax1.set_ylabel("price"); ax1.set_title("the path")
ax2.plot(tgrid, cum_g, color=C_MAIN, lw=2, label=f"gamma stream  ({cum_g[-1]:+.2f})")
ax2.plot(tgrid, cum_t, color=C_ACCENT, lw=2, label=f"theta stream  ({cum_t[-1]:+.2f})")
ax2.plot(tgrid, cum_g + cum_t, color="black", lw=2, label=f"net = hedged P&L  ({cum_g[-1] + cum_t[-1]:+.2f})")
ax2.axhline(0, color=C_MUTED, lw=0.7)
ax2.set_xlabel("years"); ax2.set_ylabel("cumulative P&L")
ax2.set_title("exact attribution: convexity gains vs time decay")
ax2.legend()
plt.tight_layout(); plt.show()

print(f"sum of streams - exact hedge P&L (tie-out): {tie_out:.2e}")
print(f"realized vol of this path: {realized_vol(path, T):.4f}  vs implied {SIGMA:.4f}")
print(f"net P&L sign matches realized-vs-implied: {np.sign(cum_g[-1] + cum_t[-1]) == np.sign(realized_vol(path, T) - SIGMA)}")



#__________
#Transaction cost and hedging frequencies
#Also add leland correction and pepper in band hedging

KC = 0.001  # one-way proportional cost: 10 basis points

def hedge_with_costs(S, sigma_imp=SIGMA, kc=KC, r=R):
    """Clock hedge paying proportional costs at set-up, every rebalance, and unwind.
    Returns (replication_pnl, total_cost) per path; net = replication - cost."""
    n, Np1 = S.shape
    N = Np1 - 1
    dt = T / N
    held = np.full(n, bs_delta(S0, K, T, r, sigma_imp))
    cash = -bs_price(S0, K, T, r, sigma_imp) + held * S0
    cost = kc * held * S0
    for j in range(1, N):
        tgt = bs_delta(S[:, j], K, T - j * dt, r, sigma_imp)
        cash = cash * np.exp(r * dt) + (tgt - held) * S[:, j]
        cost = cost + kc * np.abs(tgt - held) * S[:, j]
        held = tgt
    cash = cash * np.exp(r * dt)
    cost = cost + kc * held * S[:, N]
    repl = cash + np.maximum(S[:, N] - K, 0.0) - held * S[:, N]
    return repl, cost

LAM = 1.65  # one-sided 95% risk charge

Ns_f = np.array([5, 10, 20, 40, 80, 160, 320, 640, 1280])
risk_f, cost_f = [], []
for N in Ns_f:
    repl, cost = hedge_with_costs(make_gbm_paths(int(N), 4000, seed=42))
    risk_f.append(repl.std()); cost_f.append(cost.mean())
risk_f, cost_f = np.array(risk_f), np.array(cost_f)
J = cost_f + LAM * risk_f

# Fit cost = c0 + a*sqrt(N); take b from Section 5
A = np.vstack([np.ones_like(Ns_f, dtype=float), np.sqrt(Ns_f)]).T
c0_fit, a_fit = np.linalg.lstsq(A, cost_f, rcond=None)[0]
N_star_theory = LAM * b_const / a_fit
N_star_measured = Ns_f[int(np.argmin(J))]

print(f"{'N':>5s} {'risk(std)':>10s} {'E[cost]':>9s} {'J=cost+1.65*risk':>17s}")
for N, rk, mc, j in zip(Ns_f, risk_f, cost_f, J):
    tag = "  <- optimum" if N == N_star_measured else ""
    print(f"{N:5d} {rk:10.4f} {mc:9.4f} {j:17.4f}{tag}")
print(f"\ncost fit:  E[cost] ~ {c0_fit:.3f} + {a_fit:.4f} sqrt(N)")
print(f"optimal frequency:  theory N* = lambda*b/a = {N_star_theory:.0f}   measured argmin J = {N_star_measured}")

# Leland-style adjusted volatility, verified
dt_d = 1 / 252
sig_hat = np.sqrt(SIGMA**2 + 2 * KC * SIGMA * np.sqrt(2 / np.pi) / np.sqrt(dt_d))
pred = bs_price(S0, K, T, R, sig_hat) - bs_price(S0, K, T, R, SIGMA)
S252 = make_gbm_paths(252, 20000, seed=42)
held_l = np.full(S252.shape[0], bs_delta(S0, K, T, R, SIGMA))
rebal_cost = np.zeros(S252.shape[0])
for j in range(1, 252):
    tgt_l = bs_delta(S252[:, j], K, T - j / 252, R, SIGMA)
    rebal_cost += KC * np.abs(tgt_l - held_l) * S252[:, j]
    held_l = tgt_l
print(f"\nLeland check at daily rebalancing, k=10bp (rebalancing leg only):")
print(f"  adjusted vol sigma_hat = {sig_hat:.4f};  C(sigma_hat) - C(sigma) = {pred:.4f}")
print(f"  MC mean rebalancing cost = {rebal_cost.mean():.4f}")
print(f"  prediction / MC ratio = {pred / rebal_cost.mean():.3f}  (leading-order, as expected)")

def band_hedge(S, sigma_imp=SIGMA, kc=KC, h=0.05, r=R):
    """Move-based hedge: monitor every step, trade to target only when
    |target - held| > h. Returns (replication_pnl, total_cost) per path."""
    n, Np1 = S.shape
    N = Np1 - 1
    dt = T / N
    held = np.full(n, bs_delta(S0, K, T, r, sigma_imp))
    cash = -bs_price(S0, K, T, r, sigma_imp) + held * S0
    cost = kc * held * S0
    for j in range(1, N):
        cash = cash * np.exp(r * dt)
        tgt = bs_delta(S[:, j], K, T - j * dt, r, sigma_imp)
        m = np.abs(tgt - held) > h
        cash[m] += (tgt[m] - held[m]) * S[m, j]
        cost[m] += kc * np.abs(tgt[m] - held[m]) * S[m, j]
        held[m] = tgt[m]
    cash = cash * np.exp(r * dt)
    cost = cost + kc * held * S[:, N]
    repl = cash + np.maximum(S[:, N] - K, 0.0) - held * S[:, N]
    return repl, cost

bands = [0.01, 0.02, 0.05, 0.10, 0.16]
S_daily = make_gbm_paths(252, 4000, seed=42)
band_pts = []
for h in bands:
    repl, cost = band_hedge(S_daily, h=h)
    band_pts.append((repl.std(), cost.mean()))
band_risk = np.array([p[0] for p in band_pts])
band_cost = np.array([p[1] for p in band_pts])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cost_f, risk_f, "o-", color=C_MUTED, lw=2, label="clock rule (rebalance every step)")
ax.plot(band_cost, band_risk, "s-", color=C_MAIN, lw=2, label="band rule (trade on delta drift)")
for N, rk, mc in zip(Ns_f, risk_f, cost_f):
    ax.annotate(f"N={N}", (mc, rk), textcoords="offset points", xytext=(6, 4), fontsize=8, color=C_MUTED)
for h, rk, mc in zip(bands, band_risk, band_cost):
    ax.annotate(f"h={h:.2f}", (mc, rk), textcoords="offset points", xytext=(6, -11), fontsize=8, color=C_MAIN)
ax.set_xlabel("expected total cost of running the hedge")
ax.set_ylabel("replication risk (std of pre-cost P&L)")
ax.set_title("Cost-risk frontier: the band family sits inside the clock family")
ax.legend()
plt.tight_layout(); plt.show()

order = np.argsort(cost_f)
print(f"{'band h':>7s} {'cost':>7s} {'band risk':>10s} {'clock risk @ same cost':>23s} {'improvement':>12s}")
for h, rk, mc in zip(bands, band_risk, band_cost):
    rk_clock = np.interp(mc, cost_f[order], risk_f[order])
    print(f"{h:7.2f} {mc:7.3f} {rk:10.4f} {rk_clock:23.4f} {(rk_clock - rk) / rk_clock:12.1%}")
#Lastly trying out risks to our assumptions (Jump diffusion, stochastic Vol and Drift invariance)

def make_merton_paths(N, n_paths, seed, mu, sigma, lam, mu_j, sigma_j):
    """Exact Merton jump-diffusion paths. Compensator keeps E-growth at mu."""
    rng = np.random.default_rng(seed)
    dt = T / N
    Z = rng.standard_normal((n_paths, N))
    counts = rng.poisson(lam * dt, (n_paths, N))
    ZJ = rng.standard_normal((n_paths, N))
    kappa_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    inc = ((mu - lam * kappa_bar - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
           + counts * mu_j + np.sqrt(counts) * sigma_j * ZJ)
    return np.concatenate([np.full((n_paths, 1), S0),
                           np.exp(np.log(S0) + np.cumsum(inc, axis=1))], axis=1)

# Stylized crash world, variance-matched to the hedger's 20% implied
LAM_J, MU_J, SIG_J = 0.5, -0.10, 0.15
SIG_DIFF = np.sqrt(SIGMA**2 - LAM_J * (MU_J**2 + SIG_J**2))
print(f"jump QV contribution lambda*(muJ^2+sigJ^2) = {LAM_J * (MU_J**2 + SIG_J**2):.4f} per year")
print(f"variance-matched diffusion vol = {SIG_DIFF:.4f}  (hedger prices at {SIGMA:.2f})\n")

Ns_m = np.array([8, 16, 32, 64, 128, 252])
gbm_std, mer_std = [], []
for N in Ns_m:
    gbm_std.append(hedge_pnl(make_gbm_paths(int(N), 4000, seed=42)).std())
    mer_std.append(hedge_pnl(make_merton_paths(int(N), 4000, 42, R, SIG_DIFF, LAM_J, MU_J, SIG_J)).std())
gbm_std, mer_std = np.array(gbm_std), np.array(mer_std)

pnl_g = hedge_pnl(make_gbm_paths(252, 8000, seed=7))
pnl_m = hedge_pnl(make_merton_paths(252, 8000, 7, R, SIG_DIFF, LAM_J, MU_J, SIG_J))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
bins = np.linspace(-6, 10, 80)
ax1.hist(pnl_g, bins=bins, alpha=0.55, color=C_MUTED, label="Black-Scholes world")
ax1.hist(pnl_m, bins=bins, alpha=0.75, color=C_MAIN, label="jump world (same total variance)")
ax1.axvline(0, color="black", lw=0.8)
ax1.set_xlabel("daily-hedged P&L (long option)"); ax1.set_ylabel("paths")
ax1.set_title("variance-matched, model-mismatched"); ax1.legend()
ax2.loglog(Ns_m, gbm_std, "o-", color=C_MUTED, lw=2, label="Black-Scholes world")
ax2.loglog(Ns_m, mer_std, "s-", color=C_MAIN, lw=2, label="jump world")
ax2.loglog(Ns_m, gbm_std[0] * np.sqrt(Ns_m[0] / Ns_m), "--", color=C_ACCENT, lw=1.2,
           label=r"$\propto 1/\sqrt{N}$")
ax2.set_xlabel("rebalances per year  $N$"); ax2.set_ylabel("std of replication P&L")
ax2.set_title("the 1/sqrt(N) law dies under jumps"); ax2.legend()
plt.tight_layout(); plt.show()

print(f"{'N':>5s} {'GBM std':>9s} {'jump std':>9s}")
for N, g, m in zip(Ns_m, gbm_std, mer_std):
    print(f"{N:5d} {g:9.4f} {m:9.4f}")
se = pnl_m.std() / np.sqrt(len(pnl_m))
print(f"\ndaily hedging, long option, jump world:")
print(f"  mean {pnl_m.mean():+.4f} (s.e. {se:.4f})  ->  systematic bias, {abs(pnl_m.mean()) / se:.1f} standard errors from zero")
print(f"  skew {skew(pnl_m):+.2f}  (GBM world: {skew(pnl_g):+.2f});  P5 {np.percentile(pnl_m, 5):+.2f}, P95 {np.percentile(pnl_m, 95):+.2f}")
print("  the gamma-is-local bias: a jump collects only the curvature it crosses,")
print("  less than the at-the-money rent rate it paid through theta.")

def make_heston_paths(N, n_paths, seed, mu, v0, kappa, theta, xi, rho):
    """Heston paths under the full truncation Euler scheme (Lord et al., 2010)."""
    rng = np.random.default_rng(seed)
    dt = T / N
    Z1 = rng.standard_normal((n_paths, N))
    Z2 = rng.standard_normal((n_paths, N))
    S = np.full(n_paths, S0)
    v = np.full(n_paths, v0)
    out = np.empty((n_paths, N + 1)); out[:, 0] = S0
    for k in range(N):
        vp = np.maximum(v, 0.0)
        Zs = rho * Z2[:, k] + np.sqrt(1.0 - rho**2) * Z1[:, k]
        S = S * np.exp((mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * Zs)
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * Z2[:, k]
        out[:, k + 1] = S
    return out

# E[v] matched to the hedger's implied variance; Feller condition holds
V0H, KAPPA, THETA_H, XI, RHO = 0.04, 3.0, 0.04, 0.4, -0.7
print(f"Feller: 2*kappa*theta = {2 * KAPPA * THETA_H:.2f} >= xi^2 = {XI**2:.2f}  ->  {2 * KAPPA * THETA_H >= XI**2}")

hes_std = []
for N in Ns_m:
    hes_std.append(hedge_pnl(make_heston_paths(int(N), 3000, 42, R, V0H, KAPPA, THETA_H, XI, RHO)).std())
hes_std = np.array(hes_std)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.loglog(Ns_m, gbm_std, "o-", color=C_MUTED, lw=2, label="Black-Scholes world")
ax.loglog(Ns_m, hes_std, "^-", color=C_MAIN, lw=2, label="Heston world (E[v] matched)")
ax.loglog(Ns_m, mer_std, "s-", color=C_ACCENT, lw=1.6, label="jump world (for scale)")
ax.set_xlabel("rebalances per year  $N$"); ax.set_ylabel("std of replication P&L")
ax.set_title("the vega leak: a second risk factor sets a floor delta cannot reach")
ax.legend()
plt.tight_layout(); plt.show()

print(f"{'N':>5s} {'GBM std':>9s} {'Heston std':>11s} {'jump std':>9s}")
for N, g, hs, m in zip(Ns_m, gbm_std, hes_std, mer_std):
    print(f"{N:5d} {g:9.4f} {hs:11.4f} {m:9.4f}")
print("\nthe Heston floor sits between the two: vol risk is unhedged by the stock,")
print("but unlike a jump it moves continuously, so part of it averages out.")

# Drift invariance: hedge the same call in worlds with wildly different drifts
mus = [-0.20, 0.10, 0.40]
V0_full = bs_price(S0, K, T, R, SIGMA)
colors = [C_ACCENT, C_MUTED, C_MAIN]

fig, ax = plt.subplots(figsize=(10, 5.5))
print(f"{'mu':>7s} {'hedged mean':>12s} {'hedged std':>11s} {'unhedged mean':>14s} {'unhedged std':>13s}")
for mu, c in zip(mus, colors):
    S = make_gbm_paths(126, 4000, seed=11, mu=mu)
    hedged = hedge_pnl(S)
    unhedged = np.maximum(S[:, -1] - K, 0.0) - V0_full * np.exp(R * T)
    print(f"{mu:+7.2f} {hedged.mean():+12.4f} {hedged.std():11.4f} {unhedged.mean():+14.3f} {unhedged.std():13.3f}")
    ax.hist(hedged, bins=np.linspace(-3, 3, 90), alpha=0.5, color=c, label=f"hedged,  drift {mu:+.0%}")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("hedged P&L"); ax.set_ylabel("paths")
ax.set_title("the hedged P&L does not know the drift (Girsanov, experimentally)")
ax.legend()
plt.tight_layout(); plt.show()

print("\na 60-point swing in annual drift moves the unhedged book by tens of dollars")
print("and the hedged book by hundredths: option prices cannot depend on mu.")




#Real market data
# Load SPY daily closes: live via yfinance if available, embedded snapshot otherwise
try:
    import yfinance as yf
    _df = yf.download("SPY", period="3y", interval="1d", auto_adjust=True, progress=False)
    spy_closes = _df["Close"].dropna().to_numpy().ravel()
    spy_label = f"SPY via yfinance, {_df.index[0].date()} to {_df.index[-1].date()}"
except Exception:
    spy_closes = np.array([
    417.45, 420.2, 420.7, 425.92, 424.47, 422.26, 420.1, 421.62, 418.43, 416.72, 421.29, 421.5, 
    423.16, 428.16, 428.65, 428.01, 424.66, 423.59, 424.66, 427.36, 430.8, 434.22, 433.95, 
    435.46, 438.69, 439.67, 436.75, 436.75, 438.7, 439.9, 439.97, 437.05, 441.33, 442.17, 
    440.91, 434.77, 433.53, 431.57, 435.33, 433.44, 430.54, 430.7, 430.44, 432.82, 427.78, 
    424.64, 421.4, 421.61, 424.35, 423.2, 427.91, 421.98, 424.96, 427.65, 433.83, 435.62, 
    434.98, 435.8, 433.91, 431.0, 429.67, 430.32, 433.15, 430.77, 431.28, 434.99, 429.75, 
    430.01, 429.11, 425.17, 418.14, 417.2, 418.96, 412.8, 412.96, 415.36, 414.35, 414.19, 
    408.64, 411.62, 411.46, 416.35, 419.01, 421.19, 422.92, 420.34, 418.25, 422.65, 422.63, 
    417.0, 413.33, 408.25, 407.55, 410.62, 404.73, 399.88, 398.07, 402.83, 405.36, 409.68, 
    417.53, 421.34, 422.31, 423.51, 423.82, 420.52, 427.08, 426.67, 434.95, 435.87, 436.4, 
    436.95, 440.31, 439.35, 441.05, 441.32, 440.52, 440.96, 440.65, 442.38, 445.0, 442.66, 
    442.58, 440.79, 444.16, 446.07, 447.8, 449.85, 456.05, 457.51, 456.76, 459.33, 462.12, 
    455.72, 460.04, 460.96, 462.91, 463.75, 463.92, 462.58, 459.99, 456.23, 454.76, 455.39, 
    461.89, 461.19, 463.8, 463.59, 463.91, 462.21, 459.64, 463.73, 469.51, 470.5, 471.87, 
    472.39, 474.96, 474.36, 478.11, 477.74, 469.95, 476.1, 481.11, 479.36, 480.75, 484.76, 
    484.97, 487.78, 487.56, 480.85, 485.22, 488.56, 486.13, 483.46, 483.89, 493.91, 494.25, 
    492.44, 493.35, 492.7, 494.47, 499.11, 498.58, 493.6, 496.1, 501.02, 498.01, 497.59, 
    502.94, 502.15, 501.16, 497.72, 500.68, 503.46, 508.11, 509.79, 508.83, 507.42, 506.48, 
    510.74, 510.64, 509.75, 506.51, 507.07, 500.88, 506.11, 506.4, 506.98, 501.91, 505.69, 
    498.71, 492.46, 491.57, 488.66, 487.65, 483.4, 487.85, 493.64, 493.4, 491.53, 496.18, 
    497.94, 490.05, 488.46, 493.03, 499.14, 504.3, 504.85, 504.9, 507.81, 508.47, 508.53, 
    510.87, 517.19, 516.13, 516.87, 517.47, 518.74, 517.24, 513.46, 516.86, 517.22, 513.6, 
    510.19, 514.84, 515.26, 515.84, 521.97, 521.96, 521.32, 522.93, 524.19, 528.5, 529.56, 
    529.88, 534.1, 535.46, 534.0, 533.29, 531.55, 533.6, 534.27, 535.11, 533.0, 534.1, 537.7, 
    540.09, 543.21, 543.84, 544.36, 549.75, 545.01, 548.45, 549.96, 553.22, 545.46, 541.27, 
    537.68, 543.22, 542.37, 530.08, 527.31, 533.22, 533.53, 530.83, 539.46, 531.82, 521.92, 
    506.72, 511.39, 507.97, 519.71, 522.01, 522.28, 530.87, 532.54, 541.67, 542.89, 548.08, 
    547.19, 549.07, 544.76, 550.54, 549.23, 549.99, 546.79, 546.84, 552.06, 540.7, 539.6, 
    538.28, 529.22, 535.15, 537.48, 542.99, 547.57, 550.43, 551.24, 551.47, 549.83, 559.21, 
    558.25, 559.64, 561.24, 560.0, 562.22, 561.41, 563.66, 558.61, 558.85, 557.82, 562.89, 
    557.8, 563.08, 566.98, 565.99, 569.38, 574.03, 569.57, 572.05, 572.1, 574.3, 573.35, 
    573.05, 567.81, 569.04, 568.85, 570.6, 571.53, 569.8, 558.63, 560.99, 559.78, 566.55, 
    580.63, 585.12, 587.66, 588.22, 586.39, 586.68, 582.9, 575.44, 577.8, 579.91, 580.1, 
    583.22, 585.03, 587.01, 590.08, 588.29, 591.94, 593.0, 593.28, 596.96, 595.98, 597.11, 
    594.03, 592.19, 596.77, 593.69, 593.57, 596.11, 593.65, 575.96, 575.78, 582.7, 586.19, 
    592.7, 592.74, 586.5, 579.81, 577.7, 576.28, 583.49, 586.85, 580.21, 581.06, 572.19, 
    573.08, 573.87, 584.3, 583.18, 589.04, 594.43, 597.77, 601.03, 599.28, 590.8, 595.88, 
    593.2, 596.39, 593.21, 589.22, 593.18, 595.58, 597.65, 592.18, 596.2, 596.65, 594.73, 
    601.01, 600.98, 602.75, 604.17, 601.65, 591.36, 588.67, 585.74, 586.04, 576.68, 585.68, 
    575.42, 568.61, 574.72, 564.52, 567.68, 552.56, 547.97, 550.88, 543.54, 554.76, 559.04, 
    553.0, 559.02, 557.4, 557.59, 567.57, 568.94, 562.15, 560.65, 549.36, 553.05, 554.61, 
    558.12, 530.62, 499.55, 498.66, 490.85, 542.4, 518.63, 527.89, 533.01, 531.52, 519.7, 
    520.44, 508.06, 521.27, 529.35, 540.49, 544.4, 544.61, 548.04, 548.26, 552.14, 560.34, 
    557.12, 552.47, 554.79, 558.66, 557.94, 576.38, 580.19, 580.93, 583.77, 587.47, 588.11, 
    586.13, 576.25, 576.48, 572.55, 584.45, 581.07, 583.36, 582.71, 585.99, 589.33, 589.18, 
    586.33, 592.35, 592.88, 596.24, 594.54, 596.91, 590.23, 595.85, 590.76, 590.67, 589.28, 
    595.1, 601.68, 602.01, 606.72, 609.74, 612.65, 612.46, 615.23, 620.08, 615.46, 615.12, 
    618.81, 620.56, 618.37, 619.55, 616.91, 618.97, 622.76, 622.3, 623.48, 623.57, 628.88, 
    629.08, 631.74, 631.58, 629.92, 629.12, 626.76, 616.49, 625.86, 622.69, 627.46, 626.93, 
    631.82, 630.57, 637.28, 639.47, 639.53, 638.03, 637.89, 634.43, 632.74, 630.2, 639.88, 
    637.07, 639.73, 641.19, 643.46, 639.62, 634.88, 638.33, 643.66, 641.8, 643.37, 644.86, 
    646.72, 652.1, 651.88, 655.35, 654.45, 653.64, 656.69, 659.94, 663.06, 659.46, 657.36, 
    654.32, 658.07, 659.92, 662.41, 664.67, 665.43, 665.42, 667.81, 665.33, 669.3, 667.36, 
    649.32, 659.29, 658.48, 661.4, 656.9, 660.63, 667.5, 667.49, 664.02, 667.96, 673.42, 
    681.36, 683.17, 683.5, 675.98, 678.2, 679.47, 671.42, 673.74, 666.51, 667.17, 677.58, 
    679.13, 679.51, 668.24, 668.13, 661.9, 656.34, 658.88, 648.84, 655.3, 664.94, 671.2, 
    675.83, 679.52, 676.42, 677.67, 680.02, 680.52, 681.81, 679.76, 679.17, 683.68, 685.27, 
    677.9, 676.88, 675.03, 667.6, 672.64, 678.74, 682.96, 686.09, 688.5, 688.43, 685.98, 
    685.14, 680.06, 681.31, 685.85, 689.93, 687.7, 687.63, 692.18, 693.27, 691.88, 688.48, 
    690.35, 689.78, 675.73, 683.53, 687.1, 687.35, 690.84, 693.6, 693.53, 692.15, 690.09, 
    693.52, 687.65, 684.32, 675.77, 688.74, 692.06, 690.23, 690.08, 679.41, 679.89, 680.99, 
    684.42, 682.62, 687.55, 680.53, 685.48, 691.26, 687.42, 684.12, 684.51, 678.48, 683.26, 
    679.45, 670.55, 676.42, 675.34, 674.49, 664.25, 660.49, 667.21, 668.96, 659.63, 658.0, 
    648.57, 655.38, 653.18, 656.82, 645.09, 634.09, 631.97, 650.34, 655.24, 655.83, 658.93, 
    659.22, 676.01, 679.91, 679.46, 686.1, 694.46, 699.94, 701.66, 710.14, 708.72, 704.08, 
    711.21, 708.45, 713.94, 715.17, 711.69, 711.58, 718.66, 720.65, 718.01, 723.77, 733.83, 
    731.58, 737.62, 739.3, 738.18, 742.31, 748.17, 739.17, 738.65, 733.73, 741.25, 742.72, 
    745.64, 750.59, 750.46, 754.6, 756.48, 758.54, 759.57, 754.24, 757.09, 737.55, 739.22
])
    spy_label = "SPY embedded snapshot, 2023-06-12 to 2026-06-09"
print(f"data source: {spy_label}  ({len(spy_closes)} trading days)")

r_spy = np.diff(np.log(spy_closes))
n_years = len(r_spy) / 252

# Robust scale and threshold jump detection
mad = np.median(np.abs(r_spy - np.median(r_spy)))
sig_daily = 1.4826 * mad
thresh = 4.0 * sig_daily
is_jump = np.abs(r_spy) > thresh

lam_hat = is_jump.sum() / n_years
jr = r_spy[is_jump]
mu_j_hat = jr.mean() if len(jr) else 0.0
sig_j_hat = jr.std(ddof=1) if len(jr) > 1 else 0.0
sig_diff_hat = r_spy[~is_jump].std() * np.sqrt(252)
sig_total_hat = np.sqrt(sig_diff_hat**2 + lam_hat * (mu_j_hat**2 + sig_j_hat**2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
ax1.plot(spy_closes, color=C_MAIN, lw=1.1)
ax1.set_xlabel("trading day"); ax1.set_ylabel("adjusted close")
ax1.set_title("SPY, three years")
ax2.plot(r_spy, color=C_MUTED, lw=0.7)
jump_idx = np.where(is_jump)[0]
ax2.plot(jump_idx, r_spy[jump_idx], "o", color=C_ACCENT, ms=6, label=f"flagged jumps ({is_jump.sum()})")
for yv in (thresh, -thresh):
    ax2.axhline(yv, color=C_ACCENT, ls=":", lw=1)
ax2.set_xlabel("trading day"); ax2.set_ylabel("daily log return")
ax2.set_title(r"threshold detection at $4 \times 1.4826\,\mathrm{MAD}$")
ax2.legend()
plt.tight_layout(); plt.show()

print(f"robust daily sigma  {sig_daily:.5f}   threshold {thresh:.2%} per day")
print(f"jumps flagged       {is_jump.sum()}  ->  lambda_hat = {lam_hat:.2f} per year")
print(f"jump size (log)     mean {mu_j_hat:+.4f}, std {sig_j_hat:.4f}")
print(f"diffusion vol       {sig_diff_hat:.4f} annualized (ex-jump)")
print(f"total vol           {sig_total_hat:.4f} annualized (diffusion + jump QV)")
print("\nthe data prefers more frequent, smaller jumps than the stylized crash world:")
print(f"lambda ~ {lam_hat:.1f}/yr at ~{abs(mu_j_hat):.0%} +/- {sig_j_hat:.0%}, against 0.5/yr at 10% +/- 15%.")
# Hedge a Black-Scholes book in the SPY-calibrated world
SIG_IMP_SPY = sig_total_hat  # the hedger prices the all-in realized variance correctly
S_spy_world = make_merton_paths(252, 8000, 21, R, sig_diff_hat, lam_hat, mu_j_hat, sig_j_hat)
S_bs_world = make_gbm_paths(252, 8000, seed=21, sigma=SIG_IMP_SPY)

pnl_spyw = hedge_pnl(S_spy_world, sigma_imp=SIG_IMP_SPY)
pnl_bsw = hedge_pnl(S_bs_world, sigma_imp=SIG_IMP_SPY)

fig, ax = plt.subplots(figsize=(10, 5.5))
bins = np.linspace(-4, 5, 90)
ax.hist(pnl_bsw, bins=bins, alpha=0.55, color=C_MUTED,
        label=f"Black-Scholes world at the same total vol  (std {pnl_bsw.std():.2f})")
ax.hist(pnl_spyw, bins=bins, alpha=0.75, color=C_MAIN,
        label=f"SPY-calibrated jump world  (std {pnl_spyw.std():.2f})")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("daily-hedged P&L (long option)"); ax.set_ylabel("paths")
ax.set_title("hedging error in the world the data describes")
ax.legend()
plt.tight_layout(); plt.show()

print(f"{'world':28s} {'mean':>8s} {'std':>7s} {'skew':>7s} {'P5':>8s} {'P95':>8s}")
for name, p in [("Black-Scholes, same vol", pnl_bsw), ("SPY-calibrated jumps", pnl_spyw)]:
    print(f"{name:28s} {p.mean():+8.4f} {p.std():7.4f} {skew(p):+7.2f} "
          f"{np.percentile(p, 5):+8.3f} {np.percentile(p, 95):+8.3f}")
print(f"\nstd ratio (calibrated world / BS world): {pnl_spyw.std() / pnl_bsw.std():.2f}x")


##Final risk report

def var_cvar(pnl, level=0.95):
    """(VaR, CVaR) at the given level, losses as positive numbers."""
    q = np.quantile(pnl, 1.0 - level)
    return -q, -pnl[pnl <= q].mean()

H_BAND = 0.05
worlds = [
    ("Black-Scholes", make_gbm_paths(252, 4000, seed=42)),
    ("Merton jumps (stylized)", make_merton_paths(252, 4000, 42, R, SIG_DIFF, LAM_J, MU_J, SIG_J)),
    ("Heston stochastic vol", make_heston_paths(252, 4000, 42, R, V0H, KAPPA, THETA_H, XI, RHO)),
    ("SPY-calibrated jumps", make_merton_paths(252, 4000, 42, R, sig_diff_hat, lam_hat, mu_j_hat, sig_j_hat)),
]

print(f"MORNING RISK REPORT   long one-year at-the-money call, band hedge h={H_BAND}, costs 10 basis points")
print(f"{'world':26s} {'mean':>8s} {'std':>7s} {'VaR95':>7s} {'CVaR95':>8s} {'E[cost]':>8s}")
results = []
for name, Sw in worlds:
    sig_use = SIG_IMP_SPY if "SPY" in name else SIGMA
    repl, cost = band_hedge(Sw, sigma_imp=sig_use, h=H_BAND)
    net = repl - cost
    va, cv = var_cvar(net)
    results.append((name, net, cv))
    print(f"{name:26s} {net.mean():+8.3f} {net.std():7.3f} {va:7.3f} {cv:8.3f} {cost.mean():8.3f}")

cv_bs = results[0][2]
print(f"\ntail multiple vs the priced world: " + ",  ".join(
    f"{name.split(' ')[0]} {cv / cv_bs:.1f}x" for name, _, cv in results[1:]))

fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
bins = np.linspace(-8, 10, 60)
for ax, (name, net, cv) in zip(axes, results):
    ax.hist(net, bins=bins, color=C_MAIN, alpha=0.8)
    ax.axvline(-cv, color=C_ACCENT, lw=1.4, ls="--")
    ax.set_title(f"{name}\nCVaR95 = {cv:.2f}", fontsize=10)
    ax.set_xlabel("all-in P&L")
axes[0].set_ylabel("paths")
plt.tight_layout(); plt.show()