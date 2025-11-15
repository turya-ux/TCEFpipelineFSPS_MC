# ================================================================
# TKEF SED-resolved pipeline — Improved Numerical Stability Version
# ================================================================
# Changes:
# - Stable Planck function (avoid overflow)
# - Replace trapz with numpy.trapezoid
# - Correct dust temperature (40 K default)
# - Add metadata JSON for reproducibility
# =================================================================

import os, sys, json, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import integrate, stats
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM

# ----------------------------
# Try importing FSPS if present
# ----------------------------
use_fsps = False
try:
    import fsps
    use_fsps = True
except:
    use_fsps = False

# ----------------------------
#  Constants and Cosmology
# ----------------------------
OUTDIR = 'tktf_sed_output_fixed'
os.makedirs(OUTDIR, exist_ok=True)

k_B = const.k_B.value
c   = const.c.value
h   = const.h.value

cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
H0_s = cosmo.H0.to("s-1").value
R_H  = c / H0_s
V_H  = 4/3*np.pi * R_H**3
S0   = 2.257e122   # horizon entropy

# ----------------------------
# Frequency grid
# ----------------------------
nu = np.logspace(11, 18, 4000)  # Hz

# ----------------------------
# Numerically stable Planck function
# ----------------------------
def B_nu_stable(T, nu):
    """
    Stable Planck function using expm1 to avoid overflow.
    B_nu = (2h nu^3 / c^2) * 1/(exp(h nu/kT)-1)
    """
    x = h * nu / (k_B * T)
    # safe: expm1(x) gives exp(x)-1, stable for small x
    denom = np.expm1(x)
    B = (2*h*nu**3 / c**2) / denom
    # Clean values
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    return B

# ----------------------------
# AGN template
# ----------------------------
def agn_template(nu):
    nu0 = 1e15
    L = np.where(nu < nu0, (nu/nu0)**(-0.5), (nu/nu0)**(-1.2))
    # small high-frequency tail
    L += 0.01*(nu/1e17)**(-0.9)
    L = np.nan_to_num(L)
    L /= np.max(L) if np.max(L)>0 else 1.0
    return L

# ----------------------------
# Greybody (correct dust)
# ----------------------------
def greybody(nu, T_d, beta=1.7):
    g = (nu**beta) * B_nu_stable(T_d, nu)
    g = np.nan_to_num(g)
    g /= np.max(g) if np.max(g)>0 else 1.0
    return g

# ----------------------------
# Utility: build sigma = ∫ L_nu/(k_B T_nu) dnu
# ----------------------------
def s_from_shape(Lshape, L_total):
    import numpy as _np
    integral = _np.trapezoid(Lshape, nu)
    if integral <= 0:
        return 0.0
    L_nu = Lshape * (L_total / integral)
    T_nu = (h*nu)/(3*k_B)
    s_nu = L_nu / (k_B * T_nu)
    sigma = _np.trapezoid(s_nu, nu)
    return sigma

# ===============================================================
# Build stellar template (FSPS or fallback)
# ===============================================================
if use_fsps:
    sp = fsps.StellarPopulation(zcontinuous=1)
    ages = [0.01,0.03,0.1,0.3,1.0,3.0,10.0]

    star_shape = np.zeros_like(nu)
    for age in ages:
        w, spec = sp.get_spectrum(tage=age, peraa=True)
        w_m = np.array(w)*1e-10
        nu_spec = c / w_m
        L_lambda = np.array(spec)
        L_nu_spec = L_lambda * (w_m**2) / c
        star_shape += np.interp(nu, nu_spec[::-1], L_nu_spec[::-1], left=0, right=0)

    star_shape = np.nan_to_num(star_shape)
    if np.max(star_shape)>0:
        star_shape /= np.max(star_shape)
else:
    # fallback BB ~ 5800 K
    star_shape = B_nu_stable(5800.0, nu)
    star_shape = np.nan_to_num(star_shape)
    star_shape /= np.max(star_shape)

# -------------------------------------------------
# AGN & Dust (correct dust = 40 K default here)
# -------------------------------------------------
agn_shape   = agn_template(nu)
dust_shape  = greybody(nu, 40.0)      # <-- FIXED (was 1e6 K)
shock_shape = greybody(nu, 1e6)       # optional X-ray shock, kept explicit

# -------------------------------------
# Baseline luminosity densities (z=0)
# -------------------------------------
L_V_SF_z0    = 1.350e-32
L_V_BH_z0    = 1.929e-34
L_V_shock_z0 = 1.197e-34

# -------------------------------------
# Compute baseline spectral sigma and mu
# -------------------------------------
sigma_sf0   = s_from_shape(star_shape, L_V_SF_z0)
sigma_bh0   = s_from_shape(agn_shape,  L_V_BH_z0)
sigma_dust0 = s_from_shape(dust_shape, L_V_shock_z0)

sigma_tot0 = sigma_sf0 + sigma_bh0 + sigma_dust0
mu0        = sigma_tot0 * V_H
gamma0     = mu0 / S0

print("Baseline Totals:")
print("sigma_tot0 =", sigma_tot0)
print("mu0 =", mu0)
print("gamma0 =", gamma0)

# =====================================================
# Monte-Carlo
# =====================================================
N = 10000
rng = np.random.default_rng(20251103)

Tstar = rng.normal(5800, 800, N)  # fallback range
Tstar = np.clip(Tstar, 2000, None)

f_agn  = np.exp(rng.normal(np.log(0.01), 0.7, N))
f_dust = rng.uniform(0.3, 0.9, N)   # dust fraction more realistic

L_V_SF    = rng.normal(L_V_SF_z0, 0.2*L_V_SF_z0, N)
L_V_BH    = rng.normal(L_V_BH_z0, 0.3*L_V_BH_z0, N)
L_V_dust  = rng.normal(L_V_shock_z0, 0.5*L_V_shock_z0, N)

mu_samps    = np.zeros(N)
sigma_samps = np.zeros(N)
gamma_samps = np.zeros(N)

for i in range(N):
    Ts = Tstar[i]
    stellar_i = B_nu_stable(Ts, nu)
    stellar_i /= np.max(stellar_i) if np.max(stellar_i)>0 else 1.0

    s_sf   = s_from_shape(stellar_i, max(L_V_SF[i],0))
    s_bh   = s_from_shape(agn_shape, max(L_V_BH[i],0)*f_agn[i])
    s_dust = s_from_shape(dust_shape, max(L_V_dust[i],0)*f_dust[i])

    sigma_tot = s_sf + s_bh + s_dust
    mu = sigma_tot * V_H

    sigma_samps[i] = sigma_tot
    mu_samps[i]    = mu
    gamma_samps[i] = mu / S0

# =====================================================
# Store metadata
# =====================================================
meta = {
    "timestamp": time.ctime(),
    "N": N,
    "seed": 20251103,
    "H0[km/s/Mpc]": 67.4,
    "Om0": 0.315,
    "note": "Stable Planck, dust fixed, trapezoid integration"
}
with open(os.path.join(OUTDIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

# =====================================================
# Save CSV results
# =====================================================
df = pd.DataFrame({
    'Tstar':Tstar,
    'f_agn':f_agn,
    'f_dust':f_dust,
    'L_V_SF':L_V_SF,
    'L_V_BH':L_V_BH,
    'L_V_dust':L_V_dust,
    'sigma_tot':sigma_samps,
    'mu':mu_samps,
    'gamma':gamma_samps
})

df.to_csv(os.path.join(OUTDIR,'spectral_fsps_mc_fixed.csv'), index=False)

# =====================================================
# Plots
# =====================================================
plt.figure(figsize=(6,4))
plt.hist(mu_samps, bins=70)
plt.xlabel("mu (k_B/s)"); plt.title("mu distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'mu_hist_fixed.png'))
plt.close()

plt.figure(figsize=(6,4))
plt.hist(gamma_samps, bins=70)
plt.xlabel("gamma (s^-1)"); plt.title("gamma distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'gamma_hist_fixed.png'))
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(Tstar, mu_samps, s=3)
plt.xlabel("Tstar (K)"); plt.ylabel("mu")
plt.title("mu vs Tstar")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'mu_vs_Tstar_fixed.png'))
plt.close()

print("DONE. Outputs in:", OUTDIR)

