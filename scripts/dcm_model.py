"""
dcm_model.py
============
Core numerical model for the Deep Chlorophyll Maximum (DCM) study,
following Huisman et al. (2006) Nature 439:322-325.

Methodology adapted from the paper's Supplementary Information:
  - Advection : 3rd-order upwind-biased scheme (Hundsdorfer & Verwer 2003)
  - Diffusion : 2nd-order symmetric central differences
  - Light : cumulative trapezoidal integral (vectorised)
  - Solver : BDF (variable-order, variable-step implicit multistep),
             scipy's equivalent of the VODE solver used in the paper.
             Analytical sparse Jacobian supplied to prevent numerical
             Jacobian probing from causing NaN blow-up.

Boundary conditions:
  Phytoplankton : zero-flux at z=0 and z=zB
  Nutrients     : zero-flux at z=0; N = NB (Dirichlet) at z=zB
"""

import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp

CM2_S_TO_M2_H = 0.36   # 1 cm² s⁻¹ = 0.36 m² h⁻¹

# Default parameter values
DEFAULT_PARAMS = dict(
    Iin = 600.0,     # Incident light [µmol photons m⁻² s⁻¹]
    Kbg = 0.045,     # Background turbidity [m⁻¹]
    k = 6e-10,       # Phytoplankton absorption [m² cell⁻¹]
    zB = 300.0,      # Column depth [m]
    kappa = 0.12,    # Turbulent diffusivity [cm² s⁻¹]
    mu_max = 0.04,   # Max growth rate [h⁻¹]
    HI = 20.0,       # Light half-saturation [µmol photons m⁻² s⁻¹]
    HN = 0.025,      # Nutrient half-saturation [mmol m⁻³]
    m = 0.01,        # Loss rate [h⁻¹]
    v = 0.042,       # Sinking velocity [m h⁻¹]
    alpha = 1e-9,    # Nutrient per cell [mmol cell⁻¹]
    epsilon = 0.5,   # Nutrient recycling [–]
    NB = 10.0,       # Bottom nutrient concentration [mmol m⁻³]
)

# Seasonal light  (North Pacific subtropical gyre)
def Iin_seasonal(t_h):
    """Sinusoidal annual cycle: 30→60 mol photons m⁻² d⁻¹ → µmol m⁻² s⁻¹."""
    c = 1e6 / 86400.0
    Iw, Is = 30.0*c, 60.0*c
    period = 365.25 * 24.0
    return 0.5*(Iw+Is) - 0.5*(Is-Iw)*np.cos(2*np.pi*t_h/period)


# Physics helpers
def light_profile(z, P, Iin, Kbg, k):
    """I(z) = Iin·exp(−Kbg·z − k·∫₀ᶻ P dσ)  [vectorised trapezoidal]."""
    dz = z[1] - z[0]
    integral = np.zeros_like(z)
    integral[1:] = np.cumsum(0.5*(P[:-1]+P[1:])*dz)
    return Iin * np.exp(-Kbg*z - k*integral)


def growth_rate(N, I, mu_max, HN, HI):
    """Von Liebig minimum of two Monod functions."""
    return mu_max * np.minimum(N/(HN+N), I/(HI+I))


# 3rd-order upwind advection (Hundsdorfer & Verwer 2003)
def advection_upwind3(P, dz, v):
    """
    3rd-order upwind-biased advection for v > 0 (sinking = increasing z).

    Face flux at i+½:  f_{i+½} = (1/6)(−P_{i−1} + 5P_i + 2P_{i+1})

    Ghost cells (zero-flux BC):
      surface: P_{−2} = P_{−1} = P_0   (no phytoplankton entering from above)
      bottom:  P_n = P_{n+1} = P_{n−1}  (zero-flux)

    The zero-flux condition at the surface is enforced by setting
    f_{−½} = 0  (no flux into the top cell from above).
    """
    n = len(P)
    Pg = np.empty(n + 4)
    Pg[0] = P[0]
    Pg[1] = P[0]
    Pg[2:-2] = P
    Pg[-2] = P[-1]
    Pg[-1] = P[-1]

    i = np.arange(n)
    f_right = (1.0/6.0)*(-Pg[i+1] + 5.0*Pg[i+2] + 2.0*Pg[i+3])
    f_left  = (1.0/6.0)*(-Pg[i+0] + 5.0*Pg[i+1] + 2.0*Pg[i+2])
    f_left[0] = 0.0   # zero-flux BC at surface

    return v * (f_right - f_left) / dz


# 2nd-order central diffusion
def diffusion_op(U, dz, kappa, bc_bot=None):
    """κ·∂²U/∂z², 2nd-order central.  bc_bot: Dirichlet value at bottom."""
    d2 = np.empty_like(U)
    d2[1:-1] = (U[2:] - 2.0*U[1:-1] + U[:-2]) / dz**2
    d2[0]    = (U[1]  - U[0])  / dz**2           # zero-flux surface

    if bc_bot is not None:
        d2[-1] = (bc_bot - 2.0*U[-1] + U[-2]) / dz**2
    else:
        d2[-1] = (U[-2] - U[-1]) / dz**2          # zero-flux bottom
    return kappa * d2


# Analytical sparse Jacobian
def _adv3_matrix(nz, dz, v):
    """
    Linearised advection operator as sparse matrix.
    Row i: d(adv_i)/dP_j  from the 3rd-order upwind stencil.
    adv_i = v*(f_right_i - f_left_i)/dz
    f_right_i = (1/6)(-P_{i-1}+5P_i+2P_{i+1})
    f_left_i  = (1/6)(-P_{i-2}+5P_{i-1}+2P_i)  [with ghost cells]
    d(adv_i)/dP_i   = v/dz * (5/6 - 2/6)  = v/dz * 3/6 = v/(2dz)
    d(adv_i)/dP_{i-1} = v/dz * (2/6·(-1) ... full stencil below
    """
    c = v / dz
    # For the Jacobian we just use the standard stencil for interior rows.
    a_m2 = c/6.0
    a_m1 = -c
    a_0  = c/2.0
    a_p1 = c/3.0

    # Build as sum of diagonal bands
    diag_0  = np.full(nz, a_0)
    diag_m1 = np.full(nz-1, a_m1)
    diag_m2 = np.full(nz-2, a_m2)
    diag_p1 = np.full(nz-1, a_p1)
    # Row 0 (surface): 
    diag_0[0]  = v/(dz) * 4.0/6.0
    diag_p1[0] = v/(dz) * 2.0/6.0
    # Row 1 (standard interior stencil for rows 1..n-1)
    return sp.diags([diag_m2, diag_m1, diag_0, diag_p1],
                    [-2, -1, 0, 1], shape=(nz,nz), format='csr')


def _diff_matrix(nz, dz, kappa, bc_bot=False):
    """κ·∂²U/∂z² as tridiagonal sparse matrix."""
    c = kappa / dz**2
    main = np.full(nz, -2.0*c)
    off = np.full(nz-1, c)
    main[0]  = -c   # zero-flux surface
    main[-1] = -c if not bc_bot else -2.0*c  # Neumann or Dirichlet bottom
    return sp.diags([off, main, off], [-1, 0, 1], shape=(nz,nz), format='csr')


def _build_jacobian(z, params, P, N, Iin):
    """Analytical 2nz×2nz sparse Jacobian of the single-species RHS."""
    nz = len(z); dz = z[1]-z[0]
    kappa = params['kappa'] * CM2_S_TO_M2_H
    v = params['v']; mu_max=params['mu_max']
    HI = params['HI']; HN=params['HN']
    alpha = params['alpha']; eps=params['epsilon']; m=params['m']

    I = light_profile(z, P, Iin, params['Kbg'], params['k'])
    mu = growth_rate(N, I, mu_max, HN, HI)

    fN = N/(HN+N); fI = I/(HI+I)
    light_lim = fI < fN
    dmu_dI = np.where(light_lim, mu_max*HI/(HI+I)**2, 0.0)
    dmu_dN = np.where(~light_lim, mu_max*HN/(HN+N)**2, 0.0)
    dI_dPi = -params['k']*I*dz*0.5   # diagonal approximation of nonlocal term
    dmu_dPloc = dmu_dI * dI_dPi

    A = _adv3_matrix(nz, dz, v)
    DP = _diff_matrix(nz, dz, kappa, bc_bot=False)
    DN = _diff_matrix(nz, dz, kappa, bc_bot=True)

    dFP_dP = sp.diags(mu - m + dmu_dPloc*P, 0) - A + DP
    dFP_dN = sp.diags(P * dmu_dN, 0)
    dFN_dP = sp.diags((-alpha*mu + eps*alpha*m) + P*(-alpha*dmu_dPloc), 0)
    dFN_dN = DN + sp.diags(-alpha*P*dmu_dN, 0)

    return sp.bmat([[dFP_dP, dFP_dN],[dFN_dP, dFN_dN]], format='csr')


# RHS factories
def make_rhs_single(z, params, Iin_func=None):
    nz = len(z); dz = z[1]-z[0]
    kappa = params['kappa'] * CM2_S_TO_M2_H
    Iin0 = params.get('Iin', 600.0)
    m = params['m']; v=params['v']
    mu_max = params['mu_max']; HI=params['HI']; HN=params['HN']
    alpha = params['alpha']; epsilon=params['epsilon']; NB=params['NB']
    Kbg = params['Kbg']; k=params['k']

    def rhs(t, y):
        P = np.maximum(y[:nz], 0.0)
        N = np.maximum(y[nz:], 0.0)
        Iin = Iin_func(t) if Iin_func is not None else Iin0
        I = light_profile(z, P, Iin, Kbg, k)
        mu = growth_rate(N, I, mu_max, HN, HI)
        dPdt = (mu-m)*P - advection_upwind3(P,dz,v) + diffusion_op(P,dz,kappa)
        dNdt = (-alpha*mu+epsilon*alpha*m)*P + diffusion_op(N,dz,kappa,bc_bot=NB)
        out = np.concatenate([dPdt, dNdt])
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def jac(t, y):
        P = np.maximum(y[:nz], 0.0)
        N = np.maximum(y[nz:], 0.0)
        Iin = Iin_func(t) if Iin_func is not None else Iin0
        return _build_jacobian(z, params, P, N, Iin)

    return rhs, jac


def make_rhs_multi(z, species_params, shared_params, Iin_func=None):
    nz = len(z); dz = z[1]-z[0]
    ns = len(species_params)
    Kbg = shared_params['Kbg']; NB=shared_params['NB']
    kappa = shared_params['kappa'] * CM2_S_TO_M2_H
    Iin0 = shared_params.get('Iin', 600.0)

    def rhs(t, y):
        Ps = [np.maximum(y[i*nz:(i+1)*nz], 0.0) for i in range(ns)]
        N = np.maximum(y[ns*nz:], 0.0)
        Iin = Iin_func(t) if Iin_func is not None else Iin0
        Ptk = sum(sp_['k']*Ps[i] for i,sp_ in enumerate(species_params))
        integral = np.zeros(nz)
        integral[1:] = np.cumsum(0.5*(Ptk[:-1]+Ptk[1:])*dz)
        I = Iin * np.exp(-Kbg*z - integral)
        derivs = []; dNdt = diffusion_op(N, dz, kappa, bc_bot=NB)
        for i, sp_ in enumerate(species_params):
            mu_i = growth_rate(N, I, sp_['mu_max'], sp_['HN'], sp_['HI'])
            dPi = ((mu_i-sp_['m'])*Ps[i]
                     - advection_upwind3(Ps[i],dz,sp_['v'])
                     + diffusion_op(Ps[i],dz,kappa))
            derivs.append(dPi)
            dNdt += (-sp_['alpha']*mu_i + sp_['epsilon']*sp_['alpha']*sp_['m'])*Ps[i]
        out = np.concatenate(derivs+[dNdt])
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    return rhs


# Runners
def run_simulation(z, params, t_span_days,
                   Iin_func=None, rtol=1e-6, atol=1e-10,
                   n_out=500, y0=None):
    """
    Integrate the single-species model using BDF (similar to VODE).
    Tighter default tolerances (rtol=1e-6, atol=1e-10) to resolve the
    oscillations accurately.
    """
    nz = len(z)
    t_span = (t_span_days[0]*24.0, t_span_days[1]*24.0)
    t_eval = np.linspace(t_span[0], t_span[1], n_out)
    if y0 is None:
        y0 = np.concatenate([np.ones(nz)*1e6,
                             np.linspace(0.1, params['NB'], nz)])
    rhs, jac = make_rhs_single(z, params, Iin_func=Iin_func)
    sol = solve_ivp(rhs, t_span, y0, method='BDF',
                    jac=jac,
                    t_eval=t_eval, rtol=rtol, atol=atol,
                    dense_output=False)
    P = np.maximum(sol.y[:nz, :], 0.0)
    N = np.maximum(sol.y[nz:, :], 0.0)
    return dict(t=sol.t/24.0, z=z, P=P, N=N, sol=sol)


def run_simulation_multi(z, species_params, shared_params, t_span_days,
                         Iin_func=None, rtol=1e-6, atol=1e-10,
                         n_out=500, y0=None):
    nz = len(z); ns=len(species_params)
    t_span = (t_span_days[0]*24.0, t_span_days[1]*24.0)
    t_eval = np.linspace(t_span[0], t_span[1], n_out)
    if y0 is None:
        y0 = np.concatenate([np.ones(nz)*1e4 for _ in range(ns)]
                            + [np.linspace(0.1, shared_params['NB'], nz)])
    rhs = make_rhs_multi(z, species_params, shared_params, Iin_func=Iin_func)
    sol = solve_ivp(rhs, t_span, y0, method='BDF',
                    t_eval=t_eval, rtol=rtol, atol=atol,
                    dense_output=False)
    Ps = [np.maximum(sol.y[i*nz:(i+1)*nz,:],0.0) for i in range(ns)]
    N  = np.maximum(sol.y[ns*nz:,:],0.0)
    return dict(t=sol.t/24.0, z=z, Ps=Ps, N=N, sol=sol)


def make_grid(zB=300.0, nz=100):
    return np.linspace(0.0, zB, nz)
