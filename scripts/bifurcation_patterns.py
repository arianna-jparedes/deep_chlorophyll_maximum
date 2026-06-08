"""
fig3_bifurcation.py
===================
Reproduces Figure 3 of Huisman et al. (2006):
  Bifurcation patterns in a constant light environment.

Each panel is saved as an individual PNG in graphs/fig3_panels/:
  fig3_panel_a.png — bifurcation diagram: local min/max of ∫P dz vs κ
  fig3_panel_b.png — zoom on the chaotic window
  fig3_panel_c.png — period and relative amplitude vs sinking velocity v
  fig3_panel_d.png — period and relative amplitude vs turbulent diffusivity κ

Changes vs original script that are consistent with dcm_model.py:
  - NZ=100 (dz=3m): NZ=60 was too coarse — numerical diffusion 2.4× physical
    kappa, shifting bifurcation points. NZ=100 reduces this to ~1.5×, which
    is a reasonable compromise between accuracy and speed for a parameter sweep.
  - rtol=1e-5, atol=1e-9: slightly tighter than the old 1e-4/1e-8 to avoid
    solver error contaminating the bifurcation diagram, but looser than the
    fig2 values (1e-6/1e-10) since we only need statistical extrema.
  - np.trapz → np.trapezoid (np.trapz is removed in NumPy 2.x).
  - T_DISCARD=2000d kept: adequate to eliminate both the initial surface bloom
    transient (~500d) and early oscillation transient.
  - kappa range for panel a extended to 0.30 to match paper Fig 3a x-axis.
  - One file per panel, saved immediately so a failure doesn't lose earlier work.
"""

import sys, os, traceback, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from dcm_model import DEFAULT_PARAMS, make_grid, run_simulation

# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR  = os.path.join(SCRIPTS_DIR, '..', 'graphs')
PANELS_DIR  = os.path.join(GRAPHS_DIR, 'fig3_panels')
os.makedirs(PANELS_DIR, exist_ok=True)

# Grid: NZ=100 balances speed and numerical diffusion for parameter sweeps
NZ = 100
Z  = make_grid(zB=DEFAULT_PARAMS['zB'], nz=NZ)

# Time settings: run long enough for chaos/oscillations to fully develop
# T_DISCARD eliminates both the surface-bloom transient (~500d) and
# any early oscillation transient, leaving only the attractor dynamics.
T_TOTAL   = 6000   # days — total simulation length
T_DISCARD = 2000   # days — discard this initial portion

# Solver tolerances: slightly relaxed vs fig2 for speed across many runs
RTOL = 1e-5
ATOL = 1e-9


# ---------------------------------------------------------------------------
# Signal analysis helpers
# ---------------------------------------------------------------------------
def integrate_P(P, z):
    """Column-integrated phytoplankton [cells m⁻²]."""
    return np.trapezoid(P, z, axis=0)


def local_extrema(signal):
    """Local maxima and minima of signal after transient."""
    order = max(5, len(signal) // 100)
    idx_max = argrelextrema(signal, np.greater, order=order)[0]
    idx_min = argrelextrema(signal, np.less,    order=order)[0]
    return signal[idx_max], signal[idx_min]


def mean_period(signal, t):
    """Mean oscillation period estimated from local maxima spacing."""
    order = max(5, len(signal) // 100)
    idx_max = argrelextrema(signal, np.greater, order=order)[0]
    if len(idx_max) < 2:
        return np.nan
    return np.mean(np.diff(t[idx_max]))


def rel_amplitude(signal):
    """Relative amplitude: (max − min) / mean."""
    mu = np.mean(signal)
    return (np.max(signal) - np.min(signal)) / mu if mu > 0 else 0.0


# ---------------------------------------------------------------------------
# Simulation runners for each panel type
# ---------------------------------------------------------------------------
def run_bifurcation_kappa(kappa_values, params_base, label=''):
    """
    Sweep κ values, discard transient, collect local extrema of ∫P dz.
    Returns list of (kappa, maxima_array, minima_array).
    """
    # n_out: 1 point per day over the kept portion is sufficient
    n_out = T_TOTAL - T_DISCARD + 1
    results = []
    for i, kappa in enumerate(kappa_values):
        params = params_base.copy()
        params['kappa'] = kappa
        print(f"  {label} κ={kappa:.4f} cm²/s  ({i+1}/{len(kappa_values)})", end='\r')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = run_simulation(Z, params, t_span_days=(0, T_TOTAL),
                                 n_out=T_TOTAL + 1,   # 1 pt/day
                                 rtol=RTOL, atol=ATOL)
        keep  = res['t'] >= T_DISCARD
        Pint  = integrate_P(res['P'][:, keep], Z)
        maxs, mins = local_extrema(Pint)
        if len(maxs) == 0: maxs = np.array([Pint[-1]])
        if len(mins) == 0: mins = np.array([Pint[-1]])
        results.append((kappa, maxs, mins))
    print()
    return results


def run_vs_sinking(v_values, kappa_fixed, params_base):
    """Sweep sinking velocities; return (periods, amplitudes)."""
    periods, amplitudes = [], []
    for i, v in enumerate(v_values):
        params = params_base.copy()
        params['v']     = v
        params['kappa'] = kappa_fixed
        print(f"  v={v:.4f} m/h  ({i+1}/{len(v_values)})", end='\r')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = run_simulation(Z, params, t_span_days=(0, T_TOTAL),
                                 n_out=T_TOTAL + 1,
                                 rtol=RTOL, atol=ATOL)
        keep = res['t'] >= T_DISCARD
        Pint = integrate_P(res['P'][:, keep], Z)
        periods.append(mean_period(Pint, res['t'][keep]))
        amplitudes.append(rel_amplitude(Pint))
    print()
    return np.array(periods), np.array(amplitudes)


def run_vs_kappa_pd(kappa_values, params_base):
    """Sweep κ values; return (periods, amplitudes)."""
    periods, amplitudes = [], []
    for i, kappa in enumerate(kappa_values):
        params = params_base.copy()
        params['kappa'] = kappa
        print(f"  κ={kappa:.4f} cm²/s  ({i+1}/{len(kappa_values)})", end='\r')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = run_simulation(Z, params, t_span_days=(0, T_TOTAL),
                                 n_out=T_TOTAL + 1,
                                 rtol=RTOL, atol=ATOL)
        keep = res['t'] >= T_DISCARD
        Pint = integrate_P(res['P'][:, keep], Z)
        periods.append(mean_period(Pint, res['t'][keep]))
        amplitudes.append(rel_amplitude(Pint))
    print()
    return np.array(periods), np.array(amplitudes)


# ---------------------------------------------------------------------------
# Individual panel savers
# ---------------------------------------------------------------------------
COLOR_SCALE = 1e9   # express biomass in ×10⁹ cells m⁻²


def save_panel_a(bif_data, kappas, out_path):
    """Bifurcation diagram: ∫P dz local extrema vs κ."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Figure 3a — Bifurcation diagram\nHuisman et al. (2006)',
                 fontsize=12, fontweight='bold')
    for kappa, maxs, mins in bif_data:
        ax.plot([kappa]*len(maxs), maxs/COLOR_SCALE, '.', color='steelblue', ms=2, alpha=0.8)
        ax.plot([kappa]*len(mins), mins/COLOR_SCALE, '.', color='steelblue', ms=2, alpha=0.8)
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=11)
    ax.set_ylabel('Phytoplankton biomass\n(×10⁹ cells m⁻²)', fontsize=11)
    ax.set_xlim(kappas[0], kappas[-1])
    ax.text(0.02, 0.97, 'a', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {out_path}')


def save_panel_b(bif_data, kappas, out_path):
    """Zoomed bifurcation diagram on the chaotic window."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Figure 3b — Chaotic window detail\nHuisman et al. (2006)',
                 fontsize=12, fontweight='bold')
    for kappa, maxs, mins in bif_data:
        ax.plot([kappa]*len(maxs), maxs/COLOR_SCALE, '.', color='steelblue', ms=3, alpha=0.9)
        ax.plot([kappa]*len(mins), mins/COLOR_SCALE, '.', color='steelblue', ms=3, alpha=0.9)
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=11)
    ax.set_ylabel('Phytoplankton biomass\n(×10⁹ cells m⁻²)', fontsize=11)
    ax.set_xlim(kappas[0], kappas[-1])
    ax.text(0.02, 0.97, 'b', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {out_path}')


def save_panel_c(v_values, periods, amplitudes, out_path):
    """Period and relative amplitude vs sinking velocity."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Figure 3c — Period and amplitude vs sinking velocity\n'
                 'Huisman et al. (2006)', fontsize=12, fontweight='bold')
    ax2 = ax.twinx()
    valid = ~np.isnan(periods)
    ax.plot(v_values[valid],  periods[valid],    'b-o', ms=5, label='Period (days)')
    ax2.plot(v_values,        amplitudes,         'r-s', ms=5, label='Rel. amplitude')
    ax.set_xlabel('Sinking velocity v (m h⁻¹)', fontsize=11)
    ax.set_ylabel('Mean period of oscillation (days)', color='b', fontsize=11)
    ax2.set_ylabel('Relative amplitude', color='r', fontsize=11)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.text(0.02, 0.97, 'c', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {out_path}')


def save_panel_d(kappas, periods, amplitudes, out_path):
    """Period and relative amplitude vs turbulent diffusivity."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Figure 3d — Period and amplitude vs turbulent diffusivity\n'
                 'Huisman et al. (2006)', fontsize=12, fontweight='bold')
    ax2 = ax.twinx()
    valid = ~np.isnan(periods)
    ax.plot(kappas[valid],  periods[valid],   'b-o', ms=5, label='Period (days)')
    ax2.plot(kappas,        amplitudes,        'r-s', ms=5, label='Rel. amplitude')
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=11)
    ax.set_ylabel('Mean period of oscillation (days)', color='b', fontsize=11)
    ax2.set_ylabel('Relative amplitude', color='r', fontsize=11)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.text(0.02, 0.97, 'd', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    params_base = DEFAULT_PARAMS.copy()

    # ------------------------------------------------------------------
    # Panel a: full κ range bifurcation diagram
    # Extended to 0.30 to match paper Fig 3a x-axis
    # ------------------------------------------------------------------
    print("Panel a: bifurcation diagram (full κ range 0.04–0.30)...")
    kappas_a = np.linspace(0.04, 0.30, 40)
    try:
        bif_a = run_bifurcation_kappa(kappas_a, params_base, label='a')
        save_panel_a(bif_a, kappas_a,
                     os.path.join(PANELS_DIR, 'fig3_panel_a.png'))
    except Exception:
        print('  Panel a FAILED:'); traceback.print_exc()

    # ------------------------------------------------------------------
    # Panel b: zoomed chaotic window
    # Range 0.105–0.130 cm²/s where period-doubling cascade occurs
    # ------------------------------------------------------------------
    print("Panel b: chaotic window detail (κ = 0.105–0.130)...")
    kappas_b = np.linspace(0.105, 0.130, 30)
    try:
        bif_b = run_bifurcation_kappa(kappas_b, params_base, label='b')
        save_panel_b(bif_b, kappas_b,
                     os.path.join(PANELS_DIR, 'fig3_panel_b.png'))
    except Exception:
        print('  Panel b FAILED:'); traceback.print_exc()

    # ------------------------------------------------------------------
    # Panel c: period and amplitude vs sinking velocity
    # κ fixed at 0.12 cm²/s (within oscillation regime)
    # v range spans stable→oscillating→chaos transition
    # ------------------------------------------------------------------
    print("Panel c: period/amplitude vs sinking velocity (κ=0.12)...")
    v_values = np.linspace(0.025, 0.055, 25)
    try:
        per_v, amp_v = run_vs_sinking(v_values, kappa_fixed=0.12,
                                       params_base=params_base)
        save_panel_c(v_values, per_v, amp_v,
                     os.path.join(PANELS_DIR, 'fig3_panel_c.png'))
    except Exception:
        print('  Panel c FAILED:'); traceback.print_exc()

    # ------------------------------------------------------------------
    # Panel d: period and amplitude vs turbulent diffusivity
    # Range covers stable→oscillating→stable transition
    # ------------------------------------------------------------------
    print("Panel d: period/amplitude vs turbulent diffusivity...")
    kappas_d = np.linspace(0.10, 0.24, 25)
    try:
        per_k, amp_k = run_vs_kappa_pd(kappas_d, params_base)
        save_panel_d(kappas_d, per_k, amp_k,
                     os.path.join(PANELS_DIR, 'fig3_panel_d.png'))
    except Exception:
        print('  Panel d FAILED:'); traceback.print_exc()

    print(f'\nAll panels → {PANELS_DIR}/')


if __name__ == '__main__':
    main()
