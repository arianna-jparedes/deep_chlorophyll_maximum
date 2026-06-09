"""
Bifurcation patterns in a constant light environment.
Speed strategy (two phased integration per parameter value):
    Phase 1 (0 → T_DISCARD): rtol=1e-3, atol=1e-6  — fast transient burn-in.
    Phase 2 (T_DISCARD → T_TOTAL): rtol=1e-5, atol=1e-9  — accurate attractor sampling.
  Parallelisation via multiprocessing.Pool across all available cores.
"""

import sys, os, traceback, time, warnings

# Prevent BLAS/LAPACK from spawning threads
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from multiprocessing import Pool, cpu_count

from dcm_model import DEFAULT_PARAMS, make_grid, run_simulation

# Directory paths
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PANELS_DIR = os.path.join(SCRIPTS_DIR, '..', 'graphs')
os.makedirs(PANELS_DIR, exist_ok=True)

# Parameters
NZ = 250
Z = make_grid(zB=DEFAULT_PARAMS['zB'], nz=NZ)
T_DISCARD = 2000
T_TOTAL = 15000
RTOL_FAST, ATOL_FAST = 1e-3, 1e-6
RTOL_FINE, ATOL_FINE = 1e-5, 1e-9
COLOR_SCALE = 1e9

# Two-phase worker
def _run_one(args):
    """
    Integrate one parameter value using two-phase strategy.
    args = (param_name, param_value, extra_params, job_idx, total_jobs, label)
    Returns (param_value, Pint_array, t_array) for the attractor period only.
    """
    param_name, param_value, extra_params, job_idx, total_jobs, label = args

    params = DEFAULT_PARAMS.copy()
    params.update(extra_params)
    params[param_name] = param_value

    t_wall = time.time()
    nz = len(Z)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Phase 1: loose tolerances
        t1 = time.time()
        res1 = run_simulation(Z, params,
                              t_span_days=(0, T_DISCARD),
                              n_out=51,
                              rtol=RTOL_FAST, atol=ATOL_FAST)
        dt1 = time.time() - t1

        # Hand end-state of phase 1 to phase 2 as initial condition
        y0 = np.concatenate([res1['P'][:, -1], res1['N'][:, -1]])

        # Phase 2: tight tolerances
        t2 = time.time()
        res2 = run_simulation(Z, params,
                              t_span_days=(T_DISCARD, T_TOTAL),
                              n_out=(T_TOTAL - T_DISCARD) + 1,
                              rtol=RTOL_FINE, atol=ATOL_FINE,
                              y0=y0)
        dt2 = time.time() - t2

    Pint = np.trapezoid(res2['P'], Z, axis=0)
    total = time.time() - t_wall

    return param_value, Pint, res2['t']

# Parallel sweep
def _parallel_sweep(param_name, param_values, extra_params, label, n_workers):
    args = [(param_name, v, extra_params, i, len(param_values), label)
            for i, v in enumerate(param_values)]
    with Pool(processes=n_workers) as pool:
        raw = pool.map(_run_one, args)
    raw.sort(key=lambda x: x[0])
    return raw


def _continuation_sweep(param_name, param_values, extra_params, label):
    """
    Sequential continuation sweep: sweep from HIGH to LOW parameter value,
    using the final state of each simulation as the initial condition for
    the next one.
    """
    # Sweep from HIGH to LOW (stable → chaotic direction)
    param_values_desc = param_values[::-1]
    nz = len(Z)
    y0 = None
    raw = []

    for i, pval in enumerate(param_values_desc):
        params = DEFAULT_PARAMS.copy()
        params.update(extra_params)
        params[param_name] = pval

        t_wall = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Phase 1
            t1 = time.time()
            res1 = run_simulation(Z, params,
                                  t_span_days=(0, T_DISCARD),
                                  n_out=51,
                                  rtol=RTOL_FAST, atol=ATOL_FAST,
                                  y0=y0)
            dt1 = time.time() - t1
            y0_phase2 = np.concatenate([res1['P'][:, -1], res1['N'][:, -1]])

            # Phase 2
            t2 = time.time()
            res2 = run_simulation(Z, params,
                                  t_span_days=(T_DISCARD, T_TOTAL),
                                  n_out=(T_TOTAL - T_DISCARD) + 1,
                                  rtol=RTOL_FINE, atol=ATOL_FINE,
                                  y0=y0_phase2)
            dt2 = time.time() - t2

        Pint = np.trapezoid(res2['P'], Z, axis=0)
        total = time.time() - t_wall
        raw.append((pval, Pint, res2['t']))

        # Pass end-state of phase 2 as IC for next kappa value
        y0 = np.concatenate([res2['P'][:, -1], res2['N'][:, -1]])

    # Return sorted low→high to match plotting convention
    raw.sort(key=lambda x: x[0])
    return raw

# Two-way sweep
def _continuation_sweep_both_directions(param_name, param_values, extra_params, label):
    """
    Run continuation sweep in both directions and combine results.
    High→Low captures the chaotic/oscillating region correctly.
    Low→High captures the stable region and transition at high kappa correctly.
    """
    raw_hl = _continuation_sweep(param_name, param_values, extra_params, label + '_hl')  
    raw_lh = _continuation_sweep(param_name, param_values[::-1], extra_params, label + '_lh')

    # Merging
    hl_dict = {pval: (Pint, t) for pval, Pint, t in raw_hl}
    lh_dict = {pval: (Pint, t) for pval, Pint, t in raw_lh}

    raw_combined = []
    for pval in sorted(hl_dict.keys()):
        Pint_hl, t_hl = hl_dict[pval]
        Pint_lh, t_lh = lh_dict[pval]
        
        # Concatenating
        Pint_combined = np.concatenate([Pint_hl, Pint_lh])
        t_combined    = np.concatenate([t_hl,    t_lh])
        raw_combined.append((pval, Pint_combined, t_combined))

    return raw_combined

# Signal analysis
def _local_extrema(signal):
    order = 100
    idx_max = argrelextrema(signal, np.greater, order=order)[0]
    idx_min = argrelextrema(signal, np.less,    order=order)[0]
    return signal[idx_max], signal[idx_min]

def _mean_period(signal, t):
    order = 100
    idx = argrelextrema(signal, np.greater, order=order)[0]
    if len(idx) < 2:
        return np.nan
    amp = _rel_amplitude(signal)
    if amp < 0.05:   # stable regime
        return np.nan
    return np.median(np.diff(t[idx]))

def _rel_amplitude(signal):
    maxs, mins = _local_extrema(signal)
    if len(maxs) == 0 or len(mins) == 0:
        return 0.0
    return (np.mean(maxs) - np.mean(mins)) / np.mean(maxs)

def _extrema_from_sweep(raw):
    bif = []
    for pval, Pint, t in raw:
        maxs, mins = _local_extrema(Pint)
        if len(maxs) == 0: maxs = np.array([Pint[-1]])
        if len(mins) == 0: mins = np.array([Pint[-1]])
        bif.append((pval, maxs, mins))
    return bif

def _period_amp_from_sweep(raw):
    periods, amps = [], []
    for _, Pint, t in raw:
        periods.append(_mean_period(Pint, t))
        amps.append(_rel_amplitude(Pint))
    return np.array(periods), np.array(amps)


# Panel savers
def save_panel_a(bif_data, kappas, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Bifurcation diagram',
                 fontsize=12, fontweight='bold')
    for kappa, maxs, mins in bif_data:
        ax.plot([kappa]*len(maxs), maxs/COLOR_SCALE, '.', color='steelblue', ms=0.5, alpha=0.8)
        ax.plot([kappa]*len(mins), mins/COLOR_SCALE, '.', color='steelblue', ms=0.5, alpha=0.8)
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=11)
    ax.set_ylabel('Phytoplankton biomass\n(×10⁹ cells m⁻²)', fontsize=11)
    ax.set_xlim(kappas[0], kappas[-1])
    ax.text(0.02, 0.97, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_panel_b(bif_data, kappas, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Chaotic window',
                 fontsize=12, fontweight='bold')
    for kappa, maxs, mins in bif_data:
        ax.plot([kappa]*len(maxs), maxs/COLOR_SCALE, '.', color='steelblue', ms=0.5, alpha=0.9)
        ax.plot([kappa]*len(mins), mins/COLOR_SCALE, '.', color='steelblue', ms=0.5, alpha=0.9)
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=11)
    ax.set_ylabel('Phytoplankton biomass\n(×10⁹ cells m⁻²)', fontsize=11)
    ax.set_xlim(kappas[0], kappas[-1])
    ax.text(0.02, 0.97, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_panel_c(v_values, periods, amplitudes, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Period and amplitude vs sinking velocity', fontsize=12, fontweight='bold')
    ax2 = ax.twinx()
    valid = ~np.isnan(periods)
    ax.plot(v_values[valid], periods[valid], 'b-', lw=2, label='Period (days)')
    ax2.plot(v_values, amplitudes, 'r-', lw=2, label='Rel. amplitude')
    ax.set_xlabel('Sinking velocity v (m h⁻¹)', fontsize=11)
    ax.set_ylabel('Mean period of oscillation (days)', color='b', fontsize=11)
    ax2.set_ylabel('Relative amplitude', color='r', fontsize=11)
    ax.set_ylim(0, 250)
    ax2.set_ylim(0, 1)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.text(0.02, 0.97, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_panel_d(kappas, periods, amplitudes, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Period and amplitude vs turbulent diffusivity', fontsize=12, fontweight='bold')
    ax2 = ax.twinx()
    valid = ~np.isnan(periods)
    ax.plot(kappas[valid], periods[valid], 'b-', lw=2, label='Period (days)')
    ax2.plot(kappas, amplitudes, 'r-', lw=2, label='Rel. amplitude')
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=11)
    ax.set_ylabel('Mean period of oscillation (days)', color='b', fontsize=11)
    ax2.set_ylabel('Relative amplitude', color='r', fontsize=11)
    ax.set_ylim(0, 250)
    ax2.set_ylim(0, 1)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.text(0.02, 0.97, 'd', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

# Main
def main():
    n_workers = max(1, cpu_count())
    
    # Panel a: full κ bifurcation diagram
    t0 = time.time()
    kappas_a = np.linspace(0.04, 0.24, 300)
    try:
        raw_a = _continuation_sweep_both_directions('kappa', kappas_a, {}, 'a')
        bif_a = _extrema_from_sweep(raw_a)
        save_panel_a(bif_a, kappas_a, os.path.join(PANELS_DIR, 'fig3_panel_a.png'))
    except Exception:
        print('Panel a FAILED:'); traceback.print_exc()

    # Panel b: period-doubling / chaotic window detail
    t0 = time.time()
    kappas_b = np.linspace(0.105, 0.130, 300)
    try:
        raw_b = _continuation_sweep('kappa', kappas_b, {}, 'b')
        bif_b = _extrema_from_sweep(raw_b)
        save_panel_b(bif_b, kappas_b, os.path.join(PANELS_DIR, 'fig3_panel_b.png'))
    except Exception:
        print('Panel b FAILED:'); traceback.print_exc()

    # Panel c: period and amplitude vs sinking velocity
    t0 = time.time()
    v_values = np.linspace(0.025, 0.045, 50)
    try:
        raw_c = _parallel_sweep('v', v_values, {'kappa': 0.12}, 'c', n_workers)
        per_v, amp_v = _period_amp_from_sweep(raw_c)
        save_panel_c(v_values, per_v, amp_v,
                     os.path.join(PANELS_DIR, 'fig3_panel_c.png'))
    except Exception:
        print('Panel c FAILED:'); traceback.print_exc()

    # Panel d: period and amplitude vs turbulent diffusivity
    t0 = time.time()
    kappas_d = np.linspace(0.11, 0.23, 50)
    try:
        raw_d = _parallel_sweep('kappa', kappas_d, {}, 'd', n_workers)
        per_k, amp_k = _period_amp_from_sweep(raw_d)
        save_panel_d(kappas_d, per_k, amp_k,
                     os.path.join(PANELS_DIR, 'fig3_panel_d.png'))
    except Exception:
        print('Panel d FAILED:'); traceback.print_exc()

if __name__ == '__main__':
    main()
