"""
Bifurcation patterns in a constant light environment.
Speed strategy (two phased integration per parameter value):
    Phase 1 (0 → T_DISCARD): rtol=1e-3, atol=1e-6  — fast transient burn-in.
    Phase 2 (T_DISCARD → T_TOTAL): rtol=1e-5, atol=1e-9  — accurate attractor sampling.
  Parallelisation via multiprocessing.Pool across all available cores.
"""

import sys, os, traceback, time, warnings

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from multiprocessing import Pool, cpu_count

from dcm_model import DEFAULT_PARAMS, make_grid, run_simulation

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PANELS_DIR = os.path.join(SCRIPTS_DIR, '..', 'graphs')
os.makedirs(PANELS_DIR, exist_ok=True)

NZ = 250
Z = make_grid(zB=DEFAULT_PARAMS['zB'], nz=NZ)
T_DISCARD = 2000
T_TOTAL = 15000
RTOL_FAST, ATOL_FAST = 1e-3, 1e-6
RTOL_FINE, ATOL_FINE = 1e-5, 1e-9
COLOR_SCALE = 1e9

def _run_one(args):
    param_name, param_value, extra_params, job_idx, total_jobs, label = args
    params = DEFAULT_PARAMS.copy()
    params.update(extra_params)
    params[param_name] = param_value
    t_wall = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res1 = run_simulation(Z, params, t_span_days=(0, T_DISCARD),
                              n_out=51, rtol=RTOL_FAST, atol=ATOL_FAST)
        y0 = np.concatenate([res1['P'][:, -1], res1['N'][:, -1]])
        res2 = run_simulation(Z, params, t_span_days=(T_DISCARD, T_TOTAL),
                              n_out=(T_TOTAL - T_DISCARD) + 1,
                              rtol=RTOL_FINE, atol=ATOL_FINE, y0=y0)

    Pint = np.trapezoid(res2['P'], Z, axis=0)
    return param_value, Pint, res2['t']

def _parallel_sweep(param_name, param_values, extra_params, label, n_workers):
    args = [(param_name, v, extra_params, i, len(param_values), label)
            for i, v in enumerate(param_values)]
    with Pool(processes=n_workers) as pool:
        raw = pool.map(_run_one, args)
    raw.sort(key=lambda x: x[0])
    return raw

def _continuation_sweep(param_name, param_values, extra_params, label):
    param_values_desc = param_values[::-1]
    y0 = None
    raw = []
    for i, pval in enumerate(param_values_desc):
        params = DEFAULT_PARAMS.copy()
        params.update(extra_params)
        params[param_name] = pval
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res1 = run_simulation(Z, params, t_span_days=(0, T_DISCARD),
                                  n_out=51, rtol=RTOL_FAST, atol=ATOL_FAST, y0=y0)
            y0_phase2 = np.concatenate([res1['P'][:, -1], res1['N'][:, -1]])
            res2 = run_simulation(Z, params, t_span_days=(T_DISCARD, T_TOTAL),
                                  n_out=(T_TOTAL - T_DISCARD) + 1,
                                  rtol=RTOL_FINE, atol=ATOL_FINE, y0=y0_phase2)
        Pint = np.trapezoid(res2['P'], Z, axis=0)
        raw.append((pval, Pint, res2['t']))
        y0 = np.concatenate([res2['P'][:, -1], res2['N'][:, -1]])
    raw.sort(key=lambda x: x[0])
    return raw

def _continuation_sweep_both_directions(param_name, param_values, extra_params, label):
    raw_hl = _continuation_sweep(param_name, param_values, extra_params, label + '_hl')
    raw_lh = _continuation_sweep(param_name, param_values[::-1], extra_params, label + '_lh')
    hl_dict = {pval: (Pint, t) for pval, Pint, t in raw_hl}
    lh_dict = {pval: (Pint, t) for pval, Pint, t in raw_lh}
    raw_combined = []
    for pval in sorted(hl_dict.keys()):
        Pint_hl, t_hl = hl_dict[pval]
        Pint_lh, t_lh = lh_dict[pval]
        raw_combined.append((pval, np.concatenate([Pint_hl, Pint_lh]),
                                   np.concatenate([t_hl, t_lh])))
    return raw_combined

def _local_extrema(signal):
    order = 100
    idx_max = argrelextrema(signal, np.greater, order=order)[0]
    idx_min = argrelextrema(signal, np.less,    order=order)[0]
    return signal[idx_max], signal[idx_min]

def _rel_amplitude(signal):
    maxs, mins = _local_extrema(signal)
    if len(maxs) == 0 or len(mins) == 0:
        return 0.0
    return (np.median(maxs) - np.median(mins)) / np.median(maxs)

def _mean_period(signal, t):
    order = 100
    idx = argrelextrema(signal, np.greater, order=order)[0]
    if len(idx) < 2:
        return 0.0
    if _rel_amplitude(signal) < 0.05:
        return 0.0
    return np.median(np.diff(t[idx]))

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

# Panel savers — original figure size (7×5), fonts bumped up
def save_panel_a(bif_data, kappas, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for kappa, maxs, mins in bif_data:
        ax.plot([kappa]*len(maxs), maxs/COLOR_SCALE, '.', color='steelblue', ms=0.5, alpha=0.8)
        ax.plot([kappa]*len(mins), mins/COLOR_SCALE, '.', color='steelblue', ms=0.5, alpha=0.8)
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=16)
    ax.set_ylabel('Phytoplankton biomass\n(×10⁹ cells m⁻²)', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_xlim(kappas[0], kappas[-1])
    ax.text(0.02, 0.97, 'a', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_panel_b(bif_data, kappas, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for kappa, maxs, mins in bif_data:
        ax.plot([kappa]*len(maxs), maxs/COLOR_SCALE, '.', color='steelblue', ms=0.5, alpha=0.9)
        ax.plot([kappa]*len(mins), mins/COLOR_SCALE, '.', color='steelblue', ms=0.5, alpha=0.9)
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=16)
    ax.set_ylabel('Phytoplankton biomass\n(×10⁹ cells m⁻²)', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_xlim(kappas[0], kappas[-1])
    ax.text(0.02, 0.97, 'b', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_panel_c(v_values, periods, amplitudes, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax2 = ax.twinx()
    valid = ~np.isnan(periods)
    ax.plot(v_values[valid], periods[valid], 'b-', lw=2, zorder=0, label='Period (days)')
    ax2.plot(v_values, amplitudes, 'r-', lw=2, zorder=0, label='Rel. amplitude')
    ax.set_xlabel('Sinking velocity v (m h⁻¹)', fontsize=16)
    ax.set_ylabel('Mean period of oscillation (days)', color='b', fontsize=16)
    ax2.set_ylabel('Relative amplitude', color='r', fontsize=16)
    ax.set_ylim(0, 250)
    ax2.set_ylim(0, 1)
    ax.tick_params(axis='y', labelcolor='b', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax2.tick_params(axis='y', labelcolor='r', labelsize=14)
    ax.text(0.02, 0.97, 'c', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
    ax.set_xlim(v_values[0], v_values[-1])
    _osc = np.array(amplitudes) > 0.05
    _v0  = v_values[np.where(_osc)[0][0]]
    _v1  = 0.043
    ax.axvspan(v_values[0], _v0, alpha=0.2, color='green', zorder=0, label='No-oscillation')
    ax.axvspan(_v1, v_values[-1], alpha=0.2, color='teal', zorder=0, label='Chaotic region')
    ax.text((v_values[0]+_v0)/2, 125, 'No-oscillation\nregion', fontsize=12, ha='center', va='center', color='darkgreen', rotation=90, style='italic')
    ax.text((_v1+v_values[-1])/2, 125, 'Chaotic\nregion',       fontsize=12, ha='center', va='center', color='teal',      rotation=90, style='italic')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=14, loc='lower center')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_panel_d(kappas, periods, amplitudes, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax2 = ax.twinx()
    valid = ~np.isnan(periods)
    ax.plot(kappas[valid], periods[valid], 'b-', lw=2, zorder=0, label='Period (days)')
    ax2.plot(kappas, amplitudes, 'r-', lw=2, zorder=0, label='Rel. amplitude')
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=16)
    ax.set_ylabel('Mean period of oscillation (days)', color='b', fontsize=16)
    ax2.set_ylabel('Relative amplitude', color='r', fontsize=16)
    ax.set_ylim(0, 250)
    ax2.set_ylim(0, 1)
    ax.tick_params(axis='y', labelcolor='b', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=14)
    ax.text(0.02, 0.97, 'd', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
    ax.set_xlim(kappas[0], kappas[-1])
    _osc = np.array(amplitudes) > 0.05
    _k0  = 0.11
    _k1  = kappas[np.where(_osc)[0][-1]]
    ax.axvspan(kappas[0], _k0, alpha=0.2, color='teal',  zorder=0, label='Chaotic region')
    ax.axvspan(_k1, kappas[-1], alpha=0.2, color='green', zorder=0, label='No-oscillation')
    ax.text((kappas[0]+_k0)/2, 125, 'Chaotic\nregion',         fontsize=12, ha='center', va='center', color='teal',      rotation=90, style='italic')
    ax.text((_k1+kappas[-1])/2, 125, 'No-oscillation\nregion', fontsize=12, ha='center', va='center', color='darkgreen', rotation=90, style='italic')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    n_workers = max(1, cpu_count())

    # Panel a: full κ bifurcation diagram
    kappas_a = np.linspace(0.04, 0.24, 300)
    try:
        raw_a = _continuation_sweep_both_directions('kappa', kappas_a, {}, 'a')
        bif_a = _extrema_from_sweep(raw_a)
        save_panel_a(bif_a, kappas_a, os.path.join(PANELS_DIR, 'fig3_panel_a.pdf'))
    except Exception:
        print('Panel a FAILED:'); traceback.print_exc()

    # Panel b: period-doubling / chaotic window detail
    kappas_b = np.linspace(0.105, 0.130, 400)
    try:
        raw_b = _continuation_sweep('kappa', kappas_b, {}, 'b')
        bif_b = _extrema_from_sweep(raw_b)
        save_panel_b(bif_b, kappas_b, os.path.join(PANELS_DIR, 'fig3_panel_b.pdf'))
    except Exception:
        print('Panel b FAILED:'); traceback.print_exc()

    # Panel c: period and amplitude vs sinking velocity
    v_values = np.linspace(0.025, 0.046, 50)
    try:
        raw_c = _parallel_sweep('v', v_values, {'kappa': 0.12}, 'c', n_workers)
        per_v, amp_v = _period_amp_from_sweep(raw_c)
        save_panel_c(v_values, per_v, amp_v,
                     os.path.join(PANELS_DIR, 'fig3_panel_c.pdf'))
    except Exception:
        print('Panel c FAILED:'); traceback.print_exc()

    # Panel d: period and amplitude vs turbulent diffusivity
    kappas_d = np.linspace(0.10, 0.23, 50)
    try:
        raw_d = _parallel_sweep('kappa', kappas_d, {}, 'd', n_workers)
        per_k, amp_k = _period_amp_from_sweep(raw_d)
        save_panel_d(kappas_d, per_k, amp_k,
                     os.path.join(PANELS_DIR, 'fig3_panel_d.pdf'))
    except Exception:
        print('Panel d FAILED:'); traceback.print_exc()

if __name__ == '__main__':
    main()
