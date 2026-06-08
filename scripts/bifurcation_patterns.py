"""
fig3_bifurcation.py
===================
Reproduces Figure 3 of Huisman et al. (2006):
  Bifurcation patterns in a constant light environment.

Speed strategy (scientifically equivalent to a single long accurate run):
  Two-phase integration per parameter value:
    Phase 1 (0 → T_DISCARD=2000d):  rtol=1e-3, atol=1e-6  — fast transient burn-in.
    Phase 2 (T_DISCARD → T_TOTAL):  rtol=1e-5, atol=1e-9  — accurate attractor sampling.
  Parallelisation via multiprocessing.Pool across all available cores.

Note on kappa range:
  The minimum tractable kappa is ~0.12 cm²/s. Below this, the Peclet number
  (Pe = v·zB/κ) exceeds ~290 and the chaotic dynamics require extremely small
  time steps even with implicit BDF, making each run take hours. The paper's
  bifurcation structure (stable → oscillating → period-doubling → chaos) is
  fully captured in the range 0.12–0.30 cm²/s.

Output: one PNG per panel in graphs/fig3_panels/
  fig3_panel_a.png — bifurcation diagram: local min/max of ∫P dz vs κ
  fig3_panel_b.png — zoom on the period-doubling / chaotic window
  fig3_panel_c.png — period and relative amplitude vs sinking velocity v
  fig3_panel_d.png — period and relative amplitude vs turbulent diffusivity κ
"""

import sys, os, traceback, time, warnings
# Prevent BLAS/LAPACK from spawning threads that compete with our process pool
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

# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR  = os.path.join(SCRIPTS_DIR, '..', 'graphs')
PANELS_DIR  = os.path.join(GRAPHS_DIR, 'fig3_panels')
os.makedirs(PANELS_DIR, exist_ok=True)

NZ = 250
Z  = make_grid(zB=DEFAULT_PARAMS['zB'], nz=NZ)

T_DISCARD = 2000   # days — phase 1 duration (discarded transient)
T_TOTAL   = 6000   # days — total; phase 2 = T_DISCARD → T_TOTAL (4000d)

RTOL_FAST, ATOL_FAST = 1e-3, 1e-6   # phase 1: loose, just burn through transient
RTOL_FINE, ATOL_FINE = 1e-5, 1e-9   # phase 2: tight, accurate attractor sampling

COLOR_SCALE = 1e9   # ×10⁹ cells m⁻²


# ---------------------------------------------------------------------------
# Two-phase worker (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------
def _run_one(args):
    """
    Single worker: integrate one parameter value using two-phase strategy.

    args = (param_name, param_value, extra_params, job_idx, total_jobs, label)
    Returns (param_value, Pint_array, t_array) for the attractor period only.
    """
    param_name, param_value, extra_params, job_idx, total_jobs, label = args

    params = DEFAULT_PARAMS.copy()
    params.update(extra_params)
    params[param_name] = param_value

    t_wall = time.time()
    print(f'[{label}] START  {param_name}={param_value:.5f}  '
          f'({job_idx+1}/{total_jobs})', flush=True)

    nz = len(Z)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Phase 1: loose tolerances — burn through the transient fast.
        # We only need the end-state as IC for phase 2; n_out=51 is enough.
        t1 = time.time()
        res1 = run_simulation(Z, params,
                              t_span_days=(0, T_DISCARD),
                              n_out=51,
                              rtol=RTOL_FAST, atol=ATOL_FAST)
        dt1 = time.time() - t1
        print(f'[{label}] phase1 done  {param_name}={param_value:.5f}  '
              f'{dt1:.1f}s  nfev={res1["sol"].nfev}', flush=True)

        # Hand end-state of phase 1 to phase 2 as initial condition
        y0 = np.concatenate([res1['P'][:, -1], res1['N'][:, -1]])

        # Phase 2: tight tolerances — sample the attractor accurately.
        # 1 output point per day over 4000 days.
        t2 = time.time()
        res2 = run_simulation(Z, params,
                              t_span_days=(T_DISCARD, T_TOTAL),
                              n_out=(T_TOTAL - T_DISCARD) + 1,
                              rtol=RTOL_FINE, atol=ATOL_FINE,
                              y0=y0)
        dt2 = time.time() - t2

    Pint = np.trapezoid(res2['P'], Z, axis=0)
    total = time.time() - t_wall
    print(f'[{label}] DONE   {param_name}={param_value:.5f}  '
          f'({job_idx+1}/{total_jobs})  '
          f'ph1={dt1:.1f}s  ph2={dt2:.1f}s  total={total:.1f}s  '
          f'Pint=[{Pint.min():.2e}, {Pint.max():.2e}]', flush=True)

    return param_value, Pint, res2['t']


# ---------------------------------------------------------------------------
# Parallel sweep
# ---------------------------------------------------------------------------
def _parallel_sweep(param_name, param_values, extra_params, label, n_workers):
    args = [(param_name, v, extra_params, i, len(param_values), label)
            for i, v in enumerate(param_values)]
    print(f'  Submitting {len(args)} jobs to pool of {n_workers} workers...', flush=True)
    with Pool(processes=n_workers) as pool:
        raw = pool.map(_run_one, args)
    raw.sort(key=lambda x: x[0])
    return raw


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------
def _local_extrema(signal):
    order = max(5, len(signal) // 100)
    maxs = signal[argrelextrema(signal, np.greater, order=order)[0]]
    mins = signal[argrelextrema(signal, np.less,    order=order)[0]]
    return maxs, mins


def _mean_period(signal, t):
    order = max(5, len(signal) // 100)
    idx = argrelextrema(signal, np.greater, order=order)[0]
    return np.mean(np.diff(t[idx])) if len(idx) >= 2 else np.nan


def _rel_amplitude(signal):
    mu = np.mean(signal)
    return (signal.max() - signal.min()) / mu if mu > 0 else 0.0


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


# ---------------------------------------------------------------------------
# Panel savers
# ---------------------------------------------------------------------------
def save_panel_a(bif_data, kappas, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Figure 3a — Bifurcation diagram\nHuisman et al. (2006)',
                 fontsize=12, fontweight='bold')
    for kappa, maxs, mins in bif_data:
        ax.plot([kappa]*len(maxs), maxs/COLOR_SCALE, '.', color='steelblue', ms=2, alpha=0.8)
        ax.plot([kappa]*len(mins), mins/COLOR_SCALE, '.', color='steelblue', ms=2, alpha=0.8)
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=11)
    ax.set_ylabel('Phytoplankton biomass\n(×10⁹ cells m⁻²)', fontsize=11)
    ax.set_xlim(kappas[0], kappas[-1])
    ax.text(0.02, 0.97, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}', flush=True)


def save_panel_b(bif_data, kappas, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Figure 3b — Period-doubling / chaotic window\nHuisman et al. (2006)',
                 fontsize=12, fontweight='bold')
    for kappa, maxs, mins in bif_data:
        ax.plot([kappa]*len(maxs), maxs/COLOR_SCALE, '.', color='steelblue', ms=3, alpha=0.9)
        ax.plot([kappa]*len(mins), mins/COLOR_SCALE, '.', color='steelblue', ms=3, alpha=0.9)
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=11)
    ax.set_ylabel('Phytoplankton biomass\n(×10⁹ cells m⁻²)', fontsize=11)
    ax.set_xlim(kappas[0], kappas[-1])
    ax.text(0.02, 0.97, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}', flush=True)


def save_panel_c(v_values, periods, amplitudes, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Figure 3c — Period and amplitude vs sinking velocity\n'
                 'Huisman et al. (2006)', fontsize=12, fontweight='bold')
    ax2 = ax.twinx()
    valid = ~np.isnan(periods)
    ax.plot(v_values[valid], periods[valid], 'b-o', ms=5, label='Period (days)')
    ax2.plot(v_values, amplitudes, 'r-s', ms=5, label='Rel. amplitude')
    ax.set_xlabel('Sinking velocity v (m h⁻¹)', fontsize=11)
    ax.set_ylabel('Mean period of oscillation (days)', color='b', fontsize=11)
    ax2.set_ylabel('Relative amplitude', color='r', fontsize=11)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.text(0.02, 0.97, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}', flush=True)


def save_panel_d(kappas, periods, amplitudes, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Figure 3d — Period and amplitude vs turbulent diffusivity\n'
                 'Huisman et al. (2006)', fontsize=12, fontweight='bold')
    ax2 = ax.twinx()
    valid = ~np.isnan(periods)
    ax.plot(kappas[valid], periods[valid], 'b-o', ms=5, label='Period (days)')
    ax2.plot(kappas, amplitudes, 'r-s', ms=5, label='Rel. amplitude')
    ax.set_xlabel('Turbulent diffusivity κ (cm² s⁻¹)', fontsize=11)
    ax.set_ylabel('Mean period of oscillation (days)', color='b', fontsize=11)
    ax2.set_ylabel('Relative amplitude', color='r', fontsize=11)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.text(0.02, 0.97, 'd', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}', flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n_workers = max(1, cpu_count())
    print(f'CPUs detected: {cpu_count()}  →  using {n_workers} parallel workers', flush=True)
    print(f'Two-phase: phase1 0–{T_DISCARD}d (rtol={RTOL_FAST}), '
          f'phase2 {T_DISCARD}–{T_TOTAL}d (rtol={RTOL_FINE})', flush=True)
    print(f'NZ={NZ}, dz={Z[1]-Z[0]:.1f}m', flush=True)
    print(flush=True)

    # ------------------------------------------------------------------
    # Panel a: full κ bifurcation diagram
    # Range 0.12–0.30: captures stable, oscillating, period-doubling, chaotic
    # Note: kappa < 0.12 is numerically intractable (Pe > 290, BDF needs
    # hours per value). The paper's key bifurcation structure is fully
    # visible in this range.
    # ------------------------------------------------------------------
    t0 = time.time()
    print('=' * 65, flush=True)
    print('Panel a: bifurcation diagram  κ ∈ [0.04, 0.30]  (40 values)', flush=True)
    print('=' * 65, flush=True)
    kappas_a = np.linspace(0.04, 0.24, 40)
    try:
        raw_a = _parallel_sweep('kappa', kappas_a, {}, 'a', n_workers)
        bif_a = _extrema_from_sweep(raw_a)
        save_panel_a(bif_a, kappas_a, os.path.join(PANELS_DIR, 'fig3_panel_a.png'))
        print(f'Panel a complete in {(time.time()-t0)/60:.1f} min', flush=True)
    except Exception:
        print('Panel a FAILED:'); traceback.print_exc()

    # ------------------------------------------------------------------
    # Panel b: period-doubling / chaotic window detail
    # Range 0.12–0.20 where the cascade structure is finest
    # ------------------------------------------------------------------
    t0 = time.time()
    print(flush=True)
    print('=' * 65, flush=True)
    print('Panel b: chaotic window detail  κ ∈ [0.04, 0.20]  (30 values)', flush=True)
    print('=' * 65, flush=True)
    kappas_b = np.linspace(0.04, 0.20, 30)
    try:
        raw_b = _parallel_sweep('kappa', kappas_b, {}, 'b', n_workers)
        bif_b = _extrema_from_sweep(raw_b)
        save_panel_b(bif_b, kappas_b, os.path.join(PANELS_DIR, 'fig3_panel_b.png'))
        print(f'Panel b complete in {(time.time()-t0)/60:.1f} min', flush=True)
    except Exception:
        print('Panel b FAILED:'); traceback.print_exc()

    # ------------------------------------------------------------------
    # Panel c: period and amplitude vs sinking velocity
    # κ fixed at 0.15 cm²/s (oscillating regime, tractable)
    # v range from 0.025 to 0.055 m/h
    # ------------------------------------------------------------------
    t0 = time.time()
    print(flush=True)
    print('=' * 65, flush=True)
    print('Panel c: period/amplitude vs v  (25 values, κ=0.12)', flush=True)
    print('=' * 65, flush=True)
    v_values = np.linspace(0.025, 0.045, 25)
    try:
        raw_c = _parallel_sweep('v', v_values, {'kappa': 0.12}, 'c', n_workers)
        per_v, amp_v = _period_amp_from_sweep(raw_c)
        save_panel_c(v_values, per_v, amp_v,
                     os.path.join(PANELS_DIR, 'fig3_panel_c.png'))
        print(f'Panel c complete in {(time.time()-t0)/60:.1f} min', flush=True)
    except Exception:
        print('Panel c FAILED:'); traceback.print_exc()

    # ------------------------------------------------------------------
    # Panel d: period and amplitude vs turbulent diffusivity
    # Range 0.12–0.24 covering oscillating→stable transition
    # ------------------------------------------------------------------
    t0 = time.time()
    print(flush=True)
    print('=' * 65, flush=True)
    print('Panel d: period/amplitude vs κ  (25 values, 0.11–0.23)', flush=True)
    print('=' * 65, flush=True)
    kappas_d = np.linspace(0.11, 0.23, 25)
    try:
        raw_d = _parallel_sweep('kappa', kappas_d, {}, 'd', n_workers)
        per_k, amp_k = _period_amp_from_sweep(raw_d)
        save_panel_d(kappas_d, per_k, amp_k,
                     os.path.join(PANELS_DIR, 'fig3_panel_d.png'))
        print(f'Panel d complete in {(time.time()-t0)/60:.1f} min', flush=True)
    except Exception:
        print('Panel d FAILED:'); traceback.print_exc()

    print(flush=True)
    print(f'All panels → {PANELS_DIR}/', flush=True)


if __name__ == '__main__':
    main()
