"""
Model simulations at different intensities of vertical mixing.
    - (a): Constant environment and stable DCM (κ=0.50 cm² s⁻¹)
    - (b): Constant environment and mild oscillations in DCM (κ=0.20 cm² s⁻¹)
    - (c): Constant environment and large amplitude oscillations in DCM (κ=0.12 cm² s⁻¹)
    - (d): Seasonal environment and DCM tracks seasonal variability (κ=0.50 cm² s⁻¹)
    - (e): Seasonal environment and double periodicity of DCM (κ=0.14 cm² s⁻¹)
    - (f): Seasonal environment and chaotic DCM (κ=0.08 cm² s⁻¹)
"""

import sys, os, traceback, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dcm_model import DEFAULT_PARAMS, make_grid, run_simulation, Iin_seasonal

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR  = os.path.join(SCRIPTS_DIR, '..', 'graphs')
os.makedirs(GRAPHS_DIR, exist_ok=True)

NZ = 250 # 150
Z = make_grid(zB=DEFAULT_PARAMS['zB'], nz=NZ)
Z_DISPLAY = (70, 120)
DAYS_PER_OUTPUT = 1.0

# Spin-up
T_SPINUP = 1700

# (kappa, label, t_display_days, seasonal)
CONFIGS = [
    (0.50, 'a', 2500, False),
    (0.18, 'b', 1200, False),
    (0.12, 'c', 1200, False),
    (0.50, 'd', 2500, True),
    (0.14, 'e', 2500, True),
    (0.08, 'f', 2500, True),
]
KAPPA_LABEL = {'a':0.50,'b':0.18,'c':0.12,'d':0.50,'e':0.14,'f':0.08}

# Runner
def run_case(kappa, seasonal, t_display):
    params = DEFAULT_PARAMS.copy()
    params['kappa'] = kappa
    fn = Iin_seasonal if seasonal else None
    t_total = T_SPINUP + t_display
    n_out = t_total + 1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = run_simulation(Z, params, (0, t_total),
                             Iin_func=fn, n_out=n_out,
                             rtol=1e-6, atol=1e-10)

    # Discard the spin-up: keep only t >= T_SPINUP
    keep  = res['t'] >= T_SPINUP
    t_out = res['t'][keep] - T_SPINUP
    return dict(t=t_out, z=res['z'],
                P=res['P'][:, keep],
                N=res['N'][:, keep],
                sol=res['sol'])

# Graphs
def make_colorplot(ax, t, z, field, cmap, vmin, vmax, xlabel=True, ylabel=True):
    iz = (z >= Z_DISPLAY[0]) & (z <= Z_DISPLAY[1])
    F = field[iz, :]

    im = ax.imshow(F,
                   aspect='auto',
                   origin='upper',
                   extent=[t[0], t[-1], Z_DISPLAY[1], Z_DISPLAY[0]],
                   cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   interpolation='bilinear')
    ax.set_ylim(Z_DISPLAY[1], Z_DISPLAY[0])
    if ylabel:
        ax.set_ylabel('Depth (m)', fontsize=10)
    if xlabel:
        ax.set_xlabel('Time (days)', fontsize=10)
    ax.tick_params(labelsize=9)
    return im


def save_panel(res, lbl, seasonal, out_path):
    kv  = KAPPA_LABEL[lbl]
    env = 'Seasonal' if seasonal else 'Constant'

    t = res['t']
    P = res['P'] / 1e7
    N = res['N']

    Pvmax = np.nanpercentile(P, 99.5) or 1.0
    Nvmax = np.nanpercentile(N, 45) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f'Panel {lbl} — {env} environment,  κ = {kv} cm² s⁻¹\n'
                 f'Huisman et al. (2006)', fontsize=12, fontweight='bold')

    im0 = make_colorplot(axes[0], t, res['z'], P, 'jet', 0, Pvmax)
    axes[0].set_title('P  (×10⁷ cells m⁻³)', fontsize=10)
    axes[0].text(0.02, 0.96, lbl, transform=axes[0].transAxes,
                 fontsize=13, fontweight='bold', va='top', color='white')
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=8)

    im1 = make_colorplot(axes[1], t, res['z'], N, 'jet', 0, Nvmax)
    axes[1].set_title('N  (mmol nutrient m⁻³)', fontsize=10)
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    results = {}

    for kappa, lbl, t_display, seasonal in CONFIGS:
        env   = 'seasonal' if seasonal else 'constant'
        t_total = T_SPINUP + t_display
        print(f'Panel {lbl}: κ={kappa} cm²/s, {env}, spinup={T_SPINUP}d + display={t_display}d ...', flush=True)
        panel_path = os.path.join(GRAPHS_DIR, f'fig2_panel_{lbl}.png')
        try:
            res = run_case(kappa, seasonal, t_display)
            ok  = res['sol'].success
            print(f'  {"OK" if ok else "WARN"}  nfev={res["sol"].nfev}  '
                  f'P_max={res["P"].max():.2e}  display_pts={res["t"].size}')
            save_panel(res, lbl, seasonal, panel_path)
            results[lbl] = res
        except Exception:
            print(f'  FAILED:'); traceback.print_exc()

    print(f'Panels   → {GRAPHS_DIR}/')

if __name__ == '__main__':
    main()
