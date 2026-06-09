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
from dcm_model import DEFAULT_PARAMS, make_grid, run_simulation, Iin_seasonal

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR  = os.path.join(SCRIPTS_DIR, '..', 'graphs')
os.makedirs(GRAPHS_DIR, exist_ok=True)

NZ = 250
Z = make_grid(zB=DEFAULT_PARAMS['zB'], nz=NZ)
Z_DISPLAY = (70, 120)

T_SPINUP = 1700

CONFIGS = [
    (0.50, 'a', 2500, False),
    (0.18, 'b', 1200, False),
    (0.12, 'c', 1200, False),
    (0.50, 'd', 2500, True),
    (0.14, 'e', 2500, True),
    (0.08, 'f', 2500, True),
]
KAPPA_LABEL = {'a':0.50,'b':0.18,'c':0.12,'d':0.50,'e':0.14,'f':0.08}

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

    # Discard spin-up: keep only t >= T_SPINUP
    keep  = res['t'] >= T_SPINUP
    t_out = res['t'][keep] - T_SPINUP
    return dict(t=t_out, z=res['z'],
                P=res['P'][:, keep],
                N=res['N'][:, keep],
                sol=res['sol'])

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
        ax.set_ylabel('Depth (m)', fontsize=16)
    if xlabel:
        ax.set_xlabel('Time (days)', fontsize=16)
    ax.tick_params(labelsize=16)
    return im, iz


def add_nutrient_contours(ax, t, z, N, iz, vmin, vmax, n_levels=8):
    """Overlay contour lines on the nutrient panel using the same jet colormap
    to highlight the oscillating wave structure."""
    F = N[iz, :]
    z_sub = z[iz]
    levels = np.linspace(vmin + (vmax - vmin) * 0.05, vmax * 0.95, n_levels)
    ax.contour(t, z_sub, F,
               levels=levels,
               cmap='jet', vmin=vmin, vmax=vmax,
               linewidths=0.8, alpha=0.6)


def save_panel(res, lbl, seasonal, out_path):
    t = res['t']
    P = res['P'] / 1e7
    N = res['N']

    # Compute percentiles only on the displayed depth slice (iz),
    # so deep high-N water below the display window does not inflate vmax
    iz = (res['z'] >= Z_DISPLAY[0]) & (res['z'] <= Z_DISPLAY[1])
    P_disp = P[iz, :]
    N_disp = N[iz, :]

    Pvmax = np.nanpercentile(P_disp, 99.5) or 1.0
    Nvmax = np.nanpercentile(N_disp, 98)   or 1.0
    Nvmin = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    im0, iz0 = make_colorplot(axes[0], t, res['z'], P, 'jet', 0, Pvmax)
    axes[0].set_title('P  (×10⁷ cells m⁻³)', fontsize=16)
    axes[0].text(0.02, 0.96, lbl, transform=axes[0].transAxes,
                 fontsize=18, fontweight='bold', va='top', color='white')
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=14)

    im1, iz1 = make_colorplot(axes[1], t, res['z'], N, 'jet', Nvmin, Nvmax)
    add_nutrient_contours(axes[1], t, res['z'], N, iz1, Nvmin, Nvmax)
    axes[1].set_title('N  (mmol nutrient m⁻³)', fontsize=16)
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    results = {}
    for kappa, lbl, t_display, seasonal in CONFIGS:
        panel_path = os.path.join(GRAPHS_DIR, f'fig2_panel_{lbl}.pdf')
        try:
            res = run_case(kappa, seasonal, t_display)
            save_panel(res, lbl, seasonal, panel_path)
            results[lbl] = res
        except Exception:
            print(f'  FAILED:'); traceback.print_exc()

if __name__ == '__main__':
    main()
