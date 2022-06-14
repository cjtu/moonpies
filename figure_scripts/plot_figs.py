"""Moonpies plotting module."""
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns
from moonpies import moonpies as mp
from moonpies import default_config
from moonpies import plotting as mplt

# Plot helpers
CFG = default_config.Cfg(seed=1)
FIGDIR = str(Path(CFG.figs_path).resolve() / "_")[:-1]  # make str with trailing slash
mplt.reset_plot_style()  # Sets Moonpies style

# Helpers
def _save_or_show(fig, ax, fsave, figdir, version='', **kwargs):
    """Save figure or show fig and return ax."""
    if fsave:
        # Append version
        fsave = Path(fsave)
        fsave = fsave.with_name(f'{fsave.stem}_{version}{fsave.suffix}')
        fig.savefig(Path(figdir) / fsave, **kwargs)
    else:
        plt.show()
    return ax


def plot_all(figdir=FIGDIR):
    """Produce all figures in figdir with default args."""
    # ballistic_hop(figdir=figdir)
    basin_ice(figdir=figdir)


def ballistic_hop(fsave='bhop_lat.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot ballistic hop efficiency by latitude.

    :Authors:
        K. M. Luchsinger, C. J. Tai Udovicic
    """
    mplt.reset_plot_style()
    coldtraps = ['Haworth', 'Shoemaker', 'Faustini', 'Amundsen', 'Cabeus',
                 'Cabeus B', 'de Gerlache', "Idel'son L", 'Sverdrup', 
                 'Shackleton', 'Wiechert J', "Slater"]
    lats = np.array([87.5, 88., 87.1, 84.4, 85.3, 82.3, 88.3, 84., 88.3, 
                     89.6, 85.2, 88.1])
    label_offset = np.array([(-0.7, 0.1), (-0.9, -0.6), (-0.5, 0.2), (-0.3, -0.7), (-0.1, -0.7), (-0.2, -0.7), (0.1, 0.2), 
                            (-0.5, 0.2), (-0.1, -0.8), (-0.95, -0.45), (-0.7, 0.2), (-0.45, 0.1)])
    coldtraps_moores = ["Haworth", "Shoemaker", "Faustini", "de Gerlache", 
                        "Sverdrup", "Shackleton", "Cabeus"]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.grid(True)
    # Plot cold trap bhop vs. lat
    bhops = mp.read_ballistic_hop_csv(cfg.bhop_csv_in)
    for i, (name, lat) in enumerate(zip(coldtraps, lats)):
        bhop = 100*bhops.loc[name]
        color = 'tab:orange'
        marker = 's'
        kw = dict(markerfacecolor='white', markeredgewidth=2, zorder=10)
        label = None
        if name in coldtraps_moores:
            color = 'tab:blue'
            marker = 'o'
            kw = {}
        if name == 'Haworth':
            label = 'Moores et al. (2016)'
        elif name == 'Cabeus B':
            label = 'This work'
            
        ax.plot(lat, bhop, marker, c=color, label=label, **kw)
        off_x, off_y = label_offset[i]
        ax.annotate(name, (lat, bhop), xytext=(lat + off_x, bhop+off_y), ha='left', va='bottom')
    
    # Cannon et al. (2020) constant bhop
    ax.axhline(5.4, c='tab:gray', ls='--')
    ax.annotate('Cannon et al. (2020)', (90, 5.3), ha='right', va='top')


    # Simple linear fit to moores data
    bhop_moores = [100*bhops.loc[name] for name in coldtraps if name in coldtraps_moores]
    lats_moores = [lat for name, lat in zip(coldtraps, lats) if name in coldtraps_moores]
    fit = np.poly1d(np.squeeze(np.polyfit(lats_moores, bhop_moores, 1)))
    lat = np.linspace(89.6, 85.6, 10)
    ax.plot(lat, fit(lat), '--')
    ax.annotate("Fit to Moores et al. (2016)", (86.6, fit(86.6)), va='top', ha='right')

    # Line from Cabeus B to Faustini
    bhop_cf = 100*bhops.loc[['Cabeus B', 'Faustini']].values
    ax.plot([82.3, 87.1], bhop_cf, '--', c='tab:orange')

    ax.set_xlim(82, 90)
    ax.set_ylim(0, 7)
    ax.set_xlabel("Latitude [Degrees]")
    ax.set_ylabel("Ballistic Hop Efficiency [% per km$^{2}$]")
    ax.set_title("Ballistic Hop Efficiency by Latitude")
    ax.legend()
    version = mplt.plot_version(cfg, loc='ll', ax=ax)
    fig.tight_layout()
    return _save_or_show(fig, ax, fsave, figdir, version)


def basin_ice(fsave='basin_ice.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot basin ice volume by latitude.

    :Authors:
        K. R. Frizzell, C. J. Tai Udovicic
    """
    def moving_avg(x, w):
        """Return moving average of x with window size w."""
        return np.convolve(x, np.ones(w), 'same') / w
    mplt.reset_plot_style()
    # Params
    n = 500  # Number of runs (~1 min / 500 runs)
    seed0 = 200  # Starting seed
    window = 3  # Window size for moving average
    color = {'Asteroid': 'tab:gray', 'Comet': 'tab:blue'}
    marker = {'Asteroid': 'x', 'Comet': '+'}

    time_arr = mp.get_time_array()
    fig, ax = plt.subplots()
    for btype in ('Asteroid', 'Comet'):
        all_basin_ice = np.zeros([len(time_arr), n])
        cdict = cfg.to_dict()
        for i, seed in enumerate(range(seed0, seed0+n)):
            mp.clear_cache()
            cdict['seed'] = seed
            cfg = default_config.Cfg(**cdict)
            df = mp.get_crater_basin_list(cfg)
            if btype == 'Comet':
                cfg = mp.get_comet_cfg(cfg)
            b_ice_t = mp.get_basin_ice(time_arr, df, cfg)
            all_basin_ice[:, i] = b_ice_t
        x = time_arr / 1e9
        ax.semilogy(x, all_basin_ice, marker[btype], c=color[btype], alpha=0.5)
        # median = moving_avg(np.median(all_basin_ice,axis=1), window)
        mean = moving_avg(np.mean(all_basin_ice,axis=1), window)
        # pct99 = moving_avg(np.percentile(all_basin_ice, 99, axis=1), window)
        ax.plot(x, mean, '-', c=color[btype], lw=2, label=btype+' mean')
        # ax.plot(x, pct99,'--', c=color[btype], lw=1, label='99th percentile')
    ax.grid('on')
    ax.set_xlim(4.25, 3.79)
    ax.set_ylim(0.1, None)
    ax.set_title(f'Ice Delivered to South Pole by Basins ({n} runs)')
    ax.set_xlabel('Time [Ga]')
    ax.set_ylabel('Ice Thickness [m]')
    ax.legend(loc='upper right')
    version = mplt.plot_version(cfg, loc='ul', ax=ax)
    return _save_or_show(fig, ax, fsave, figdir, version)


def comet_vels(fsave='comet_vels.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot comet velocities.

    :Authors:
        K. M. Luchsinger, C. J. Tai Udovicic
    """
    mplt.reset_plot_style()
    rng = mp.get_rng(cfg)
    
    # Get probability distributions and random samples
    x = np.linspace(0, 70000, int(1e5))
    h = stats.norm.pdf(x=x, loc=cfg.halley_mean_speed, scale=cfg.halley_sd_speed)
    o = stats.norm.pdf(x=x, loc=cfg.oort_mean_speed, scale=cfg.oort_sd_speed)
    p = cfg.halley_frac * h + cfg.oort_frac * o
    speeds = mp.get_comet_speeds(int(1e6), cfg, rng)

    fig, ax = plt.subplots()
    ax.plot(x, h, c='tab:blue', label='Halley pdf')
    ax.plot(x, o, c='tab:green', label='Oort pdf')
    ax.plot(x, p, '-.', c='tab:orange', label='Weighted pdf')
    ax.hist(speeds, 50, histtype='stepfilled', density=True, label='Random sample')
    ax.set_xlim(0, 80000)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticks()/1e3)  # [m/s] -> [km/s]
    ax.set_xlabel('Comet speed [km/s]')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')
    version = mplt.plot_version(cfg, loc='ul', ax=ax)
    return _save_or_show(fig, ax, fsave, figdir, version)


def crater_basin_ages(fsave='crater_basin_ages.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot crater basin ages.

    :Authors:
        A. Madera, C. J. Tai Udovicic
    """
    mplt.reset_plot_style()
    sns.set_style("ticks", {'ytick.direction': 'in'})
    
    # Plot params
    fs_large = 12

    dc = mp.read_crater_list(CFG).set_index('cname')
    db = mp.read_basin_list(CFG).set_index('cname')

    nec_ga = db.loc["Nectaris", "age"]  # Necarian
    imb_ga = db.loc["Imbrium", "age"]  # Imbrian
    era_ga = 3.2  # Eratosthenian
    cop_ga = 1.1  # Copernican
    eras = {
        'pNe.': dict(xmin=4.5, xmax=nec_ga, facecolor='#253494', alpha=0.3),
        'Ne.': dict(xmin=nec_ga, xmax=imb_ga, facecolor='#2c7fb8', alpha=0.4),
        'Im.': dict(xmin=imb_ga, xmax=era_ga, facecolor='#41b6c4', alpha=0.3),
        'Era.': dict(xmin=era_ga, xmax=cop_ga, facecolor='#a1dab4', alpha=0.3)
    }

    # Figure and main crater axis
    fig, axc = plt.subplots(figsize=(7.2, 9))

    # Make basin inset
    left, bottom, width, height = [0.53, 0.3, 0.35, 0.55]
    axb = fig.add_axes([left, bottom, width, height])

    for ax, df in zip([axc, axb], [dc, db]):
        # Plot crater / basin ages
        df = df.sort_values(['age', 'age_upp'])
        age, low, upp = df[["age", "age_low", "age_upp"]].values.T / 1e9
        ax.errorbar(age, df.index, xerr=(low, upp), fmt='ko', ms=5, capsize=4, 
                    capthick=2)
        ax.invert_xaxis()
        if ax == axc:
            ax.set_ylabel('Craters', fontsize=fs_large, labelpad=-0.2)
            ax.set_xlabel('Absolute Model Ages [Ga]', fontsize=fs_large)
            ax.set_xlim(4.4, 1.4)
            ax.tick_params(axis='y')
        else:
            ax.set_title('Basins', pad=3, fontsize=fs_large)
            ax.set_xlabel('Absolute Model Ages [Ga]')
            ax.set_xlim(4.35, 3.79)

        # Add Chronological Periods
        for era, params in eras.items():
            ax.axvspan(**params, edgecolor='none')
            if ax == axb and era == 'Era.':
                continue
            x = max(params['xmax'], ax.get_xlim()[1])
            y = ax.get_ylim()[0]
            ax.annotate(era, xy=(x, y), xycoords='data', fontsize=fs_large, 
                        weight='bold', ha='right', va='bottom')
    mplt.reset_plot_style()
    version = mplt.plot_version(cfg, loc='lr', xyoff=(0.01, -0.1), ax=axc)
    return _save_or_show(fig, axc, fsave, figdir, version)


if __name__ == '__main__':
    plot_all()