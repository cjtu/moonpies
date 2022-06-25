"""Moonpies plotting module."""
from datetime import datetime
from pathlib import Path
from multiprocessing import Process
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from moonpies import moonpies as mp
from moonpies import default_config
from moonpies import plotting as mplt
import aggregate as agg

# Plot helpers
CFG = default_config.Cfg(seed=1)
DATE = datetime.now().strftime("%y%m%d")
FIGDIR = Path(CFG.figs_path) / f'{DATE}_v{CFG.version}'
mplt.reset_plot_style()  # Sets Moonpies style

# Helpers
def _save_or_show(fig, ax, fsave, figdir, version='', **kwargs):
    """Save figure or show fig and return ax."""
    if fsave:
        fsave = Path(figdir) / Path(fsave)  # Prepend path
        # Append version if not already in figdir or fsave
        if version not in fsave.as_posix():
            fsave = fsave.with_name(f'{fsave.stem}_{version}{fsave.suffix}')
        fsave.parent.mkdir(parents=True, exist_ok=True)  # Make dir if doesn't exist
        fig.savefig(fsave, **kwargs)
    else:
        plt.show()
    return ax


def _generate_all():
    """Produce all figures in figdir in parallel with default args."""
    # Get all functions in this module
    funcs = [obj for name, obj in globals().items() if callable(obj) and 
                obj.__module__ == __name__ and name[0] != '_']
    
    # Run each with defaults in its own process
    print(f'Starting {len(funcs)} plots...')
    procs = [Process(target=func) for func in funcs]
    _ = [p.start() for p in procs]
    _ = [p.join() for p in procs]
    print(f'All plots written to {FIGDIR}')


def ast_comet_vels(fsave='comet_vels.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot comet velocities.

    :Authors:
        K. M. Luchsinger, C. J. Tai Udovicic
    """
    mplt.reset_plot_style()
    rng = mp.get_rng(cfg)
    cfg_comet = mp.get_comet_cfg(cfg)
    n = int(1e5)
    
    # Get probability distributions and random samples
    rv_ast = mp.asteroid_speed_rv(cfg)
    rv_comet = mp.comet_speed_rv(cfg_comet)

    x = np.linspace(0, cfg.comet_speed_max, int(1e4))
    apdf = rv_ast.pdf(x)
    cpdf = rv_comet.pdf(x)
    aspeeds = mp.get_random_impactor_speeds(n, cfg, rng)
    cspeeds = mp.get_random_impactor_speeds(n, cfg_comet, rng)

    fig, axs = plt.subplots(2, sharex=True, figsize=(7.2, 9))
    # Plot asteroid and comet distributions
    ax = axs[0]
    ax.plot(x, apdf, '-.', c='tab:blue', lw=2, label='Asteroid pdf')
    ax.plot(x, cpdf, '-.', c='tab:orange', lw=2, label='Comet pdf')
    bins = np.linspace(0, cfg.comet_speed_max, 40)
    ax.hist(aspeeds, bins, histtype='stepfilled', density=True,
            color='tab:blue', alpha=0.2, label='Asteroid Random sample')
    ax.hist(cspeeds, bins, histtype='stepfilled', density=True,
            color='tab:orange', alpha=0.2, label='Comet Random sample')
    ax.set_xlim(0, 80000)
    # ax.set_xticks(ax.get_xticks())
    # ax.set_xticklabels(ax.get_xticks()/1e3)  # [m/s] -> [km/s]
    # ax.set_xlabel('Comet speed [km/s]')
    ax.set_ylabel('Density')
    ax.legend(loc='center right')
    

    # Plot comet mixed probability distribution
    ax = axs[1]
    ax.set_ylabel('Probability')
    ax.plot(x, rv_comet.sf(x), label='Survival function (SF)')
    ax.plot(x, rv_comet.cdf(x), label='CDF')
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    ax2.set_ylabel('Density')
    ax2.plot(x, rv_comet.pdf(x), '-.', c='tab:green', label='PDF')
    ax2.hist(cspeeds, bins=40, density=True, color='tab:gray', alpha=0.5, 
            zorder=0, label=f'Samples (n={n:.0e})')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, title='Comet distribution', 
              loc='center right')
    ax.set_xticks(ax.get_xticks())  # Neeed for setting labels
    ax.set_xticklabels(ax.get_xticks()/1e3)  # [m/s] -> [km/s]
    ax.set_xlabel('Speed [km/s]')
    ax.set_xlim(0, cfg.comet_speed_max)

    version = mplt.plot_version(cfg, loc='ur', ax=axs[0])
    return _save_or_show(fig, axs, fsave, figdir, version)

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

    nec_ga = db.loc["Nectaris", "age"] / 1e9  # Necarian
    imb_ga = db.loc["Imbrium", "age"] / 1e9 # Imbrian
    era_ga = 3.2  # Eratosthenian
    cop_ga = 1.1  # Copernican
    eras = {
        'pNe.': dict(xmin=4.5, xmax=nec_ga, facecolor='#253494', alpha=0.3),
        'Ne.': dict(xmin=nec_ga, xmax=imb_ga, facecolor='#2c7fb8', alpha=0.4),
        'Im.': dict(xmin=imb_ga, xmax=era_ga, facecolor='#41b6c4', alpha=0.3),
        'Era.': dict(xmin=era_ga, xmax=cop_ga, facecolor='#a1dab4', alpha=0.3)
    }

    # Figure and main crater axis
    fig, axc = plt.subplots(figsize=(7.2, 9.5))

    # Make basin inset
    left, bottom, width, height = [0.53, 0.35, 0.35, 0.5]
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


def crater_scaling(fsave='crater_scaling.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot crater scaling from final diameter to impactor diameter.

    :Authors:
        K. M. Luchsinger, C. J. Tai Udovicic
    """
    mplt.reset_plot_style()

    speed = 20000  # [m/s]
    dfb = mp.read_basin_list(cfg)
    labels = {
        'c': 'C: Small simple craters (Prieur et al., 2017)', 
        'd': 'D: Large simple craters (Collins et al., 2005)', 
        'e': 'E: Complex craters (Johnson et al., 2016)',
        'f': 'F: Basins (Johnson et al., 2016)'}
    
    # TODO: fix regimes to be continuous in impactor space. 
    # Proposal: change c_min to 250 (i_diam=3) and e_min to 19e3 (i_diam=750)
    new_regimes = {
        # regime: (rad_min, rad_max, step, sfd_slope)
        'a': (0, 0.01, None, None),  # micrometeorites (<1 mm)
        'b': (0.01, 3, 1e-4, -3.7),  # small impactors (Cannon: 10 mm - 3 m)
        'c': (250, 1.5e3, 1, -3.82),  # simple craters, steep sfd (Cannon: 100 m - 1.5 km)
        'd': (1.5e3, 15e3, 1e2, -1.8),  # simple craters, shallow sfd (Cannon: 1.5 km - 15 km)
        'e': (19e3, 300e3, 1e3, -1.8),  # complex craters, shallow sfd (Cannon: 15 km - 300 km)
    }
    cdict = cfg.to_dict()
    cdict['impact_regimes'] = new_regimes
    alt_cfg = default_config.Cfg(**cdict)

    # Plot regimes c - e
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    for i, rcfg in enumerate([cfg, alt_cfg]):
        for regime, (dmin, dmax, *_) in rcfg.impact_regimes.items():
            if regime in ('a', 'b'):
                continue
            diams = np.geomspace(dmin, dmax, 2)
            lengths = mp.diam2len(diams, speed, regime, rcfg)
            label = labels[regime] if i == 0 else ''
            fmt = '-' if i == 0 else 'rx'
            ax.loglog(diams/1e3, lengths, fmt, label=label)

    # Plot individual basins
    basin_lengths = mp.diam2len(dfb.diam, speed, 'f', cfg)
    ax.loglog(dfb.diam/1e3, basin_lengths, 'k+', label=labels['f'])
    ax.set_ylim(0.3, None)
    ax.set_xlabel('Crater Diameter [km]')
    ax.set_ylabel('Impactor Diameter [km]')
    ax.set_title(f'Crater to Impactor Scaling (with $v$={speed/1e3} km/s)')

    # Plot transitions
    ax.axvline(cfg.simple2complex/1e3, c='k', ls='--')
    ax.annotate('Simple to complex', xy=(cfg.simple2complex/1e3, 1), ha='right', rotation=90)
    ax.axvline(cfg.complex2peakring/1e3, c='k', ls='--')
    ax.annotate('Complex to basin', xy=(cfg.complex2peakring/1e3, 1), ha='right', rotation=90)
    ax.legend(title='Regime', loc='upper left', fontsize=9)
    version = mplt.plot_version(cfg, loc='ll', ax=ax)
    return _save_or_show(fig, ax, fsave, figdir, version)


def distance_bsed(fsave='distance_bsed.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot distance between basins and craters.

    :Authors:
        K. M. Luchsinger, C. J. Tai Udovicic
    """
    mplt.reset_plot_style()
    cdict = cfg.to_dict() 
    cdict['ej_threshold'] = -1
    cfg = default_config.Cfg(**cdict)

    # Ries data
    fries = Path(cfg.data_path) / 'horz_etal_1983_table2_fig19.csv'
    ries = pd.read_csv(fries, usecols=[1, 2, 4, 5])
    ries.columns = ['dist_km', 'dist_crad', 'wt_pct', 'mixing_ratio']

    # Get mixing ratio and vol frac
    coldtraps = np.array(cfg.coldtrap_names)  # Order of 2nd axis of dists
    df = mp.get_crater_basin_list(cfg)
    dists = mp.get_coldtrap_dists(df, cfg)
    thick = mp.get_ejecta_thickness_matrix(df, dists, cfg)
    mixing_ratio = mp.get_mixing_ratio_oberbeck(dists, cfg)
    
    dist_m = ries.dist_km.values * 1e3
    mr_oberbeck = mp.get_mixing_ratio_oberbeck(dist_m, cfg)
    fig, axs = plt.subplots(2, figsize=(7.2, 9))
    
    # Ries
    ax = axs[0]
    ax.loglog(dist_m, ries.mixing_ratio, 'ro', label='Ries')
    ax.loglog(dist_m, mr_oberbeck, 'k--', label='Oberbeck')
    ax.set_title("Mixing ratio with distance from crater")
    ax.set_xlabel("Distance [Crater Radii]")
    ax.set_ylabel("Mixing ratio [target:ejecta]")

    # Mixing ratio of craters
    for coldtrap in coldtraps[:6]:
        idx = df[df.cname == coldtrap].index[0]
        # dist_crad = dists/df.rad.values[:, None]
        # ax.plot(dist_crad[idx], 1-vol_frac[idx], 'x', label=coldtrap)
        ax.loglog(dists[idx], mixing_ratio[idx], 'x', label=coldtrap)
    ax.legend()
    
    # Amundsen to Haworth, Shoemaker, Faustini
    aidx = df[df.cname == 'Amundsen'].index[0]
    hidx = np.argmax(coldtraps == 'Haworth')
    sidx = np.argmax(coldtraps == 'Shoemaker')
    fidx = np.argmax(coldtraps == 'Faustini')
    dfc = df[~df.isbasin]
    xmax = np.nanmax(dists[dfc.index])
    bsed_depths = -thick*mixing_ratio
    ax = axs[1]
    ax.plot(dists[0], thick[0], label="Ejecta thickness")
    ax.plot(dists, thick, 'x', c='tab:blue')
    ax.axhline(0, c='k', label='Surface', zorder=10)
    ax.plot(dists[0], bsed_depths[0], label="Ballistic Sed. Mixing Region")
    ax.plot(dists, bsed_depths, '+', c='tab:orange')
    
    ax.axvline(dists[aidx,fidx], color='red', ls='--', label='Amundsen->Faustini')
    ax.axvline(dists[aidx,sidx], color='orange', ls='--', label='Amundsen->Shoemaker')
    ax.axvline(dists[aidx,hidx], color='gold', ls='--', label='Amundsen->Haworth')
    ax.set_xlim(0, xmax)
    ax.legend()
    ax.set_title("Ballistic Sedimentation Mixing Region")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Thickness [m]")
    version = mplt.plot_version(cfg, loc='lr', ax=ax)
    return _save_or_show(fig, ax, fsave, figdir, version)


def ejecta_bsed(fsave='ejecta_bsed.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot ejecta thickness and ballistic mixing depth.
    """
    mplt.reset_plot_style()
    s2c = cfg.simple2complex / 2
    c2p = cfg.complex2peakring / 2
    radii = {
        's': np.geomspace(1e2, s2c-10, 10),
        'c': np.geomspace(s2c+10, c2p-10, 10),
        'b': np.geomspace(c2p+10, 6.6e5, 10)}
    labels = {'s': 'Simple crater', 'c': 'Complex crater', 'b': 'Basin'}
    ms = {'s': '^', 'c': 's', 'b': 'o'}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:gray']
    dist_arr = np.array([1e3, 1e4, 1e5, 1e6])

    # Plot
    fig, axs = plt.subplots(2, sharex=True, figsize=(5, 10))
    fig.subplots_adjust(hspace=0.05)
    for dist, color in zip(dist_arr, colors):
        for ctype, rads in radii.items():
            dists = np.ones(len(rads))*dist
            with np.errstate(invalid='ignore'):
                thick = mp.get_ejecta_thickness(dists, rads, cfg)
            thick[dist<rads] = 0
            label = None
            if color == 'tab:blue':
                label = f'{labels[ctype]} (D=({2*rads[0]/1e3:.1f}, {2*rads[-1]/1e3:.0f}) km)'
            ax = axs[0]
            ax.loglog(rads, thick, ms[ctype], c=color, label=label)
            ax.set_ylabel('Ejecta Thickness [km]')
            ax.legend(fontsize=9)

            ax = axs[1]
            mixing_ratio = mp.get_mixing_ratio_oberbeck(dists, cfg)
            bsed = thick*mixing_ratio
            ax.loglog(rads, bsed, ms[ctype], c=color, label=label)
            ax.set_ylabel('Ballistic Mixing Depth [km]')
            ax.set_xlabel('Radius [km]')
            ax.set_xlim(50, 7e5)
            ax.legend(fontsize=9)
            if ctype == 's':
                axs[0].annotate(f'Distance={dist/1e3:.0f} km', (rads[0]-40, thick[0]), rotation=47)
                axs[1].annotate(f'Distance={dist/1e3:.0f} km', (rads[0]-40, bsed[0]), rotation=47)
    version = mplt.plot_version(cfg, loc='lr', xyoff=(0.01, -0.15), ax=axs[1])
    return _save_or_show(fig, ax, fsave, figdir, version)


def kde_layers(fsave='kde_layers.pdf', figdir=FIGDIR, cfg=CFG, datedir='', coldtraps=None):
    """
    Plot KDE of ejecta thickness and ballistic mixing depth.
    """
    mplt.reset_plot_style()
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    if coldtraps is None:
        coldtraps = ["Haworth", "de Gerlache", "Slater", "Cabeus"]
    pal = sns.color_palette()
    mypal = [pal[0], pal[1]]
    hues = ['No', 'Yes']
    xlim = (0.05, 1000)
    ylim = (1, 800)
    skip = 100  # skip this many points for each coldtrap (for faster kde)

    # Load aggregated layer data
    if not datedir:  # Guess most recent date folder in outdir
        outdir = Path(cfg.out_path).parents[2]
        datedir = [p for p in outdir.iterdir() if p.is_dir()][-1]
    layers, _ = agg.read_agg_dfs(datedir, flatten=True)
    layers = agg.binary_runs(layers, 'mpies', rename='bsed')

    # Clean data
    layers.loc[layers['ice'] < 0.1, 'ice'] = np.nan
    layers.loc[layers['depth'] < 0.1, 'depth'] = np.nan
    layers.dropna(inplace=True)
    layers['depth_top'] = layers['depth'] - layers['ice']
    layers['depth_top'] = layers['depth_top'].clip(lower=0.1)

    # Make plots. 4 seaborn jointgrids then put together with SeabornFig2Grid
    fig = plt.figure(figsize=(7.2, 7.2))
    jgs = []
    for i, coldtrap in enumerate(coldtraps):
        df = layers[(layers.coldtrap==coldtrap)].iloc[::skip]
        jg = sns.JointGrid(space=0)
        jg.ax_joint.annotate(f'{coldtrap}', xy=(0.5, 0.98), ha='center', va='top', xycoords='axes fraction')
        g = sns.kdeplot(x='ice', y='depth_top', hue='bsed', hue_order=hues, data=df, log_scale=True, 
                    palette=mypal, bw_adjust=1.5, thresh=0, levels=7, common_norm=False, ax=jg.ax_joint)
        # Hist: hue_order plots in reverse order, so flip hues and mypal
        sns.histplot(x='ice', hue='bsed', bins='sturges', common_norm=True,
                    hue_order=hues[::-1], palette=mypal[::-1], alpha=0.3,
                    element='step', data=df, legend=False, ax=jg.ax_marg_x)
        sns.histplot(y='depth', hue='bsed', bins='sturges', common_norm=True,
                    hue_order=hues[::-1], palette=mypal[::-1], alpha=0.3,
                    element='step', data=df, legend=False, ax=jg.ax_marg_y)
        
        # Legend only in 1st subplot
        handles = jg.ax_joint.legend_.get_lines()
        jg.ax_joint.legend_.remove()
        if i == 1:
            # legend_labels = jg.ax_joint.legend_.get_texts()
            title = 'Ballistic \n Sedimentation'
            leg = jg.ax_joint.legend(handles, hues, loc="upper right", 
                                    title=title, fontsize=10, title_fontsize=8, 
                                    bbox_to_anchor=(1, 0.9))
            leg.get_title().set_multialignment('center')

        # Cabeus LCROSS depths and legend
        if coldtrap == 'Cabeus':
            jg.ax_joint.axhline(6, ls=(0, (3, 3, 1, 3)), lw=2, color='k', zorder=10, label='Luchsinger et al. (2021)')
            jg.ax_joint.axhline(10, ls='--', color='k', lw=2, zorder=10, label='Schultz et al. (2010)')
            jg.ax_joint.legend(loc='lower right', title=r'LCROSS Penetration Depth', fontsize=8, title_fontsize=8)
        jgs.append(jg)
    
    # Set axis labels
    for i, jg in enumerate(jgs):
        xlabel = 'Ice layer thickness [m]' if i in (2, 3) else None
        ylabel = 'Depth [m]' if i in (0, 2) else None
        jg.ax_joint.set_xlabel(xlabel)
        jg.ax_joint.set_ylabel(ylabel)
        jg.ax_joint.set_xscale('log')
        jg.ax_joint.set_yscale('log')
        jg.ax_joint.set_xlim(xlim)
        jg.ax_joint.set_ylim(ylim)
        jg.ax_joint.invert_yaxis()
        jg.ax_marg_x.yaxis.get_major_formatter().set_scientific(False)
        jg.ax_marg_y.xaxis.get_major_formatter().set_scientific(False)
        jg.ax_marg_x.xaxis.set_visible(False)
        jg.ax_marg_y.yaxis.set_visible(False)

    # Move all the jointgrids into a single gridspec figure
    gs = mpl.gridspec.GridSpec(2, 2)
    for jg, gs in zip(jgs, gs):
        SeabornFig2Grid(jg, fig, gs)
    fig.tight_layout()

    # version = mplt.plot_version(cfg, loc='lr', xyoff=(0.01, -0.15), ax=fig.gca())
    return _save_or_show(fig, gs, fsave, figdir, '')#version)   


def ejecta_distance(fsave='ejecta_distance.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot ejecta speed, thick, KR, mixing depth as a function of distance.
    
    Oberbeck (1975) Figure 16 shows 4 craters, two < 1 km with no hummocky 
    terrain and larger Mosting C (4.2 km) and Harpalus (41.6 km) have hummocky 
    terrain. Show Ries basin and Meteor crater for comparison.
    """
    mplt.reset_plot_style()

    crater_diams = {
        "Schrödinger": 326e3,  # [m]
        "Harpalus": 41.6e3,  # [m] Harpalus (Oberbeck d)
        "Ries Basin": 24e3,  # [m]
        "Shackleton": 20.9e3,  # [m] smallest polar
        "Mosting C": 4.2e3,  # [m] Mosting C (Oberbeck c)
        "Meteor Crater": 1e3,  # [m]
        # "Oberbeck b": 0.66e3,  # [m]
        # "crater a": 0.56e3,  # [m]
    }
    # ries_max_obs = 36.5e3  # Max observed dist [m]
    ries_max_d = crater_diams['Ries Basin']*2  # [m] 4 crater radii

    # Configure cfg for terrestrial craters
    earth_dict = cfg.to_dict()
    earth_dict['grav_moon'] = 9.81  # [m/s^2]
    earth_dict['rad_moon'] = 6.371e6  # [m]
    earth_dict['target_density'] = 2700  # [kg/m^3]
    earth_cfg = default_config.Cfg(**earth_dict)

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.8), sharex=True, 
                            gridspec_kw={'wspace':0.35, 'hspace':0.02})
    ax1, ax2, ax3, ax4 = axs.flatten()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    d_crad = np.linspace(1, 6, 100)  # x-axis distance [crater radii]
    for i, (crater, diam) in enumerate(crater_diams.items()):
        ls = '-'
        ccfg = cfg
        plot_mixing_depth = True
        if crater in ('Ries Basin', 'Meteor Crater'):  # Terrestrial craters
            ls = '-.'
            ccfg = earth_cfg
            plot_mixing_depth = False

        # Compute speed, ke, thickness, mixing depth
        rad = diam / 2  # radius [m]
        dist = rad * d_crad  # distance [m]
        thick = mp.get_ejecta_thickness(dist, rad, ccfg)  # [m]
        vel = mp.ballistic_velocity(dist, ccfg)  # [m/s]
        mass = thick * ccfg.target_density  # [kg/m^2]
        ke = mp.kinetic_energy(mass, vel) / 1e6  # [MJ] 
        mr = mp.get_mixing_ratio_oberbeck(diam, ccfg)  # mixing ratio
        depth = thick * mr  # depth [m]
        
        # Plot
        label = f'{crater} (D={diam/1e3:.3g} km)'
        ax1.semilogy(d_crad, vel, ls=ls, c=colors[i], label=label)
        ax2.semilogy(d_crad, thick, ls=ls, c=colors[i], label=label)
        ax3.semilogy(d_crad, ke, ls=ls, c=colors[i], label=label)
        if plot_mixing_depth:
            ax4.semilogy(d_crad, depth, ls=ls, c=colors[i], label=label)
        if crater == 'Ries Basin':
            ries = np.argmin(np.abs(dist - ries_max_d))  # Index of Ries max
            rvel = vel[ries]
            rthick = thick[ries]
            rke = ke[ries]
            ax1.axhline(rvel, ls='--', c='k')
            ax1.annotate('Bunte Breccia $v_{max}=$'+f'{rvel:.0f} m/s', (4, rvel), xytext=(3.5, 1300), ha='center', arrowprops=dict(arrowstyle="->"))
            ax2.axhline(rthick, ls='--', c='k')
            ax2.annotate('Bunte Breccia\n$\delta_{min}=$'+f'{rthick:.1f} m', (4, rthick), xytext=(5.8, 30), ha='right', arrowprops=dict(arrowstyle="->"))
            ax3.axhline(rke, ls='--', c='k')
            ax3.annotate('Bunte Breccia   \n$KE_{min}=$'+f'{round(rke,-2):.0f} MJ/m$^2$', (4, rke), xytext=(5.8, 4.5e4), ha='right', arrowprops=dict(arrowstyle="->"))

    ax1.legend(bbox_to_anchor=(0., 1.02, 2.25, .102), loc=3,
            ncol=2, mode="expand", borderaxespad=0.)
    ax1.set_xlim(1, 6)
    ax1.set_ylim(20, 2000)

    ax1.set_ylabel('Ejecta velocity [$\\rm m/s$]')
    ax2.set_ylabel('Ejecta thickness [$\\rm m$]')
    ax3.set_ylabel('Ejecta kinetic energy [$\\rm MJ / m^2$]')
    ax4.set_ylabel('Balistic mixing depth [$\\rm m$]')
    ax3.set_xlabel('Distance [crater radii]')
    ax4.set_xlabel('Distance [crater radii]')
    version = mplt.plot_version(cfg, loc='ll', xyoff=(0.02, 0), ax=ax4)
    return _save_or_show(fig, axs, fsave, figdir, version)


def melt_fraction_bsed(fsave='melt_fraction_bsed.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot melt fraction as a function of ejecta thickness and ballistic mixing depth.
    """
    mplt.reset_plot_style()
    dm = mp.read_ballistic_melt_frac(cfg.bsed_frac_mean_in, cfg)
    ds = mp.read_ballistic_melt_frac(cfg.bsed_frac_std_in, cfg)

    # Get data and axes
    frac_mean = dm.to_numpy()
    frac_std = ds.to_numpy()
    temps = dm.columns.to_numpy()
    mrs = dm.index.to_numpy()
    ejecta_pct = 100*(1 / (1 + mrs))  # target mixing ratio -> ejecta fraction
    extent = [temps.min(), temps.max(), ejecta_pct.min(), ejecta_pct.max()]

    # Get crater and basin melt fractions
    crater_t = cfg.polar_ejecta_temp_init
    basin_tc = cfg.basin_ejecta_temp_init_cold
    basin_tw = cfg.basin_ejecta_temp_init_warm
    df = mp.get_crater_basin_list(cfg)
    dists = mp.get_coldtrap_dists(df, cfg)
    mr = mp.get_mixing_ratio_oberbeck(dists, cfg)
    crater_pct = 100*(1 / (1 + mr[~df.isbasin]))
    basin_pct = 100*(1 / (1 + mr[df.isbasin]))
    crater_pct = crater_pct[~np.isnan(crater_pct)]
    basin_pct = basin_pct[~np.isnan(basin_pct)]
    crater = [[crater_t, crater_t], [crater_pct.min(), crater_pct.max()]]
    basin_c = [[basin_tc, basin_tc], [basin_pct.min(), basin_pct.max()]]
    basin_w = [[basin_tw, basin_tw], [basin_pct.min(), basin_pct.max()]]
    # Plot
    fig = plt.figure(figsize=(7.2, 7))
    ax_dict = fig.subplot_mosaic(
        """
        AB
        CC
        """
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.45)    

    ax = ax_dict['A']
    p = ax.imshow(frac_mean, extent=extent, aspect='auto', interpolation='none', cmap='magma')
    ax.plot(*crater, 'o--', ms=4, c='tab:blue', label='Crater')
    ax.plot(*basin_c, 'o--', ms=4, c='tab:orange', label='Basin (cold)')
    ax.plot(*basin_w, 'o--', ms=4, c='tab:red', label='Basin (warm)')
    ax.legend()
    fig.colorbar(p, ax=ax, label='Fraction melted')
    ax.set_ylabel("Ejecta fraction [%]")
    ax.set_xlabel("Ejecta Temperature [K]")

    ax = ax_dict['B']
    p = ax.imshow(frac_std, vmax=0.1, extent=extent, aspect='auto', interpolation='none', cmap='cividis')
    ax.plot(*crater, 'o--', ms=4, c='tab:blue', label='Crater')
    ax.plot(*basin_c, 'o--', ms=4, c='tab:orange', label='Basin (cold)')
    ax.plot(*basin_w, 'o--', ms=4, c='tab:red', label='Basin (warm)')
    fig.colorbar(p, ax=ax, label='Standard deviation')
    ax.set_ylabel("Ejecta fraction [%]")
    ax.set_xlabel("Ejecta Temperature [K]")

    ax = ax_dict['C']
    axb = ax.twiny()
    ax.errorbar(temps, frac_mean.mean(axis=0), yerr=frac_std.mean(axis=0), 
                c='tab:blue', fmt='o', label='Melt frac vs. Temperature', capsize=4)
    axb.errorbar(ejecta_pct, frac_mean.mean(axis=1), yerr=frac_std.mean(axis=1), 
                c='tab:orange', fmt='^', label='Melt frac vs. Ejecta %', capsize=4)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Ejecta Temperature [K]')
    axb.set_xlabel("Ejecta fraction [%]")
    ax.set_ylabel("Mean fraction melted")
    ax.legend(loc='lower right')
    axb.legend(loc='upper left')
    version = mplt.plot_version(cfg, loc='lr', xyoff=(0.01, -0.21), ax=ax)
    return _save_or_show(fig, ax_dict, fsave, figdir, version)


def random_crater_ages(fsave='random_crater_ages.pdf', figdir=FIGDIR, cfg=CFG):
    """
    Plot random crater ages.
    """
    mplt.reset_plot_style()
    fig, axs = plt.subplots(2, figsize=(7.2, 9))

    df = mp.read_crater_list(cfg)

    # Get random ages the moonpies way (slow)
    ax = axs[0]
    nseed = 100
    cdict = cfg.to_dict()
    ages = np.zeros((nseed, len(df)))
    bins = np.arange(159.5, 420.5) / 100
    crater_list = list(df.cname)
    for i, seed in enumerate(range(nseed)):
        cdict['seed'] = seed
        cfg = default_config.Cfg(**cdict)
        rng = mp.get_rng(cfg)
        df = mp.read_crater_list(cfg)
        df_rand = mp.randomize_crater_ages(df, cfg.timestep, rng).set_index('cname')
        ages[i] = df_rand.loc[crater_list, 'age'] / 1e9

    for i in range(ages.shape[1]):
        if i < 11:
            ls = 'solid'
        elif i < 22:
            ls = 'dashed'
        else:
            ls = 'dotted'
        ax.hist(ages[:, i], bins=bins, label=crater_list[i], histtype='step', ls=ls)
    ax.legend(ncol=4, fontsize=8)
    ax.set_xlim(4.22, ages.min())
    ax.set_ylabel('Count [runs]')
    ax.set_title(f'Random crater ages ({nseed} samples)')

    # Get random ages the scipy way (fast)
    ax = axs[1]
    left, bottom, width, height = [0.65, 0.55, 0.3, 0.3]
    axb = ax.inset_axes([left, bottom, width, height])
    
    nseed = int(1e5)
    sig = df[['age_low', 'age_upp']].mean(axis=1)/2
    a = df.age_low
    b = df.age_upp
    rng = mp.get_rng(cfg)
    S = stats.truncnorm.rvs(-a/sig, b/sig, df.age, sig, (nseed, len(df)), random_state=rng)
    S = mp.round_to_ts(S, cfg.timestep) / 1e9
    bins = np.arange(159.5, 420.5) / 100
    for i in range(S.shape[1]):
        Srow = S[:, i]
        ax.hist(Srow, bins=bins, histtype='step')
        Syoung = Srow < 3
        if Syoung.any():
            axb.hist(Srow[Syoung], bins=bins, histtype='step')

    ax.set_title(f'Random crater ages ({nseed} samples)')
    ax.set_xlabel('Age [Ga]')
    axb.set_xlabel('Age [Ga]')
    ax.set_xlim(4.22, 2.89)
    axb.set_xlim(2.2, 1.6)
    ax.set_ylabel('Count [runs]')
    version = mplt.plot_version(cfg, loc='lr', xyoff=(0.01, -0.17), ax=axs[1])
    return _save_or_show(fig, axs, fsave, figdir, version)


# Class helpers
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = mpl.gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = mpl.gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
        
if __name__ == '__main__':
    _generate_all()
