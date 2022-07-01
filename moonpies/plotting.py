# Plot helpers
import json
import warnings
from pathlib import Path
import matplotlib as mpl
# import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moonpies import default_config
from moonpies import moonpies as mp

CFG = default_config.Cfg(seed=1)
CDF = mp.get_crater_basin_list(CFG)  # Only used for radii for lith_key
WARNINGS = 0  # Count warnings

# Stratigraphy column hatches with increasing density
HATCH = ('/', '\\', '|', '-', '+', 'x', 'o')
HATCHES = [h*i for h in HATCH for i in range(1, 6)] # 28 unique hatches
CMAP_COLORS = ('#D3D3D3', '#1f77b4')

# Plot helpers
def reset_plot_style(mplstyle=True, cfg=CFG):
    """Reset matplotlib style defaults, use MoonPIES mplstyle if True."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    if mplstyle:
        if mplstyle is True:
            mplstyle = Path(cfg.model_path) / ".moonpies.mplstyle"
        try:
            mpl.style.use(mplstyle)
        except (OSError, FileNotFoundError):
            global WARNINGS
            if WARNINGS < 1:
                warnings.warn(f"Could not find mplstyle file {mplstyle}")
                WARNINGS += 1


def plot_version(
    cfg=CFG, loc="ll", xy=None, xyoff=None, ax=None, bbkw=None, **kwargs
):
    """Add MoonPIES version label."""
    x, y = (0, 0) if xy is None else xy
    xoff, yoff = (0, 0) if xyoff is None else xyoff
    ax = mpl.pyplot.gca() if ax is None else ax
    bbkw = {} if bbkw is None else bbkw
    # Get position of version label
    if loc[0] == "l":  # lower
        y += 0.035 + yoff
        va = "bottom"
    elif loc[0] == "u":  # upper
        y += 1 - 0.035 + yoff
        va = "top"
    if loc[1] == "l":  # left
        x += 0.02 + xoff
        ha = "left"
    elif loc[1] == "r":  # right
        x += 1 - 0.02 + xoff
        ha = "right"
    version = f"v{cfg.version}"
    msg = f"MoonPIES {version}"
    xy = (x, y)
    kwargs = {"ha": ha, "va": va, "xycoords": "axes fraction", **kwargs}
    bb = {"boxstyle": "round", "fc": "w", **bbkw}
    ax.annotate(msg, xy, bbox=bb, **kwargs)
    return version


def plot_stratigraphy(out_path, coldtraps=None, seeds=None, fsave='', min_thick=1,
                      lith_depth_labels=False, lith_time_labels=False, 
                      hatchmin=0, cmap=None, **kwargs):
    """Plot stratigraphy."""
    reset_plot_style()

    # Get defaults
    out_path = Path(out_path)
    if not out_path.exists():
        raise FileNotFoundError(f"{out_path} does not exist.")
    fcfg = next(out_path.glob('run_config*.py'))
    cfg = default_config.read_custom_cfg(fcfg)
    
    if seeds is None:
        seeds = [cfg.seed]
    if coldtraps is None:
        coldtraps = cfg.coldtrap_names
    cmap = make_cmap(CMAP_COLORS) if cmap is None else plt.get_cmap(cmap)
    
    # Init plot
    cwidth = 1.4 
    wspace = 0.1
    if lith_depth_labels or lith_time_labels:
        cwidth = 2
        wspace = 0.5
    ncols = len(seeds) * len(coldtraps) + (len(seeds) - 1)  # Add space b/t each set of seeds
    fig = plt.figure(figsize=(cwidth*ncols, 8))
    gs = fig.add_gridspec(2, ncols, wspace=wspace, hspace=0.05, height_ratios=[1, 0.05])
    axs = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    _ = [ax.axis('off') for ax in axs[len(coldtraps)::len(coldtraps)+1]]  # Cancel axes spaces between seeds

    cbar_ax = fig.add_subplot(gs[1, :])
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    maxdepth = 0
    for s, seed in enumerate(seeds):
        out_path = out_path.parent / f'{seed:05d}'
        for i, coi in enumerate(coldtraps):
            scsv = out_path / f'strat_{coi}.csv'
            try:
                sdf = pd.read_csv(scsv)
            except FileNotFoundError:
                print(f"Could not find stratigraphy file {scsv}")
                continue
            formation_row = sdf[sdf.label == "Formation age"]
            formation_age = formation_row.iloc[0].time
            sdf = sdf.drop(formation_row.index)

            # Make strat col and plots
            sdf = clean_up_strat_col(sdf, min_thick, hatchmin)
            strat = get_strat_col_ranges(sdf)
            if strat.depth.max() > maxdepth:
                maxdepth = strat.depth.max()

            # Plot strat layers
            ax1 = axs[s*len(coldtraps) + i + s]  # +s for space b/t sets of seeds
            for si, row in strat.iterrows():
                color = cmap(norm(row.icepct))
                hatch = row.hatch if hatchmin > 0 else None
                label = row.hatchlabel if hatchmin > 0 else None
                ax1.fill_betweenx(strat.depth, 0, 1, ec=None, 
                                where=(strat.hatch==row.hatch), 
                                facecolor=color, hatch=hatch, label=label)

            # Configure axes (ax1 = absolute depth [m], ax2 = lith depth [m])
            ax1.set_title(f"{coi}\n{formation_age/1e9:.2f} Ga")
            ax1.grid(color='k', lw=1)
            ax1.tick_params(axis='y', left=False, right=False)

            # Label lines between strat layers
            if lith_depth_labels or lith_time_labels or hatchmin > 0:
                ax1.grid(False)
                ax1.tick_params(axis='y', which='both', length=0)
                ax2 = ax1.twinx()
                ax1.get_shared_y_axes().join(ax1, ax2)  # Sync ylimits
                depths = strat.depth.values
                depths = np.insert(depths, 0, 0)  #Add zero to start / top
                ax2.set_yticks(depths)
                ax2.grid(which="major", color="black", linestyle="-", linewidth=2)
                if lith_depth_labels:
                    yticklabels = [f'{d:.1f}' for d in depths]
                    label = 'Layer Depth [m]'
                elif lith_time_labels:
                    ax1.grid(True, alpha=0.5)
                    times = strat.time.values / 1e9
                    times = np.insert(times, 0, 0)  #Add zero to start / top
                    yticklabels = [f'{t:.3g}' for t in times]
                    label = 'Layer Age [Ga]'
                else:  # draw lines bewteen hatched layers but no labels
                    ax1.grid(True, alpha=0.5)
                    yticklabels = []
                    label = None
                ax2.set_yticklabels(yticklabels)
                if i == len(coldtraps) - 1:
                    ax2.set_ylabel(label, labelpad=10, rotation=-90)

    handles, labels = [], []
    for i, ax in enumerate(axs):
        ax.set_xlim(0, 1)
        ax.set_ylim(maxdepth, 0)
        ax.tick_params(axis='y', width=3)
        ax.xaxis.set_visible(False)
        ax.spines["top"].set_position(("axes", 1.0))
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
            if i > 0:
                ax.set_yticklabels([])
        _handles, _labels = ax.get_legend_handles_labels()
        handles += _handles
        labels += _labels

    
    # Set y-axis
    axs[0].tick_params(axis='y', length=8, width=2, left=True, direction='inout')
    axs[0].set_ylabel('Depth [m]', labelpad=5)
    axs[0].yaxis.get_major_formatter().set_scientific(False)
    
    # Make colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Ice in Layer [%]')

    # Make legend
    if hatchmin > 0:
        uniq_labels = sorted(list(set(labels) - {'Ice', 'Other'}))
        if 'Ice' in labels:
            uniq_labels = ['Ice'] + uniq_labels
        if 'Other' in labels:
            uniq_labels.append('Other')
        uniq_handles = []
        for label in uniq_labels:
            handle = handles[labels.index(label)]
            handle.set_facecolor(cmap(norm(0)))
            if label == 'Ice':
                handle.set_facecolor(cmap(norm(100)))
            uniq_handles.append(handle)
        cbar_ax.legend(uniq_handles, uniq_labels, loc='upper left', ncol=ncols//2, mode='expand',
                   handleheight=2, handlelength=4, fontsize=11, bbox_to_anchor=(0, -2, 1, 0.4))

    # fig.suptitle(f'Stratigraphy for: {out_path}')
    version = plot_version(cfg, loc='lr', xyoff=(0.01, -0.19), ax=axs[-1])
    if fsave:
        fsave = Path(fsave)
        fig.savefig(fsave.with_name(f'{fsave.stem}_{version}{fsave.suffix}'))
    return fig
    
# Plot stratigraphy helpers
def get_lith_key(liths, hatches=HATCHES):
    """Return dict of dict of lithology to label, hatch style and ice_pct"""
    # Hash magic ensures same lithology gets same hatch
    # If there are more liths than hatches, there will be repeats
    lith_key = {lith: hatches[hash(lith)%len(hatches)] for lith in liths}
    lith_key['Ice'] = '..'
    lith_key['Other'] = ''
    return lith_key

def clean_up_strat_col(strat_col, min_thick=0, hatchmin=1):
    """
    Clean up strat_col by removing thin layers 
    """
    thick = -np.diff(np.append(strat_col.depth.values, 0))
    
    # Merge thin layers as 'Other'
    other = (thick < hatchmin) & (strat_col.label != 'Ice')
    strat_col.loc[other, 'label'] = 'Other'
    
    agg = {"ice": "sum", "ejecta": "sum", "label": "last", "time": "last", "depth": "last"}
    strat = mp.merge_adjacent_strata(strat_col, agg)
    strat["icepct"] = np.round(100 * strat.ice / (strat.ice + strat.ejecta), 4)

    # Drop layers with thinner min_thick
    thick = -np.diff(np.append(strat.depth.values, 0))
    strat = strat.drop(strat[thick < min_thick].index)
    return strat


def get_strat_col_ranges(strat_col, cdf=CDF, savefile=None):
    """Return strat_col ranges for strat_col plot"""
    if len(strat_col) == 0:
        return strat_col
    # Make into ranges
    strat = pd.concat((strat_col.copy(), strat_col.copy()))
    strat = strat.sort_values('depth')
    top = pd.DataFrame([np.zeros(len(strat.columns))], columns=strat.columns)
    strat = pd.concat((top, strat)).reset_index(drop=True)
    labels = pd.concat((strat.label.iloc[1:], strat.label.iloc[-1:])).values
    strat['label'] = labels
    
    # Get hatch corresponding to each ejecta source from lith key
    liths = [lith for label in labels for lith in label.split(',')]
    lith_key = get_lith_key(liths)
    strat['hatch'] = ''
    strat['hatchlabel'] = ''  # Actual label hatch is based on
    for si, label in strat.label.iteritems():
        # Split multiple sources into individual, but only need one hatch
        # Get lith from crater with largest diameter
        liths = label.split(',')
        if len(liths) > 1:
            label = cdf.set_index('cname').loc[liths].sort_values('diam').iloc[-1].name
        strat.loc[si, 'hatch'] = lith_key[label]
        strat.loc[si, 'hatchlabel'] = label
    if savefile:
        strat.to_csv('strat_output.csv')
    return strat.reset_index(drop=True)

def make_cmap(hex_list, locs=None):
    """
    Returns color map from hex_list, interpolating linearly if locs=None.

    If locs is provided, map each color to its location in locs.
    
    Modified from Kerry Halupka:
    github.com/KerryHalupka/custom_colormap/blob/master/generate_colormap.py
    
    Parameters
    ----------
    hex_list (list): Hex code strings
    locs (list): Locations in [0, 1]. Must start with 0 and end with 1.
    
    Returns
    ----------
    cmap (matplotlib.colors.LinearSegmentedColormap): Color map
    """
    def hex_to_rgb(value):
        """Return rgb from hex """
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv//3], 16) for i in range(0, lv, lv//3))

    def rgb_to_dec(value):
        """Return decimal color from RGB (i.e. divide by 256)."""
        return [v / 256 for v in value]

    rgbs = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if locs is None:
        locs = list(np.linspace(0, 1, len(rgbs)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        c = [[loc, rgbs[i][num], rgbs[i][num]] for i, loc in enumerate(locs)]
        cdict[col] = c
    cmap = mpl.colors.LinearSegmentedColormap('ice', segmentdata=cdict, N=256)
    return cmap
