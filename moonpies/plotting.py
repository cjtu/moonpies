# Plot helpers
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moonpies import config
from moonpies import moonpies as mp

CFG = config.Cfg(seed=1)
CDF = mp.get_crater_basin_list(CFG)  # Only used for radii for lith_key

# Stratigraphy column hatches with increasing density
HATCH = ('/', '\\', '|', '-', '+', 'x', 'o')
HATCHES = [h*i for h in HATCH for i in range(1, 6)] # 28 unique hatches
CMAP_COLORS = ('#D3D3D3', '#1f77b4')

# Plot helpers
def reset_plot_style(mplstyle=True, cfg=CFG):
    """Reset matplotlib style defaults, use MoonPIES mplstyle if True."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    if mplstyle:
        mpl.style.use(cfg.mplstyle_in)


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


def plot_stratigraphy(out_path, coldtraps=None, runs=None, seeds=None, 
                      min_thick=1, fsave='', cmap=None, version=True, 
                      strat_kws={}, kwargs={}):
    """Plot stratigraphy."""
    reset_plot_style()

    # Get defaults
    try:
        fcfg = next(Path(out_path).rglob('run_config*.py'))
    except StopIteration as e:
        raise FileNotFoundError(f"No run_config*.py in {out_path}") from e
    cfg = config.read_custom_cfg(fcfg)
    date_path = Path(cfg.out_path).parents[1]
    
    if runs is None:
        runs = [cfg.run_name]
    if seeds is None:
        seeds = [cfg.seed]
    if coldtraps is None:
        coldtraps = cfg.coldtrap_names
    cmap = get_cmap(cmap)
    
    # Init plot
    cwidth = 1.4 
    wspace = 0.1
    if strat_kws.get('label_depths') or strat_kws.get('label_times'):
        cwidth = 2
        wspace = 0.5
    nruns = len(runs)
    ncts = len(coldtraps)
    ncols = len(seeds) * len(coldtraps) + (len(seeds) - 1)  # Add space b/t each set of seeds
    fig = plt.figure(figsize=(cwidth*ncols, 6*nruns+2))
    gs = fig.add_gridspec(nruns+1, ncols, wspace=wspace, hspace=0.05, 
                          height_ratios=[1]*nruns + [0.05])
    # axs = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    # _ = [ax.axis('off') for ax in axs[len(coldtraps)::len(coldtraps)+1]]  

    cbar_ax = fig.add_subplot(gs[nruns, :])
    maxdepth = 0
    handles, labels = [], []
    for r, run in enumerate(runs):
        rpath = date_path / run
        axs = [fig.add_subplot(gs[r, i]) for i in range(ncols)]
        # Leave blank axis between each set of seeds
        _ = [ax.axis('off') for ax in axs[ncts::ncts+1]]
        for s, seed in enumerate(seeds):
            spath = rpath / f'{seed:05d}'
            for i, coi in enumerate(coldtraps):
                scsv = spath / f'strat_{coi}.csv'
                try:
                    sdf = pd.read_csv(scsv)
                except FileNotFoundError:
                    print(f"Could not find stratigraphy file {scsv}")
                    continue
                formation_row = sdf[sdf.label == "Formation age"]
                formation_age = formation_row.iloc[0].time
                sdf = sdf.drop(formation_row.index)

                # Make strat col and plots
                sdf = clean_up_strat_col(sdf, min_thick)
                strat = get_strat_col_ranges(sdf)
                if strat.depth.max() > maxdepth:
                    maxdepth = strat.depth.max()

                # Plot strat layers
                title = ''
                if r == 0:
                    title = f"{coi}\n{formation_age/1e9:.2f} Ga"
                ax = axs[s*len(coldtraps) + i + s]  # +s adds space b/t seeds
                ylabel_right = i == len(coldtraps) - 1  # Last coldtrap
                ax = plot_strat(strat, title, ylabel_right=ylabel_right, 
                                colorbar=False, cmap=cmap, legend=False,
                                **strat_kws, ax=ax)

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
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Ice in Layer [%]')

    # Make legend
    strat_legend(handles, labels, cmap, ncols//2, cbar_ax)
    
    # fig.suptitle(f'Stratigraphy for: {out_path}')
    if version:
        version = plot_version(cfg, loc='lr', xyoff=(0.01, -0.19), ax=axs[-1])
    if fsave:
        fsave = Path(fsave)
        fig.savefig(fsave.with_name(f'{fsave.stem}_{version}{fsave.suffix}'))
    return fig


# Plot stratigraphy helpers
def plot_strat(strat, title='', colorbar=True, cmap=None, label_layers='', 
               ylabel_right=False, legend=True, ax=None):
    """Plot a single stratigraphy column."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.4, 8))
    cmap = get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    for _, row in strat.iterrows():
        color = cmap(norm(row.icepct))
        ax.fill_betweenx(strat.depth, 0, 1, ec=None, 
                        where=(strat.hatch==row.hatch), facecolor=color, 
                        hatch=row.hatch, label=row.hatchlabel)

    # Configure axes (ax1=absolute depth [m], ax2=lith depth [m])
    ax.set_title(title)
 
    # Label lines between strat layers
    ax.grid(True, color='k', lw=1, alpha=0.5)
    ax.tick_params(axis='y', which='both', length=0, left=False, right=False)
    ax2 = ax.twinx()
    ax.get_shared_y_axes().join(ax, ax2)  # Sync ylimits
    depths = np.r_[0, strat.depth.values]  # Add zero to start/top
    ax2.set_yticks(depths)
    ax2.grid(which="major", color="black", linestyle="-", linewidth=2)
    if label_layers == 'depth':
        yticklabels = [f'{d:.1f}' for d in depths]
        label = 'Layer Depth [m]'
    elif label_layers == 'time':
        times = np.r_[0, strat.time.values / 1e9]  # Add zero to start/top
        yticklabels = [f'{t:.3g}' for t in times]
        label = 'Layer Age [Ga]'
    else:  # draw lines bewteen hatched layers but no labels
        yticklabels = []
        label = None
    ax2.set_yticklabels(yticklabels)
    if ylabel_right:
        ax2.set_ylabel(label, labelpad=10, rotation=-90)
    if colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, orientation='horizontal', label='Ice in Layer [%]')
    if legend:
        hl = ax.get_legend_handles_labels()
        strat_legend(*hl, cmap, 1, ax=ax)
    return ax


def strat_legend(handles=None, labels=None, cmap=None, ncol=1, ax=None):
    """Make a legend for strat layers."""
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    if handles is None or labels is None:
        handles, labels = plt.gca().get_legend_handles_labels()
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
    ax.legend(uniq_handles, uniq_labels, loc='upper left', ncol=ncol, mode='expand',
                   handleheight=2, handlelength=4, fontsize=11, bbox_to_anchor=(0, -2, 1, 0.4))
    return ax


def get_lith_key(liths, hatches=HATCHES):
    """Return dict of dict of lithology to label, hatch style and ice_pct"""
    # Hash magic ensures same lithology gets same hatch
    # If there are more liths than hatches, there will be repeats
    lith_key = {lith: hatches[hash(lith)%len(hatches)] for lith in liths}
    lith_key['Ice'] = '..'
    lith_key['Other'] = ''  # clean_up_strat_col should no longer return other
    return lith_key


def clean_up_strat_col(strat, min_thick=0):
    """
    Clean up strat_col by removing thin layers 
    """
    if strat.empty:
        strat['icepct'] = None
        return strat
    agg = {"ice": "sum", "ejecta": "sum", "label": "last", "time": "last", 
           "depth": "last"}
    thickness = -np.diff(np.r_[strat.depth, 0])  # Need 0 at top of col for diff

    # Set all "thin" layers to Other and merge them if adjacent
    strat.loc[thickness < min_thick, 'label'] = 'Other' 
    strat = mp.merge_adjacent_strata(strat, agg)

    # Find all remaining Other layers and merge them into next lower layer
    others = strat[strat.label == 'Other']
    newlabel = []
    if strat.loc[0, 'label'] == 'Other':
        others = others.drop(0, axis=0, errors='ignore')
        newlabel.append(strat.loc[1, 'label'])  # handle bottom later
    newlabel = newlabel.extend([strat.label.loc[i-1] for i in others.index])
    strat.loc[strat.label == 'Other', 'label'] = newlabel
    strat = mp.merge_adjacent_strata(strat, agg)

    # Compute ice %
    strat["icepct"] = 100 * strat.ice / (strat.ice + strat.ejecta)
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


# Custom colormap helper
def get_cmap(cmap=None):
    """Return matplotlib colormap or custom cmap from list of hex colors."""
    if cmap is None:
        return make_cmap(CMAP_COLORS)
    elif isinstance(cmap, str) or isinstance(cmap, mpl.colors.Colormap):
        return plt.get_cmap(cmap)
    else:
        return make_cmap(cmap)


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
