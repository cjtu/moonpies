"""
Stratigraphy column plots
A. Madera
Last updated: 11/29/2021 (CJTU)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from moonpies import moonpies as mp
from moonpies import default_config

FIGDIR = ''  # Full path or '' for default (moonpies/figs)
DATADIR = ''  # Full path or '' for default (moonpies/data)
SEED = '65106'  # Set seed as string or '' to pick first seed from datadir
COI = 'Faustini'  # Set coldtrap of interest or '' for default ("Haworth")
RUNNAME = ''  # append runname to filename
MIN_THICKNESS = 10  # [m] minimum layer thickness

# Set default config
CFG = default_config.Cfg()

# Set Fig paths
if not FIGDIR:
    FIGDIR = CFG.figspath
FIGDIR = Path(FIGDIR).resolve()

# Set data path
if not DATADIR:
    DATADIR = Path(CFG.outpath).parents[0]
DATADIR = Path(DATADIR)

# Set output path
if not SEED:
    SEED = next(DATADIR.iterdir()).stem
OUTDIR = DATADIR / SEED

# Set stratigraphy input csv
if not COI:
    COI = 'Haworth'
COI_CSV = OUTDIR / f'strat_{COI}.csv'

# Set output paths
OUT_STRAT = FIGDIR / f'strat_{COI}_{SEED}{"_"+RUNNAME}.png'
OUT_KEY = FIGDIR / f'strat_{COI}_{SEED}_key{"_"+RUNNAME}.png'

plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['figure.titlesize'] = 20


def get_continuous_cmap(hex_list, locs=None):
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
    cmp (matplotlib.colors.LinearSegmentedColormap): Color map
    """
    def hex_to_rgb(value):
        """Return rgb from hex """
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv//3], 16) for i in range(0, lv, lv//3))

    def rgb_to_dec(value):
        """Return decimal color from RGB (i.e. divide by 256)."""
        return [v/256 for v in value]

    rgbs = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if locs is None:
        locs = list(np.linspace(0, 1, len(rgbs)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        c = [[loc, rgbs[i][num], rgbs[i][num]] for i, loc in enumerate(locs)]
        cdict[col] = c
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

# Make lithology style dictionary
hatches = [
    "/", "O", "\\", "+", 
    "//", "||", "-\\", "-", 
    "\\\\", "x", "O|", "\\|"
    "o","|","++","OO",
    "xx","..","--","xx","O."]

# Make diverging ice% colormap (gray, white, blue)
colors = ['#8b8c8d', '#ffffff', '#3c8df0']
ICE_CM = get_continuous_cmap(colors)


def get_lith_key(lithology, cmap=ICE_CM):
    """Return dict of dict of lithology to label, hatch style and color"""
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    ice_key = len(lithology) + 1
    ice_c = cmap(norm(100))
    lith_key = {
        ice_key: {"lith": "Ice", "lith_num": -2, "hatch": ".", "color": ice_c}
    }
    for i, label in enumerate(lithology[lithology != "Ice"]):
        ice_pct = coi_strat.iloc[i].icepct
        color = cmap(norm(ice_pct))
        lith_key[i] = {'lith': label, 'hatch': hatches[i % len(hatches)], 
                        'color':color, 'ice_pct': ice_pct}
    return lith_key


def makekey(lith_key, savepath):
    """Plot lithology legend"""
    x = [0, 1]
    y = [1, 0]

    ncols = 3
    nrows = int(np.ceil(len(lith_key.keys())/4))
    xsize = 9
    ysize = nrows * 1.5
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True, 
                             figsize=(xsize, ysize), 
                             subplot_kw={'xticks':[], 'yticks':[]})

    for ax, (_, v) in zip(axes.flat, lith_key.items()):
        title = v["lith"]
        if "Ice" == title:
            pass
        else:
            title += f" Ejecta ({v['ice_pct']:.0f}% ice)"
        ax.plot(x, y, linewidth=0)
        ax.fill_betweenx(y, 0, 1, facecolor=v["color"], hatch=v["hatch"])
        ax.set_xlim(0, 0.1)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=12)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches='tight', dpi=300)
    print(f"Saved key to {savepath}")


#Setting up definitions for strat columns
def clean_up_strat_col(strat_col, thick_thresh=MIN_THICKNESS):
    """
    Clean up strat_col by removing thin layers 
    """
    thick = -np.diff(np.append(strat_col.depth.values, 0))
    strat_col = strat_col.drop(strat_col[thick < thick_thresh].index)
    agg = {"ice": "sum", "ejecta": "sum", "label": "last", "time": "last", "depth": "last"}
    strat = mp.merge_adjacent_strata(strat_col, agg)
    strat["icepct"] = np.round(100 * strat.ice / (strat.ice + strat.ejecta), 4)
    return strat


def get_strat_col_ranges(strat_col, savefile=None):
    """Return strat_col ranges for strat_col plot"""
    # Make into ranges
    strat = pd.concat((strat_col.copy(), strat_col.copy()))
    strat = strat.sort_values('depth')
    strat = strat.iloc[:-1]
    top = strat.iloc[0:1]
    top.loc[:, ['depth', 'ejecta', 'ice', 'time']] = (0,0,0,0)
    strat = pd.concat((top, strat))
    strat['label'] = pd.concat((strat.iloc[1:]['label'], strat.iloc[-1:]['label'])).values

    if savefile:
        strat.to_csv('strat_output.csv')
    return strat


def makeplot(strat, savepath, coi=COI, cmap=ICE_CM):
    """Plot strat columns"""
    # Get the depth boundaries of each distinct layer in strat
    adj_check = (strat.label != strat.label.shift()).cumsum()
    distinct_layers = strat.groupby(['label', adj_check], as_index=False,
                                sort=False).agg({"depth" : ['max'], "time": ['max']})
    yticks_depth = distinct_layers.depth['max'].values
    yticks_depth = np.insert(yticks_depth, 0, 0)  #Add zero to start

    yticks_time = distinct_layers.time['max'].values
    yticks_time = np.insert(yticks_time, 0, 0)

    top_depth = 0
    bot_depth = yticks_depth.max()

    # Init strat col plot
    fig, (ax1, cax) = plt.subplots(2, figsize=(4.5, 10),
                                  gridspec_kw={"height_ratios":[1, 0.05]})
    fig.suptitle(f"{coi} Stratigraphy")
    ax1.set_xlim(0, 1)
    ax2 = ax1.twinx()

    # Plot strat layers
    for k, v in lith_key.items():
        ax1.fill_betweenx(strat["depth"], 0, 1, where=(strat["lith"] == k), 
                          facecolor=v["color"], hatch=v["hatch"])

    # Configure axes (ax1 = absolute depth [m], ax2 = lith depth [m])
    ax1.set_ylabel('Depth [m]', labelpad=10)
    ax1.grid(False)
    ax1.tick_params(axis='y', width=3)
    
    ax2.set_ylabel('Lithology Depths [m]', labelpad=25, rotation=-90)
    ax2.set_ylim(bot_depth, top_depth)
    ax2.set_yticks(yticks_depth)
    ax2.grid(which="major", color="black", linestyle="-", linewidth=2)

    for ax in [ax1, ax2]:
        ax.set_ylim(bot_depth, top_depth)
        ax.tick_params(axis='y', width=3)
        ax.xaxis.set_visible(False)
        ax.spines["top"].set_position(("axes", 1.0))
        for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
    
    # Make colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cax, orientation='horizontal', label='Ice in Layer [%]')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01)
    fig.savefig(savepath, bbox_inches="tight")
    print(f"Saved strat col plot to {savepath}")

if __name__ == "__main__":
    # Read strat and initialize
    coi_strat = pd.read_csv(COI_CSV)
    cfg = default_config.Cfg(seed=float(SEED))
    crater_list = mp.get_crater_list(cfg=cfg)

    # Truncate strat to its formation age
    formation_age = crater_list[crater_list.cname == COI].age.values[0]
    coi_strat = coi_strat[coi_strat.time < formation_age]

    #Make strat col and plots
    coi_strat = clean_up_strat_col(coi_strat)
    strat = get_strat_col_ranges(coi_strat)

    #Get lith key and dict of lithology:numerical key
    lith_key = get_lith_key(coi_strat.sort_values('depth').label.unique())
    lith2key = {v["lith"]: k for k, v in lith_key.items()}
    strat["lith"] = strat["label"].map(lith2key)

    makeplot(strat, OUT_STRAT, COI, ICE_CM)
    makekey(lith_key, OUT_KEY)
