"""Make strat column figure."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from moonpies import moonpies as mp
from moonpies import default_config


# Make lithology style dict
hatches = [ '-\\', '||', '\\|', '//','--','++', 'x','O','.','/','\\','|',
            '-','+', 'O|','O.','\\\\','xx','OO','..','**','*', '|*','x*','*-']

# Make ice% colormap (first color is gray, last is blue)
# colors = np.array([(60, 60, 60), (173, 216, 230)]) / 255
# ICE_CM = mpl.colors.LinearSegmentedColormap.from_list("Custom", colors, N=100)
ICE_CM = mpl.colors.viridis

def get_lith_key(strat_df, cmap=ICE_CM):
    strat_df = strat_df.drop_duplicates('label')
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    lith_key = {
        len(strat_df) + 1: {
            'lith': 'Ice', 
            'lith_num': -2, 
            'hatch': 'o', 
            'color': cmap(norm(100))},
    }
    for i, row in strat_df.iterrows():
        if row.label == 'Ice':
            continue
        # Set up strat colors
        color = cmap(norm(row.ice_pct))
        lith_key[i] = {'lith': row.label, 'hatch': hatches[i], 'color': color}
    return lith_key


# Functions
def get_strat(ice_df, ej_df, coldtrap='Haworth', thresh=1e-6):
    """Return strat column df of coldtrap from ice_df, ej_df."""
    # Get columns of strat df from ice_df and ej_df
    strat = ice_df[['time']].copy()
    strat['ice'] = ice_df[coldtrap]
    strat['ejecta'] = ej_df[coldtrap]
    
    # Zero out all thicknesses below thresh
    strat[strat < thresh] = 0

    # Get indices where ice and ejecta exist
    iice = strat.ice > 0
    iej = strat.ejecta > 0
    iboth = iice & iej


    # Get total thickness - cumulative sum of ice and ejecta depths
    strat['depth'] = strat.ice.cumsum() + strat.ejecta.cumsum()
    strat['depth'] = strat.depth.max() - strat.depth

    # Get ice % where ice and ejecta exist (ice / ejecta)
    strat['ice_pct'] = 0
    strat.loc[strat.ice > 0, 'ice_pct'] = 100
    strat.loc[iboth, 'ice_pct'] = 100 * strat.ice[iboth] / strat.ejecta[iboth]

    # Label rows
    strat['label'] = ''
    strat.loc[iej, 'label'] = ej_df[iej].ejecta_source.fillna('')
    strat.loc[iice & ~iej, 'label'] = 'Ice'
    
    # Drop rows with no ice or ejecta
    strat = strat[iice | iej] 

    # Combine adjacent rows with same label into layers
    adj_check = (strat.label != strat.label.shift()).cumsum()
    strat = strat.groupby([
        'label', adj_check], as_index=False, sort=False).agg({
                                    'label': 'last',
                                    'time': 'last',
                                    'depth': 'last',
                                    'ice_pct': 'mean',
                                    'ice': 'sum',
                                    'ejecta': 'sum'})
    strat.to_csv('strat_output.csv')
    return strat


def get_strat_layers(strat_df, timestart=4.25e9):
    """Return strat_df as unique layers (range start to end) for plotting."""
    strat_df = strat_df.copy()
    first_depth = np.sum(strat_df[['depth', 'ice', 'ejecta']].iloc[0])
    d_top = np.insert(strat_df.iloc[:-1].depth.values, 0, first_depth)
    d_bot = strat_df.iloc[:].depth.values
    t_top = np.insert(strat_df.iloc[:-1].time.values, 0, timestart)
    t_bot = strat_df.iloc[:].time.values
    strat_top = strat_df.set_index(np.arange(0, 2*len(strat_df), 2))
    strat_top['depth'] = d_top
    strat_top['time'] = t_top
    strat_bot = strat_df.copy().set_index(np.arange(1, 2*len(strat_df)+1, 2))
    strat_bot['depth'] = d_bot
    strat_bot['time'] = t_bot
    strat_layers = pd.concat((strat_top, strat_bot)).sort_index()
    strat_layers.to_csv('strat_layers.csv')
    return strat_layers


def makekey(lith_key, coldtrap, savepath, ncols=2):
    """Plot lithology key."""
    # The x and y axes of the patch figs
    x = [0, 1]
    y = [1, 0]

    # Set up axes based on number of keys and specified ncols
    nrows = int(np.ceil(len(lith_key.keys()) / ncols))
    xsize = ncols * 3
    ysize = nrows * 1.5
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, 
                             sharey=True, figsize=(xsize, ysize),
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(f'{coldtrap} Lithology Key')
    for ax, (_, v) in zip(axes.flat, lith_key.items()):
        title = v['lith']
        if 'Ice' not in title:
            title += ' Ejecta'
        ax.plot(x, y, linewidth=0)
        ax.fill_betweenx(y, 0, 1, facecolor=v['color'], hatch=v['hatch']
        )
        ax.set_xlim(0, 0.1)
        ax.set_ylim(0, 1)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight', dpi=300)


def makeplot(strat, coldtrap, savepath):
    """Plot strat column."""  
    # Set up the plot axes
    fig, ax = plt.subplots(figsize=(16, 16))
    ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
    ax2 = ax1.twiny()
    ax3 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1)

    # As our curve scales will be detached from the top of the track,
    # this code adds the top border back in without dealing with splines
    ax10 = ax1.twiny()
    ax10.xaxis.set_visible(False)
    ax11 = ax3.twiny()
    ax11.xaxis.set_visible(False)
    ax13 = ax4.twiny()
    ax13.xaxis.set_visible(False)

    # Ballistic Speed Track
    # ax1.plot(strat['ballistic_speed'], strat['depth'], color = 'green', linewidth = 1.5)
    # ax1.set_xlabel('Balistic Speed (m/s)', fontsize=15)
    # ax1.xaxis.label.set_color('green')
    # ax1.set_xlim(0, 900)
    # ax1.set_ylabel('Depth (m)', fontsize = 15)
    # ax1.tick_params(axis='x', colors='green', labelsize=10)
    # ax1.tick_params(axis='y', labelsize=15)
    # ax1.spines['top'].set_edgecolor('green')
    # ax1.title.set_color('green')
    # ax1.set_xticks([0, 300, 600, 900])

    # Kinetic Energy overlaid Ballistic Speed Track
    # ax2.plot(strat['kinetic_e_km'], strat['depth'], color = 'red', linewidth = 1.5)
    # ax2.set_xlabel('Kinetic Energy of Ejecta (J/km^2)', fontsize=15)
    # ax2.xaxis.label.set_color('red')
    # ax2.set_xlim(15000, 160000)
    # ax2.tick_params(axis='x', colors='red', labelsize=10)
    # ax2.spines['top'].set_position(('axes', 1.08))
    # ax2.spines['top'].set_visible(True)
    # ax2.spines['top'].set_edgecolor('red')
    # ax2.set_xticks([15000, 90000, 160000])

    # lith track
    ax3.plot(strat.lith, strat.depth, color='black', linewidth=0.5)
    ax3.set_xlabel(f'{coldtrap} Stratigraphy Column', fontsize=18)
    ax3.set_xlim(0, 1)
    ax3.xaxis.label.set_color('black')
    ax3.tick_params(axis='x', colors='black', labelsize=10)
    ax3.spines['top'].set_edgecolor('black')
    ax3.set_yticks(strat.depth.values)
    for k, v in lith_key.items():
        ax3.fill_betweenx(strat.depth, 0, strat.lith,
                          where=(strat.lith == k),
                          facecolor=v['color'], hatch=v['hatch'])
    ax3.set_ylabel('Depth [m]', fontsize=15)
    # Set time array on middle plot
    ax3.set_xticks([])
    ax3b = ax3.twinx()
    ax3b.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax3b.set_yticklabels(strat.time.values / 1e9)
    ax3b.set_ylabel('Time [Ga]', rotation=270, fontsize=15)


    # Lunar Chronology Track
    ax4.plot(strat['lith'], strat['depth'], color = 'white', linewidth = 0.5)
    ax4.set_xlim(0, 1)
    ax4.set_xlabel('Lunar Epochs', fontsize = 15)
    ax4.xaxis.label.set_color('black')
    ax4.tick_params(axis='x', colors='white')
    ax4.set_ylabel('Time (Ga)', rotation=-90, fontsize=15)
    ax4.set_yticks(strat.depth.values)
    ax4.set_yticklabels(strat.time.values / 1e9)
    ax4.spines['top'].set_edgecolor('white')
    ax4.yaxis.set_label_position('right')
    ax4.yaxis.tick_right()
    ax4.tick_params(axis='y', labelsize=15)

    # Adding Epochs to Plot
    #ax4.axhspan(4.17, 1.55, alpha=0.3, color='#253494')
    #ax4.axhspan(1.55, 1.11, alpha=0.4, color='#2c7fb8')
    #ax4.axhspan(1.11, 0.75, alpha=0.3, color='#41b6c4')
    #ax4.axhspan(0.75, 0.3, alpha=0.3, color='#a1dab4')
    #ax4.axhspan(0.3, 0, alpha=0.3, color='yellow')

    # Adding Text
    # ax4.text(0.5, 4.15, 'Pre-Nectarian (4.5-3.9 Ga)', fontsize=13)
    # ax4.text(3.76, 29.2, 'Atmosphere', fontsize=13)

    # Common functions for setting up the plot can be extracted into
    # a for loop. This saves repeating code.
    for ax in [ax1]:
        ax.set_ylim(strat_layered.depth.max(), 0)
        ax.grid(which='major', color='lightgrey', linestyle='-')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.spines['top'].set_position(('axes', 1.02))

    for ax in [ax3]:
        ax.set_ylim(strat_layered.depth.max(), 0)
        ax.grid(which='major', color='none')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.spines['top'].set_position(('axes', 1.02))

    for ax in [ax4]:
        ax.set_ylim(strat_layered.depth.max(), 0)
        ax.grid(which='major', color='none')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.spines['top'].set_position(('axes', 1.02))

    # for ax in [ax3]:
    #     plt.setp(ax.get_yticklabels(), visible=False)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    plt.savefig(savepath, bbox_inches='tight', dpi=300)
    print('finish')

if __name__ == '__main__':
    # Choose cold trap crater
    coldtrap = 'Shoemaker'
    data_dir = '' # Manually set model result dir, or else get from cfg
    seed = 63771

    # Get cfg
    cfg_dict = {}
    if len(sys.argv) > 1:   
        # 1st cmd-line arg is path to custom cfg
        cfg_dict = default_config.read_custom_cfg(sys.argv[1])
    if seed:
        cfg_dict['seed'] = seed
    cfg = default_config.Cfg(**cfg_dict)
    if not data_dir:
        data_dir = Path(cfg.outpath)

    # Input paths
    f_ice = data_dir.joinpath('ice_columns_mpies.csv')
    f_ej = data_dir.joinpath('ej_columns_mpies.csv')

    # Output paths
    figbasename = f'strat_col_{coldtrap}_{cfg.run_name}_{cfg.run_date}'
    figpath = Path(cfg.figspath)
    plotpath = figpath.joinpath(figbasename + '.png')
    keypath = figpath.joinpath(figbasename + '_key.png')

    # Read in crater, ice, ej data
    crater_csv = mp.read_crater_list(cfg.crater_csv_in, cfg.crater_cols)
    ice_df = pd.read_csv(f_ice)
    ej_df = pd.read_csv(f_ej)
    time_arr = mp.get_time_array(cfg)
 
    # Make strat col and plots
    strat_df = get_strat(ice_df, ej_df, coldtrap)
    strat_layered = get_strat_layers(strat_df, cfg.timestart)
    
    # Get lith key and dict of lithology : numerical key
    lith_key = get_lith_key(strat_layered)
    lith2key = {v['lith']: k for k, v in lith_key.items()}
    strat_layered['lith'] = strat_layered['label'].map(lith2key)

    makeplot(strat_layered, coldtrap, plotpath)
    makekey(lith_key, coldtrap, keypath)
