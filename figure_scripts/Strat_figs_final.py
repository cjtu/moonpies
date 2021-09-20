import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import moonpies as mm
import pandas as pd
import numpy as np
import default_config
cfg = default_config.Cfg()

#Set Paths
mpath = 'C:\\Users\\Amade\\Internship\\moonpies_package\\'
run_path = mpath + 'data\\210804_mpies\\12888\\'

fdir = mpath + 'data\\210804_mpies\\12888\\figs\\'

COI = 'Cabeus'
coldtrap = pd.read_csv(run_path + COI + '_strat.csv')
crater_list = mm.read_crater_list(cfg)

# make cutoff for coldtrap its max possible age since i didn't save these out

#crater = crater_list[crater_list['cname'] == coldtrap]
#maxage = (crater.age + crater.age_upp).values[0]
maxage = 3.88e9  # Set here if known, else use maxage from above
coldtrap = coldtrap[coldtrap.time < maxage]

# coldtrap = pd.read_csv(run_path + "Shackleton_strat.csv")

#Output plot names
keypath = fdir + 'strat_key_cab_11304_v1.pdf'
plotpath = fdir + 'strat_cab_11304_v1.pdf'

#Make lithology style dictionary
hatches = ["\\|","-\\","O|","O.","//","\\\\","||","--","++",
           "xx","OO","..","/","\\","|","-","+","x","O","."]

#Make ice% colormap (first color is brown, last is white)
colors = np.array([(164, 140, 121),(216,216,216)])/255
ICE_CM = mpl.colors.LinearSegmentedColormap.from_list("Custom", colors, N=100)
#ICE_CM = plt.get_cmap('terrain_r')

#Set up definitions for lithology key
def get_lith_key(lithology, cmap=ICE_CM):
    norm = mpl.colors.Normalize(vmin=30, vmax=70)

    lith_key = {
        len(lithology) +1: {"lith": "Ice", "lith_num": -2, "hatch": "o", "color": '#247BA0'}}
        #len(lithology) +2:  {"lith": "Ice and Regolith", "lith_num": -1, "hatch": "o-", "color": cmap(norm(100))}}
    for i, label in enumerate(lithology):
        if label == 'Ice':
            continue
        color = cmap(norm(coldtrap.iloc[i].icepct))
        
        lith_key[i] = {'lith': label, 'hatch': hatches[i], 'color':'#A48C79'}
    return lith_key

def makekey(savepath):
    x = [0, 1]
    y = [1, 0]

    ncols = 4
    nrows = int(np.ceil(len(lith_key.keys()) /4 ))
    xsize = ncols *3
    ysize = nrows * 1.5
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(xsize,ysize), subplot_kw={'xticks':[], 'yticks':[]})

    """ Repeating the steps above"""
    for ax, (_, v) in zip(axes.flat, lith_key.items()):
            title = v["lith"]
            if "Ice" not in title:
                title += " Ejecta"
            ax.plot(x, y, linewidth=0)
            ax.fill_betweenx(y, 0, 1, facecolor=v["color"], hatch=v["hatch"])
            ax.set_xlim(0, 0.1)
            ax.set_ylim(0, 1)
            ax.set_title(title)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight', dpi=300)


#Setting up definitions for strat columns
def get_strat_col_ranges(ColdtrapOI):
    strat_col = ColdtrapOI
    
    # #Adding Ballistic Speed and Kinetic Energy
    # strat_col.insert(6, 'ball_speed', 0)
    # strat_col.insert(7, 'kinetic_e', 0)

    #Make into ranges
    strat = pd.concat((strat_col.copy(), strat_col.copy()))
    #strat = strat.sort_values('time')

    strat = strat.sort_values('depth')
    strat = strat.iloc[:-1]
    top = strat.iloc[0:1]
    top.loc[:, ['depth', 'ejecta', 'ice', 'time']] = (0,0,0,0)
    strat = pd.concat((top, strat))
    strat['label'] = pd.concat((strat.iloc[1:]['label'], strat.iloc[-1:]['label'])).values
    strat.to_csv('strat_output.csv')
    return strat

#Plotting Strat Columns
def makeplot(strat, top_depth, bottom_depth, savepath):
        #Get the depth boundaries of each distinct layer in strat
    adj_check = (strat.label != strat.label.shift()).cumsum()
    distinct_layers = strat.groupby(['label', adj_check], as_index=False,
                                sort=False).agg({"depth" : ['max'], "time": ['max']})
    yticks_depth = distinct_layers.depth['max'].values
    yticks_depth = np.insert(yticks_depth, 0, 0)  #Add zero to start

    yticks_time = distinct_layers.time['max'].values
    yticks_time = np.insert(yticks_time, 0, 0)

    fig, ax = plt.subplots(figsize=(20, 20))

    # ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
    # ax2 = ax1.twiny()
    ax3 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1)
    ax4 = ax3.twinx()
    
    # ax10 = ax1.twiny()
    # ax10.xaxis.set_visible(False)
    ax11 = ax3.twiny()
    ax11.xaxis.set_visible(True)

    # lith track
    ax3.plot(np.ones_like(strat.depth), strat["depth"], color="black", linewidth=0.5)
    ax3.set_xlabel("Cabeus Stratigraphy", fontsize=40)
    ax3.set_xlim(0, 1)
    ax3.xaxis.label.set_color('black')
    ax3.tick_params(axis="x", colors='none', labelsize=0)
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(2)
    ax3.spines["top"].set_edgecolor('none')
    ax3.spines['right'].set_edgecolor('black')
    ax3.set_ylabel('Depth[m]', fontsize=40)
    ax3.tick_params(axis='y', labelsize=25, width=3)
    ax4.set_ylim(150, top_depth)
    #ax4.set_ylim(ax4.get_ylim()[::-1])
    ax4.set_yticks(yticks_depth)
    ax4.tick_params(axis='y', labelsize=25, width=3)
    ax4.set_ylabel('Lithology Depths[m]', fontsize = 40, rotation=-90)

    for k, v in lith_key.items():
        ax3.fill_betweenx(strat["depth"], 0, 1,
                        where=(strat["lith"] == k), facecolor=v["color"], hatch=v["hatch"])
    ax3.set_xticks([0, 1])

    # for ax in [ax1]:
    #     ax.set_ylim(150, top_depth)
    #     ax.grid(which="major", color="lightgrey", linestyle="-")
    #     ax.xaxis.set_ticks_position("top")
    #     ax.xaxis.set_label_position("top")
    #     ax.spines["top"].set_position(("axes", 1.02))

    for ax in [ax3]:
        ax.set_ylim(150, top_depth)
        ax.grid(which="major", color="none")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_position(("axes", 1.02))

    
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    plt.savefig(savepath, bbox_inches="tight")

#if __name__ == "__main__":

#Make strat col and plots
strat = get_strat_col_ranges(coldtrap)

#Get lith key and dict of lithology:numerical key
lith_key = get_lith_key(coldtrap.label.unique())
lith2key = {v["lith"]: k for k, v in lith_key.items()}
strat["lith"] = strat["label"].map(lith2key)

makeplot(strat, 0, strat.depth.max(), plotpath)
makekey(keypath)
