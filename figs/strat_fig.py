"""Make strat column figure."""
import pandas as pd
import matplotlib.pyplot as plt
from moonpies import mixing as mm
import pandas as pd
import numpy as np

# Choose cold trap crater
coldtrap = "Haworth"

# Set paths
fdir = "/home/cjtu/projects/moonpies/figs/"
f_ice = "/home/cjtu/projects/moonpies/data/210702/essi00000/ice_columns_essi00000.csv"

# Output plot names
keypath = fdir + "strat_key.png"
plotpath = fdir + "strat.png"

# Make lithology style dict
hatches = ["\\|","|*","-\\","x*","O|","O.","*-","//","\\\\","||","--","++",
           "xx","OO","..","**","/","\\","|","-","+","x","O",".","*",]


def get_lith_key(ejecta_crater_names):
    lith_key = {
        len(ejecta_crater_names)
        + 1: {"lith": "Ice", "lith_num": -2, "hatch": "o", "color": "#247BA0"},
        len(ejecta_crater_names)
        + 2: {
            "lith": "Ice and Regolith",
            "lith_num": -1,
            "hatch": "o-",
            "color": "#D8D8D8",
        },
    }
    for i, cname in enumerate(ejecta_crater_names):
        lith_key[i] = {"lith": cname, "hatch": hatches[i], "color": "#A48C79"}
    return lith_key


# Functions
def get_strat_col(ice_df, ej_df, coldtrap="Haworth"):
    """Return strat column df of coldtrap from ice_df, ej_df."""
    strat_col = ice_df[["time"]].copy()
    strat_col["ice"] = ice_df.loc[:, coldtrap].values
    strat_col["ejecta"] = ej_df.loc[:, coldtrap].values
    strat_col.insert(1, "depth", 0)
    strat_col.insert(2, "label", "")

    for i, row in strat_col.iterrows():
        strat_col.loc[i + 1, "depth"] = (
            strat_col.loc[i, "depth"] + row.ice + row.ejecta
        )
        if row.ejecta:
            ej_row = ej_df.iloc[i]
            label = ej_row[ej_row == ej_row.min()].keys()[0]
        elif row.ice:
            label = "Ice"
        else:
            label = "void"
        strat_col.loc[i, "label"] = label
    strat_col = strat_col.drop(len(strat_col) - 1)
    strat_col["depth"] = strat_col.depth.max() - strat_col.depth
    strat_col = strat_col[strat_col.label != "void"]
    # Make into ranges
    strat = pd.concat((strat_col.copy(), strat_col.copy()))
    strat = strat.sort_values("time")
    strat = strat.iloc[:-1]
    top = strat.iloc[0:1]
    top.loc[:, ["depth", "ejecta", "ice"]] = (0, 0, 0)
    strat = pd.concat((top, strat))
    strat["label"] = pd.concat(
        (strat.iloc[1:]["label"], strat.iloc[-1:]["label"])
    ).values
    strat.to_csv("strat_output.csv")
    return strat


def makekey(savepath):
    """Plot lithology key."""
    x = [0, 1]
    y = [1, 0]

    ncols = 4
    nrows = int(np.ceil(len(lith_key.keys()) / 4))
    xsize = ncols * 3
    ysize = nrows * 1.5
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, 
                             sharey=True, figsize=(xsize, ysize),
                             subplot_kw={"xticks": [], "yticks": []})

    for ax, (_, v) in zip(axes.flat, lith_key.items()):
        title = v["lith"]
        if "Ice" not in title:
            title += " Ejecta"
        ax.plot(x, y, linewidth=0)
        ax.fill_betweenx(y, 0, 1, facecolor=v["color"], hatch=v["hatch"]
        )
        ax.set_xlim(0, 0.1)
        ax.set_ylim(0, 1)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight", dpi=300)


def makeplot(strat, top_depth, bottom_depth, age_min, age_max, savepath):
    fig, ax = plt.subplots(figsize=(20, 20))

    # Set up the plot axes
    ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
    ax2 = ax1.twiny()
    ax3 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, sharey=ax1)
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
    # ax1.plot(strat['ballistic_speed'], strat['depth'], color = "green", linewidth = 1.5)
    # ax1.set_xlabel('Balistic Speed (m/s)', fontsize=15)
    # ax1.xaxis.label.set_color("green")
    # ax1.set_xlim(0, 900)
    # ax1.set_ylabel('Depth (m)', fontsize = 15)
    # ax1.tick_params(axis='x', colors="green", labelsize=10)
    # ax1.tick_params(axis='y', labelsize=15)
    # ax1.spines["top"].set_edgecolor("green")
    # ax1.title.set_color('green')
    # ax1.set_xticks([0, 300, 600, 900])

    # Kinetic Energy overlaid Ballistic Speed Track
    # ax2.plot(strat['kinetic_e_km'], strat['depth'], color = 'red', linewidth = 1.5)
    # ax2.set_xlabel('Kinetic Energy of Ejecta (J/km^2)', fontsize=15)
    # ax2.xaxis.label.set_color('red')
    # ax2.set_xlim(15000, 160000)
    # ax2.tick_params(axis='x', colors='red', labelsize=10)
    # ax2.spines["top"].set_position(("axes", 1.08))
    # ax2.spines["top"].set_visible(True)
    # ax2.spines["top"].set_edgecolor('red')
    # ax2.set_xticks([15000, 90000, 160000])

    # lith track
    ax3.plot(strat["lith"], strat["depth"], color="black", linewidth=0.5)
    ax3.set_xlabel("Stratigraphy Column", fontsize=15)
    ax3.set_xlim(0, 1)
    ax3.xaxis.label.set_color("black")
    ax3.tick_params(axis="x", colors="black", labelsize=10)
    ax3.spines["top"].set_edgecolor("black")

    for k, v in lith_key.items():
        ax3.fill_betweenx(strat["depth"], 0, strat["lith"],
                          where=(strat["lith"] == k),
                          facecolor=v["color"], hatch=v["hatch"])
    ax3.set_xticks([0, 1])

    # Lunar Chronology Track
    # ax4.plot(strat['lith'], strat['age'], color = 'white', linewidth = 0.5)
    # ax4.set_xlim(0, 1)
    # ax4.set_xlabel('Lunar Epochs', fontsize = 15)
    # ax4.xaxis.label.set_color('black')
    # ax4.tick_params(axis='x', colors='white')
    # ax4.set_ylabel('Chronology (Ga)', rotation=-90, fontsize=15)
    # ax4.set_yticks([0, 0.6, 0.96, 1.11, 1.27, 1.70, 2.9, 3.35, 3.48, 3.84, 4.17])
    # ax4.set_yticklabels(['0', '2.5', '3.7', '3.8', '3.85', '3.98', '4', '4.01', '4.07', '4.13', '4.17'])
    # ax4.spines["top"].set_edgecolor('white')
    # ax4.yaxis.set_label_position("right")
    # ax4.yaxis.tick_right()
    # ax4.tick_params(axis='y', labelsize=15)

    # Adding Epochs to Plot
    # ax4.axhspan(4.17, 1.55, alpha=0.3, color='#253494')
    # ax4.axhspan(1.55, 1.11, alpha=0.4, color='#2c7fb8')
    # ax4.axhspan(1.11, 0.75, alpha=0.3, color='#41b6c4')
    # ax4.axhspan(0.75, 0.3, alpha=0.3, color='#a1dab4')
    # ax4.axhspan(0.3, 0, alpha=0.3, color='yellow')

    # Adding Text
    # ax4.text(0.5, 4.15, 'Pre-Nectarian (4.5-3.9 Ga)', fontsize=13)
    # ax4.text(3.76, 29.2, 'Atmosphere', fontsize=13)

    # Common functions for setting up the plot can be extracted into
    # a for loop. This saves repeating code.
    for ax in [ax1]:
        ax.set_ylim(bottom_depth, top_depth)
        ax.grid(which="major", color="lightgrey", linestyle="-")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_position(("axes", 1.02))

    for ax in [ax3]:
        ax.set_ylim(bottom_depth, top_depth)
        ax.grid(which="major", color="none")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_position(("axes", 1.02))

    # for ax in [ax4]:
    #     ax.set_ylim(age_max, age_min)
    #     ax.grid(which='major', color='none')
    #     ax.xaxis.set_ticks_position("top")
    #     ax.xaxis.set_label_position("top")
    #     ax.spines["top"].set_position(("axes", 1.02))

    for ax in [ax3]:
        plt.setp(ax.get_yticklabels(), visible=False)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    plt.savefig(savepath, bbox_inches="tight", dpi=300)

# TODO: update to use cfg
if __name__ == "__main__":
    # Generate ejecta thicknesses
    ice_df = pd.read_csv(f_ice)
    df = mm.read_crater_list()  # DataFrame, len: NC
    ej_thickness_time = mm.get_ejecta_thickness_matrix(
        df
    )  # [m] shape: NY,NX,NT
    ej_df = pd.DataFrame(ej_thickness_time, columns=df.cname)

    # Get lith key and dict of lithology : numerical key
    lith_key = get_lith_key(ej_df.columns)
    lith2key = {v["lith"]: k for k, v in lith_key.items()}

    # Make strat col and plots
    strat = get_strat_col(ice_df, ej_df, coldtrap)
    strat["lith"] = strat["label"].map(lith2key)
    strat = strat.sort_values("time").reset_index()
    makeplot(strat, 0, strat.depth.max(), strat.time.min(),
             strat.time.max(), plotpath)
    makekey(keypath)
