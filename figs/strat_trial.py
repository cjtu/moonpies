# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
haworth = pd.read_csv('./figs/haworth_strat_trial_data2.csv')
haworth.head()


# %%
lithology_numbers = {1: {'lith':'Ice', 'lith_num':1, 'hatch': 'o', 'color':'#247BA0'},
                 2: {'lith':'Shoemaker Ejecta', 'lith_num':2, 'hatch':'||', 'color':'#A48C79'},
                 3: {'lith':'Amundsen Ejecta', 'lith_num':3, 'hatch':'\\', 'color':'#A48C79'},
                 4: {'lith':'Schrodinger Ejecta', 'lith_num':4, 'hatch':'//', 'color':'#A48C79'},
                 5: {'lith':'Cabeus Ejecta', 'lith_num':5, 'hatch':'X', 'color':'#A48C79'},
                 6: {'lith':'Nobile Ejecta', 'lith_num':6, 'hatch':'+', 'color':'#A48C79'},
                 7: {'lith':'Scott Ejecta', 'lith_num':7, 'hatch':'\|', 'color':'#A48C79'},
                 8: {'lith':'Ice and Regolith', 'lith_num':8, 'hatch':'o-', 'color':'#D8D8D8'}}

#2E86AB
#D78521
#9C816D


# %%
df_lith = pd.DataFrame.from_dict(lithology_numbers, orient='index')

df_lith.index.name = 'lithology'

df_lith


# %%
x = [0, 1]
y = [1, 0]

fig, axes = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True, figsize=(10,5), subplot_kw={'xticks':[], 'yticks':[]})

""" Repeating the steps above"""

for ax, key in zip(axes.flat, lithology_numbers.keys()):
    ax.plot(x, y, linewidth=0)
    ax.fill_betweenx(y, 0, 1, facecolor=lithology_numbers[key]['color'], hatch=lithology_numbers[key]['hatch'])
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0, 1)
    ax.set_title(str(lithology_numbers[key]['lith']))
    
plt.tight_layout()

plt.savefig('Haworth_Strat_Trial_Key.png', bbox_inches='tight')
plt.show()


# %%
def makeplot(well, top_depth, bottom_depth, age_min, age_max):
    fig, ax = plt.subplots(figsize=(20,20))

    #Set up the plot axes
    ax1 = plt.subplot2grid((1,3), (0,0), rowspan=1, colspan = 1)
    ax2 = ax1.twiny()
    ax3 = plt.subplot2grid((1,3), (0,1), rowspan=1, colspan = 1, sharey = ax1)
    ax4 = plt.subplot2grid((1,3), (0,2), rowspan=1, colspan = 1)

    # As our curve scales will be detached from the top of the track,
    # this code adds the top border back in without dealing with splines
    ax10 = ax1.twiny()
    ax10.xaxis.set_visible(False)
    ax11 = ax3.twiny()
    ax11.xaxis.set_visible(False)
    ax13 = ax4.twiny()
    ax13.xaxis.set_visible(False)

    # Ballistic Speed Track
    # ax1.plot(haworth['ballistic_speed'], haworth['depth'], color = "green", linewidth = 1.5)
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
    # ax2.plot(haworth['kinetic_e_km'], haworth['depth'], color = 'red', linewidth = 1.5)
    # ax2.set_xlabel('Kinetic Energy of Ejecta (J/km^2)', fontsize=15)
    # ax2.xaxis.label.set_color('red')
    # ax2.set_xlim(15000, 160000)
    # ax2.tick_params(axis='x', colors='red', labelsize=10)
    # ax2.spines["top"].set_position(("axes", 1.08))
    # ax2.spines["top"].set_visible(True)
    # ax2.spines["top"].set_edgecolor('red')
    # ax2.set_xticks([15000, 90000, 160000])
    
    # Lithology track
    a = haworth['lithology']
    d = haworth['depth']
    ax3.plot(haworth['lithology'], haworth['depth'], color = "black", linewidth = 0.5)
    ax3.set_xlabel('Haworth Potential Stratigrahy Column', fontsize=15)
    ax3.set_xlim(0, 1)
    ax3.xaxis.label.set_color("black")
    ax3.tick_params(axis='x', colors="black", labelsize=10)
    ax3.spines["top"].set_edgecolor("black")

    for key in lithology_numbers.keys():
        color = lithology_numbers[key]['color']
        hatch = lithology_numbers[key]['hatch']
        d = haworth['depth']
        ax3.fill_betweenx(haworth['depth'], 0, haworth['lithology'], where=(haworth['lithology']==key),
                         facecolor=color, hatch=hatch)
    ax3.set_xticks([0, 1])
    
    # Lunar Chronology Track
    # ax4.plot(haworth['lithology'], haworth['age'], color = 'white', linewidth = 0.5)
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
    
    #Adding Epochs to Plot
    # ax4.axhspan(4.17, 1.55, alpha=0.3, color='#253494')
    # ax4.axhspan(1.55, 1.11, alpha=0.4, color='#2c7fb8')
    # ax4.axhspan(1.11, 0.75, alpha=0.3, color='#41b6c4')
    # ax4.axhspan(0.75, 0.3, alpha=0.3, color='#a1dab4')
    # ax4.axhspan(0.3, 0, alpha=0.3, color='yellow')
    
    #Adding Text
    #ax4.text(0.5, 4.15, 'Pre-Nectarian (4.5-3.9 Ga)', fontsize=13)
    #ax4.text(3.76, 29.2, 'Atmosphere', fontsize=13)
    
    # Common functions for setting up the plot can be extracted into
    # a for loop. This saves repeating code.
    # for ax in [ax1]:
    #     ax.set_ylim(bottom_depth, top_depth)
    #     ax.grid(which='major', color='lightgrey', linestyle='-')
    #     ax.xaxis.set_ticks_position("top")
    #     ax.xaxis.set_label_position("top")
    #     ax.spines["top"].set_position(("axes", 1.02))
    
    for ax in [ax3]:
        ax.set_ylim(bottom_depth, top_depth)
        ax.grid(which='major', color='none')
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
        plt.setp(ax.get_yticklabels(), visible = False)
        
    plt.tight_layout()
    fig.subplots_adjust(wspace = 0.15)
    

    # for key in lithology_numbers.keys():
    #     color = lithology_numbers[key]['color']
    #     hatch = lithology_numbers[key]['hatch']
    #     ax3.fill_betweenx(haworth['depth'], 0, haworth['lithology'], where=(haworth['lithology']==key),
    #                       facecolor=color, hatch=hatch)
        
    plt.savefig('Haworth_Strat_Trial.png', bbox_inches='tight')
    plt.show()


# %%
makeplot(haworth, 0, 278, 0, 4.17)





