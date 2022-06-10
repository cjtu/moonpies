"""
Plots of crater and basin ages with lunar geologic eras
A. Madera
Last updated 06/10/22 (CJTU)
"""
import seaborn as sns
from matplotlib import pyplot as plt
from moonpies import moonpies as mp
from moonpies import default_config

CFG = default_config.Cfg()
outpath = CFG.figs_path + '/crater_basin_ages.pdf'
# outpath.replace('.pdf', '.png')

# Read crater and basin list
dc = mp.read_crater_list(CFG).sort_values('age').set_index('cname')
dc[["age", "age_low", "age_upp"]] = dc[["age", "age_low", "age_upp"]] / 1e9

db = mp.read_basin_list(CFG).sort_values('age').set_index('cname')
db[["age", "age_low", "age_upp"]] = db[["age", "age_low", "age_upp"]] / 1e9

nec_age = db.loc["Nectaris", "age"]
imb_age = db.loc["Imbrium", "age"]
era_age = 3.2
cop_age = 1.1
eras = {
    'pNe.': dict(xmin=4.5, xmax=nec_age, facecolor='#253494', alpha=0.3),
    'Ne.': dict(xmin=nec_age, xmax=imb_age, facecolor='#2c7fb8', alpha=0.4),
    'Im.': dict(xmin=imb_age, xmax=era_age, facecolor='#41b6c4', alpha=0.3),
    'Era.': dict(xmin=era_age, xmax=cop_age, facecolor='#a1dab4', alpha=0.3)
}

# Plot params
sns.set_style("white")
fs_medium = 12
fs_large = 13

#Figure and Axes
fig, axc = plt.subplots(1, 1, figsize=(8, 8))

# Make basin inset
left, bottom, width, height = [0.6, 0.3, 0.35, 0.65]
axb = fig.add_axes([left, bottom, width, height])

for ax, df in zip([axc, axb], [dc, db]):
    # Plot crater / basin ages
    xerr = (df.age_low, df.age_upp)
    ax.errorbar(df.age, df.index, xerr=xerr, fmt='ko', ms=5, capsize=4, capthick=2)
    ax.invert_xaxis()
    if ax == axc:
        ax.set_ylabel('Craters', fontsize=fs_large, labelpad=-0.2)
        ax.set_xlabel('Absolute Model Ages [Ga]', fontsize=fs_large)
        ax.set_xlim(4.4, 1.4)
        ax.tick_params(axis='y', labelsize=fs_large)
    else:
        ax.set_title('Basins', pad=3, fontsize=fs_large)
        ax.set_xlabel('Absolute Model Ages [Ga]', fontsize=fs_medium)
        ax.set_xlim(4.35, 3.79)

    # Add Chronological Periods
    for era, params in eras.items():
        ax.axvspan(**params, edgecolor='none')
        if ax == axb and era == 'Era.':
            continue
        x = max(params['xmax'], ax.get_xlim()[1])
        y = ax.get_ylim()[0]
        ax.annotate(era, xy=(x, y), xycoords='data', fontsize=fs_medium, 
                    weight='bold', ha='right', va='bottom')

# Save Figure
dpi = 300 if '.png' in outpath else None 
plt.savefig(outpath, dpi=dpi)
print('Saved figure to ' + outpath)