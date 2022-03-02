"""Script to find and compile ice_cols from mixing.py runs into one plot."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from moonpies import default_config

# Manually set datadir else try to read from cfg
data_dir = ''

# Set ice column
col = 2  # Column to use from ice_cols.csv (1 up to #cols)

# Get cfg
cfg_dict = {}
if len(sys.argv) > 1:   # 1st cmd-line arg is path to custom cfg
    cfg_dict = default_config.read_custom_cfg(sys.argv[1]) 
cfg = default_config.Cfg(**cfg_dict)
if not data_dir:
    data_dir = Path(cfg.outpath).parent
figname = f'ensemble_ice_{cfg.run_name}_{cfg.run_date}.png'
figpath = Path(cfg.figspath).joinpath(figname).as_posix()

# Init plot
plt.style.use('tableau-colorblind10')
plt.rcParams.update({
        'figure.figsize': (10, 8),
        'figure.facecolor': 'white',
        'xtick.top': True,
        'xtick.direction': 'in',
        'ytick.right': True,
        'ytick.direction': 'in',
})

# Find all ice_columns csvs recursively
csvs = data_dir.rglob(f'ice_columns_*.csv')
f, ax = plt.subplots()
for i, csv in enumerate(csvs):
        time, ice_col = pd.read_csv(csv, usecols=[0, col]).values.T
        ax.plot(time/1e9, ice_col)
        if i % 100 == 0:
            print(f'Starting csv {i}: {csv}', flush=True)
if 'i' not in locals():
    print(f'No ice_columns csvs found, check DATA_DIR: {data_dir}')
    quit()
else:
    print(f'Finished csv {i}: {csv}', flush=True)

# Configure and save figure
ax.set_xlabel('Time (Ga)')
ax.set_ylabel('Ice retained (m)')
ax.set_ylim(0, 160)
ax.set_xlim(max(time)/1e9, 0)
f.savefig(figpath, bbox_inches='tight')
print(f'Saved figure to {figpath}')
