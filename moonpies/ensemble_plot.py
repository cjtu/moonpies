"""Script to find and compile ice_cols from mixing.py runs into one plot."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import config

cfg = config.Cfg()
# Set these paths to match your system
# DATA_DIR should be full path to run_dir from mixing.py
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]  # 1st cmd-line arg is path to data
else:
    DATA_DIR = Path(cfg.outpath).parent
MODE = 'moonpies'
DATE = '210706'
DATA_DIR = Path(DATA_DIR).resolve()
FIGPATH = f'{cfg.figspath}ensemble_ice_{cfg.run}_{cfg.run_date}.png'
ICE_COL = 2  # Column to use from ice_cols.csv (1 up to #cols)

plt.style.use('tableau-colorblind10')
plt.rcParams.update({
        'figure.figsize': (8, 8),
        'figure.facecolor': 'white',
        'xtick.top': True,
        'xtick.direction': 'in',
        'ytick.right': True,
        'ytick.direction': 'in',
})

# Find all ice_columns csvs recursively
csvs = DATA_DIR.rglob(f'ice_columns_*.csv')
f, ax = plt.subplots()
for i, csv in enumerate(csvs):
        df = pd.read_csv(csv, usecols=[0, ICE_COL])
        ax.plot(df.iloc[:, 0], df.iloc[:, 1])
        if i % 100 == 0:
            print(f'Starting csv {i}: {csv}', flush=True)
if 'i' not in locals():
    print(f'No ice_columns csvs found, check DATA_DIR: {DATA_DIR}')
    quit()
else:
    print(f'Finished csv {i}: {csv}', flush=True)

# Configure and save figure
ax.set_xlabel('Time (Ga)')
ax.set_ylabel('Ice retained (m)')
ax.set_ylim(-1, 250)
ax.set_xlim(df.time.max(), 0)
f.savefig(FIGPATH, bbox_inches='tight')
print(f'Saved figure to {FIGPATH}')
