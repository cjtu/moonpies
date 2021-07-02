"""Script to find and compile ice_cols from mixing.py runs into one plot."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Set these paths to match your system
# DATA_DIR should be full path to run_dir from mixing.py

MODE = 'essi'
DATE = '210701'
DATA_DIR = f'/home/cjtu/projects/essi21/data/{DATE}/'
FIGPATH = f'/home/cjtu/projects/essi21/figs/ensemble_ice_{MODE}_{DATE}.png'
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
csvs = Path(DATA_DIR).glob(f'**/ice_columns_{MODE}*.csv')
f, ax = plt.subplots()
for i, csv in enumerate(csvs):
        df = pd.read_csv(csv, usecols=[0, ICE_COL])
        ax.plot(df.iloc[:, 0], df.iloc[:, 1])
        if i % 100 == 0:
            print(f'Starting csv {i}: {csv}', flush=True)
if 'i' not in locals():
    print(f'No ice_columns csvs found, check DATA_DIR: {DATA_DIR}')
    quit()
ax.set_xlabel('Time (Ga)')
ax.set_ylabel('Ice retained (m)')
ax.set_ylim(-1, 160)
ax.set_xlim(df.time.max(), 0)
f.savefig(FIGPATH, bbox_inches='tight')
print(f'Saved figure to {FIGPATH}')
