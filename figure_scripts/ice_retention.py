"""
Figure plotting impactor retention vs speed in Cannon and MoonPIES mode.

Fits to Ong et al. (2010).
"""
import os
import sys
from pathlib import Path
mdir = str(Path(__file__).parents[1] / 'moonpies') + os.sep
fdir = str(Path(__file__).parents[1] / 'figures') + os.sep
sys.path.insert(0, mdir)
import numpy as np
import matplotlib.pyplot as plt
import moonpies as mp
import default_config

# Set config
cfg = default_config.Cfg()


# Data from Ong et al. (2010)
ong_x = [10, 15, 30, 45, 60]
ong_y = [1, 1.97E-01, 1.47E-02, 1.93E-03, 6.60E-06]

# Generate moonpies data
v = np.arange(70)  # speed [km/s]
retention_c = mp.ice_retention_factor(v*1e3, 'cannon')
retention_m = mp.ice_retention_factor(v*1e3, 'moonpies')

# Make plot
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(ong_x, ong_y, 'o', label='Ong et al. (2010)')
plt.semilogy(v, retention_c, label='cannon mode')
plt.semilogy(v, retention_m, label='moonpies mode')
plt.title('Ice Retention vs Impact Speed')
plt.ylabel('Ice Retention Fraction [0-1]')
plt.xlabel('Impact Speed [km/s]')
plt.ylim(1E-6, 1.5)
plt.xlim(0, 65)
plt.legend()

# Add version and save figure
mp.plot_version(cfg, loc='ll')
plt.savefig(fdir + 'impactor_retention.png', dpi=300)