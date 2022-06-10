"""
Figure plotting impactor retention vs speed in Cannon and MoonPIES mode.

Fits to Ong et al. (2010).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from moonpies import moonpies as mp
from moonpies import default_config

FIGDIR = ''  # Set or leave blank to use default (moonpies/figs)
if not FIGDIR:
    FIGDIR = default_config.Cfg().figspath
FIGDIR = str(Path(FIGDIR).resolve() / "_")[:-1]  # add trailing slash


# Set config
cfg_c = default_config.Cfg(mode='cannon')
cfg_m_asteroid = default_config.Cfg(mode='moonpies')
cfg_m_comet = default_config.Cfg(mode='moonpies', is_comet=True)
print(cfg_m_comet.is_comet)

# Data from Ong et al. (2010)
ong_x = [10, 15, 30, 45, 60]
ong_y = [1, 1.97E-01, 1.47E-02, 1.93E-03, 6.60E-06]

# Generate moonpies data
v = np.linspace(0, 70, 7000)  # speed [km/s]
retention_c = mp.ice_retention_factor(v*1e3, cfg_c)
retention_ma = mp.ice_retention_factor(v*1e3, cfg_m_asteroid)
retention_mc = mp.ice_retention_factor(v*1e3, cfg_m_comet)

# Make plot
fig, ax = plt.subplots(figsize=(7, 5))
plt.semilogy(v, retention_c, c='tab:gray', lw=3, label='Cannon et al. (2020)')
plt.semilogy(v, retention_ma, ':', c='k', lw=2, label='This work, asteroid')
plt.semilogy(v, retention_mc, '--', c='tab:blue', lw=2, label='This work, comet')
plt.plot(ong_x, ong_y, 'o', ms=8, c='k', label='Ong et al. (2010)')
plt.title('Ice Retention vs Impact Speed')
plt.ylabel('Ice Retention Fraction [0-1]')
plt.xlabel('Impact Speed [km/s]')
plt.ylim(1E-6, 1.5)
plt.xlim(0, 65)
plt.legend()

# Add version and save figure
mp.plot_version(cfg_c, loc='ll')
plt.savefig(FIGDIR + 'impactor_retention.pdf')