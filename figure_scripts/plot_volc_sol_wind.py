'''
Script to plot volatile mass of volcanic ice and solar wind ice through time
'''

import matplotlib.pyplot as plt 
# import sys and use the following command to point to the directory where moonpies.py is (This is not necessary if this script is in that directory)
import sys
sys.path.insert(0, "/Users/tylerpaladino/Documents/ISU/LPI_NASA/Codes/moonpies_package/moonpies")

from moonpies import default_config
import moonpies as mp
cfg= default_config.Cfg()
TIME_ARR = mp.get_time_array(cfg)

out_dir = '/Users/tylerpaladino/Documents/ISU/LPI_NASA/figs/'


# Volcanic volatile mass through time
fig1,ax1 = plt.subplots(figsize=(8, 5))
volc_ice_h20_mass = mp.volcanic_ice_nk(TIME_ARR, cfg)
plt.plot(TIME_ARR/1e9, volc_ice_h20_mass, label = r'$\rm H_{2}O$ min')

cfg.nk_species = 'min_co'
volc_ice_co_mass = mp.volcanic_ice_nk(TIME_ARR, cfg)
plt.plot(TIME_ARR/1e9, volc_ice_co_mass, label = 'CO min')

cfg.nk_species = 'min_s'
volc_ice_s_mass = mp.volcanic_ice_nk(TIME_ARR, cfg)
plt.plot(TIME_ARR/1e9, volc_ice_s_mass, label = 'S min')

mp.plot_version(cfg, xy=(0.8, 0.1))

plt.legend(loc='upper right')

# Inset plot
axins = plt.axes([0.6, 0.4, 0.3, 0.3])
axins.plot(TIME_ARR[100:325]/1e9, volc_ice_h20_mass[100:325])
axins.plot(TIME_ARR[100:325]/1e9, volc_ice_co_mass[100:325])
axins.plot(TIME_ARR[100:325]/1e9, volc_ice_s_mass[100:325])
axins.invert_xaxis()
axins.set_xlabel('Age [Ga]')
axins.set_ylabel('Volatile Mass [kg]')
axins.grid(False,which='both')

ax1.invert_xaxis()
ax1.set_xlabel('Age [Ga]')
ax1.set_ylabel('Volatile Mass [kg]')
ax1.set_title('Volcanic Volatile Outgassed Through Time')
plt.show()
fig1.savefig(out_dir + 'volc_time_w_inset.png')


# Solar wind volatile mass through time

fig2,ax2 = plt.subplots(figsize=(8, 5))

solar_wind_ice_fys = mp.solar_wind_ice(TIME_ARR, cfg)
cfg.faint_young_sun = False
solar_wind_ice_fys_off = mp.solar_wind_ice(TIME_ARR, cfg)

plt.plot(TIME_ARR/1e9, solar_wind_ice_fys, label='Faint Young Sun Scenario')
plt.plot(TIME_ARR/1e9,solar_wind_ice_fys_off, label='Constant Scenario')
ax2.set_xlabel('Age [Ga]')
ax2.set_ylabel('Volatile Mass [kg]')
ax2.set_title(r'Solar wind $\rm H_{2}O$ mass through time')
ax2.invert_xaxis()

mp.plot_version(cfg, xy = (.1,.8))
plt.legend()
plt.show()
fig2.savefig(out_dir + 'sol_wind.png')



# Solar wind volatile mass through time w/ volc mass plotted on top. 
fig3,ax3 = plt.subplots(figsize=(8, 5))

solar_wind_ice_fys = mp.solar_wind_ice(TIME_ARR, cfg)
cfg.faint_young_sun = False
solar_wind_ice_fys_off = mp.solar_wind_ice(TIME_ARR, cfg)

plt.plot(TIME_ARR/1e9, solar_wind_ice_fys, label='Faint Young Sun = True')
plt.plot(TIME_ARR/1e9,solar_wind_ice_fys_off, label='Faint Young Sun = False')
plt.plot(TIME_ARR/1e9, volc_ice_h20_mass, label = r'Volcanic $\rm H_{2}O$ min ')


ax3.set_xlabel('Age [Ga]')
ax3.set_ylabel('Volatile Mass [kg]')
ax3.set_title(r'Solar wind $\rm H_{2}O$ mass through time')
ax3.invert_xaxis()
mp.plot_version(cfg, xy=(0.8, 0.1))


plt.legend()
plt.show()
fig3.savefig(out_dir + 'sol_wind_w_volc.png')

