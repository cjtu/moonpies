#DOES NOT WORK WITH CURRENT MOONPIES
from moonpies import moonpies as mm
from moonpies import config
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    'figure.figsize': (6, 4),
    'figure.facecolor': 'white',
    'xtick.top': True,
    'xtick.direction': 'in',
    'ytick.right': True,
    'ytick.direction': 'in',
    'axes.grid': True
})
cfg = config.Cfg()

mm.plot_version(cfg,loc='ul')
# See figure 4 Costello (2020)
plt.figure()
time_arr = np.logspace(3, 10)

# Apollo sample control group (plotted as line rather than points)
od_morris = mm.overturn_depth_morris(time_arr)
plt.loglog(time_arr, od_morris, '--', label='Morris 1978 Apollo')

# LRO observation control group (plotted as line rather than points)
od_speyerer = mm.overturn_depth_speyerer(time_arr)
plt.loglog(time_arr, od_speyerer, '--', label='Speyerer 2016 LROC')

# Costello reported strength regime equation
od_costello_str = mm.overturn_depth_costello_str(time_arr)
plt.loglog(time_arr, od_costello_str, '--', label='Costello 2020 (n=1)')

# Set up overturn for secondaries, 99% probability
prob_pct = '99%'
regime = 'secondary'
a, b = cfg.overturn_ab[regime]
vf = cfg.impact_speeds[regime]
u = mm.overturn_u(
            a,
            b,
            "strength",
            vf,
            cfg.target_density,
            cfg.impactor_density_avg,
            cfg.target_kr,
            cfg.target_k1,
            cfg.target_k2,
            cfg.target_mu,
            cfg.target_yield_str,
            cfg.grav_moon,
            cfg.impact_angle)

# Plot 1, 10, 100 overturns
for n_overturns, c in zip((1, 10, 100), ('r', 'orange', 'c')):
    depth = mm.get_overturn_depth(
            u,
            b,
            cfg.costello_csv_in,
            time_arr,
            n_overturns,
            prob_pct)
    plt.loglog(time_arr, depth, c=c, label=f'Secondaries (n={n_overturns})')

plt.ylim(10, 1e-3)
plt.xlim(1e5, 2e9)
plt.xlabel('Time elapsed [yr]')
plt.ylabel('Depth overturned [m]')
plt.legend()

plt.savefig('/home/cjtu/projects/moonpies/figs/overturn_number_test.png', 
            bbox_inches='tight', dpi=300)
plt.show()

# Test Equation 14, Costello (2020)
plt.figure()
time_arr = np.logspace(3, 10)

# Costello reported strength and gravity regime equations
od_costello_str = mm.overturn_depth_costello_str(time_arr)
plt.loglog(time_arr, od_costello_str, '--', label='Costello 2020 (strength)')
od_costello_grav = mm.overturn_depth_costello_grav(time_arr)
plt.loglog(time_arr, od_costello_grav, '--', label='Costello 2020 (gravity)')

# Set up overturn for secondaries, 1 overturn, 99% probability
n_overturns = 1
prob_pct = '99%'
regime = 'secondary'
a, b = cfg.overturn_ab[regime]
vf = cfg.impact_speeds[regime]


# Plot strength and gravity regimes
for regime in ('strength', 'gravity'):
    u = mm.overturn_u(
            a,
            b,
            "strength",
            vf,
            cfg.target_density,
            cfg.impactor_density_avg,
            cfg.target_kr,
            cfg.target_k1,
            cfg.target_k2,
            cfg.target_mu,
            cfg.target_yield_str,
            cfg.grav_moon,
            cfg.impact_angle)
    depth = mm.get_overturn_depth(
            u,
            b,
            cfg.costello_csv_in,
            time_arr,
            n_overturns,
            prob_pct)
    plt.loglog(time_arr, depth, label=f'Our model ({regime})')

plt.ylim(10, 1e-3)
plt.xlim(1e5, 2e9)
plt.xlabel('Time elapsed [yr]')
plt.ylabel('Depth overturned [m]')
plt.legend()

plt.savefig('/home/cjtu/projects/moonpies/figs/overturn_regime_test.png', 
            bbox_inches='tight', dpi=300)
plt.show()

time_arr = np.linspace(cfg.timestart, 0)
new_cfg = config.Cfg()
new_cfg.overturn_prob_pct = '99%'

for n in (100, 10, 1):
    new_cfg.n_overturn = n
    od = mm.total_overturn_depth(time_arr, new_cfg)
    plt.semilogy(time_arr, od, label=f'99% prob of {n} overturn(s)')
plt.title('Total impact gardening depth over time')
plt.ylim(100, 1e-2)
plt.xlim(4.3e9, 0)
plt.ylabel('Depth [m]')
plt.xlabel('Time [Ga]')
plt.legend()
plt.savefig('/home/cjtu/projects/moonpies/figs/total_overturn_depth.png', 
            bbox_inches='tight', dpi=300)
plt.show()

# Table 1 from Costello et al. (2018)
df = mm.read_lambda_table(cfg.costello_csv_in)
df.plot(x='n', logx=True, logy=True)
plt.ylabel('$\lambda$')
plt.title('Table 1 Costello et al. 2018')
df.head()
