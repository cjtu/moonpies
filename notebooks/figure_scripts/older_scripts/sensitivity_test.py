import moonpies as mp
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig_dir = "/home/kristen/codes/code/moonpies_package/fig/"
data_dir = "/home/kristen/codes/code/moonpies_package/data/"

def sens_test_options():
    out = []
    options = ['Head','NK']#[0.05, 0.1, 0.5]#[1,10,100]#[900, 934, 1000, 1100]#['2001','1983']#[3e3, 6e3, 9e3]#['cannon', 'moonpies']#[17e3, 20e3, 23e3, 50e3]#[1000, 1300, 1600, 1900, 2200, 2500, 2800]#
    for op in options:
        mp.clear_cache() 
        #print('Time step = ', mystep)
        cfg = config.Cfg(seed=7494, write=False, mode='moonpies', volc_mode=op)
        #print(int((cfg.timestart - cfg.timeend) / cfg.timestep) - (cfg.timestart - cfg.timeend) / cfg.timestep)
        out.append(mp.main(cfg))
        
    ej, ice, strat_dict = out[0]
    # stats on many strat column outputs
    cnames = ['Haworth', 'Shoemaker', 'Faustini', 'Shackleton', 'Slater', 
                'Amundsen', 'Cabeus', 'Sverdrup', 'de Gerlache', "Idel'son L", 
                'Wiechert J', 'Cabeus B']
    lats = np.array([87.5, 88., 87.1, 89.6, 88.1, 84.4, 85.3, 88.3, 88.3, 84., 85.2, 82.3])

    ymax=[]
    plt.figure(figsize=(6,4))
    for i, op in enumerate(options):
        means = []
        ej, ice, strat_dict = out[i]

        # Get and plot mean of all strat columns in strat_dict
        for key in strat_dict:
            df = strat_dict[key]
            means.append(np.sum(df.ice))
            ymax.append(np.max(means))
        d = {'names':cnames,'latitudes':lats, 'mean':means}
        plotting = pd.DataFrame(d)
        plotting.sort_values('latitudes')['mean'].plot(use_index=False, label="Volcanic Mode = "+str(op))#" m s$^{-1}$")
    plt.fill_between([0,4.5],[0,0],[np.max(ymax),np.max(ymax)],alpha=0.5,hatch='/', color='grey')
    plt.xlabel("Crater")
    plt.ylabel("Total Ice Thickness [m]")
    plt.xticks(np.linspace(0,11,12),plotting.sort_values('latitudes')['names'], rotation=45)
    plt.title("Sensitivity Test - Volcanic Reference")
    plt.legend()
    mp.plot_version(cfg, loc='ll')
    plt.savefig(fig_dir+"volc_mode_test.png", dpi=300)

def sens_test_mode():
    out = []
    states = [True, False]
    for state in states:
        mp.clear_cache() 
        #print('Time step = ', mystep)
        cfg = config.Cfg(seed=7494, write=False, mode='moonpies', impact_gardening_costello=state)#
        #print(int((cfg.timestart - cfg.timeend) / cfg.timestep) - (cfg.timestart - cfg.timeend) / cfg.timestep)
        out.append(mp.main(cfg))

    ej, ice, strat_dict = out[0]
    # stats on many strat column outputs
    cnames = ['Haworth', 'Shoemaker', 'Faustini', 'Shackleton', 'Slater', 
                'Amundsen', 'Cabeus', 'Sverdrup', 'de Gerlache', "Idel'son L", 
                'Wiechert J', 'Cabeus B']
    lats = np.array([87.5, 88., 87.1, 89.6, 88.1, 84.4, 85.3, 88.3, 88.3, 84., 85.2, 82.3])

    plt.figure(figsize=(6,4))
    ymax = []
    for i, state in enumerate(states):
        means = []
        ej, ice, strat_dict = out[i]
        # Get and plot mean of all strat columns in strat_dict
        for key in strat_dict:
            df = strat_dict[key]
            means.append(np.sum(df.ice))
            ymax.append(np.max(means))
        d = {'names':cnames,'latitudes':lats, 'mean':means}
        plotting = pd.DataFrame(d)
        plotting.sort_values('latitudes')['mean'].plot(use_index=False, label="Costello et al. (2020) = "+str(state))
    plt.fill_between([0,4.5],[0,0],[np.max(ymax),np.max(ymax)],alpha=0.5,hatch='/', color='grey')
    plt.xlabel("Crater")
    plt.ylabel("Total Ice Thickness [m]")
    plt.xticks(np.linspace(0,11,12),plotting.sort_values('latitudes')['names'], rotation=45)
    plt.title("Sensitivity Test - Costello et al. (2020) Impact Gardening")
    plt.legend()
    mp.plot_version(cfg, loc='ll')
    plt.savefig(fig_dir+"costello_test.png", dpi=300)  