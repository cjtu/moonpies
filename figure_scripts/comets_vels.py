import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import moonpies as mp
import default_config

fig_dir = "/home/kristen/codes/code/moonpies_package/figs/"
data_dir = "/home/kristen/codes/code/moonpies_package/data/"

def poisson_distribution(k, lambd):
    return (lambd ** k * np.exp(-lambd)) / np.math.factorial(k)

def gaussian(x, alpha, r):
    return 1./(np.sqrt(2*alpha**np.pi))*np.exp(-np.power((x - r), 2.)/(2*alpha))

def test1(peak="poisson"):
    if peak == "gaussian":
        first = (np.random.normal(8, 6, 720))
    elif peak == "poisson":
        first = np.random.poisson(9, 360)
    second = (np.random.normal(23, 4, 200))
    third = (np.random.normal(55, 5, 80))
    comets = np.concatenate((first, second, third))
    return comets

def test2():
    x = np.linspace(0, 70, 70)
    y = 6000*gaussian(x, 5**2, 23) + 2000*gaussian(x, 5**2, 55) #+ 25000*gaussian(x, 6**2, 8) 
    #y[x<25] = 0
    for i in range(0,len(x)):
        y[i] += 500*(poisson_distribution(i, 12))
    return x, y

def get_comet_speeds(n, std = 5, mu1 = 12, mu2 = 23, mu3 = 55):
    x = np.linspace(0, 70, 70)
    y = 6000*gaussian(x, std**2, mu2) + 2000*gaussian(x, std**2, mu3) #+ 25000*gaussian(x, 6**2, 8) 
    #y[x<25] = 0
    for i in range(0,len(x)):
        y[i] += 500*(poisson_distribution(i, mu1))
    y = y / np.sum(y)
    rng = np.random.default_rng(23567)
    impactor_speeds = rng.choice(x, n, p=y)
    return impactor_speeds,y

def ong(n, mu2=20, mu3=54, std=5):
    x = np.linspace(0, 70, 500)
    y = 6000*gaussian(x, std**2, mu2) + 2000*gaussian(x, std**2, mu3) #+ 25000*gaussian(x, 6**2, 8) 
    y = y / np.sum(y)
    rng = np.random.default_rng(23567)
    impactor_speeds = rng.choice(x, n, p=y)
    return impactor_speeds, y

comet, ysample = get_comet_speeds(500)
speeds = np.linspace(0,70,500)
comets = test1(peak="poisson")
x, ytest = test2()
ong_speeds, ong_y = ong(500)

plt.figure(1)
plt.hist(comets, bins=70, histtype='step')
plt.xlabel("Velocity [km s$^{-1}$]")
plt.ylabel("Number of comets")
plt.title("Test Velocity Distribution of Comets")
plt.savefig(fig_dir+"comet_vels_gaussian_test.png")
plt.show()

plt.figure(2)
plt.plot(x, ytest)
plt.xlabel("Velocity [km s$^{-1}$]")
plt.ylabel("Number of comets")
plt.title("Velocity Distribution of Comets")
plt.savefig(fig_dir+"comet_vels_analytical_two_cutoff.png")
plt.show()

plt.figure(3)
plt.hist(comet, bins=50, histtype='step')
plt.xlabel("Velocity [km s$^{-1}$]")
plt.ylabel("Number of comets")
plt.title("Sampled Velocity Distribution of Comets")
plt.savefig(fig_dir+"comet_vels_gaussian_sample.png")
plt.show()

cfg = default_config.Cfg(mode='moonpies')
plt.figure(4, figsize=(6,4))
plt.plot(speeds, ong_y*1e3)
mp.plot_version(cfg, loc='ll')
plt.xlabel("Velocity [km s$^{-1}$]")
plt.ylabel("Number of Comets")
plt.title("Velocity Distribution of Comets")
plt.savefig(fig_dir+"comet_vels_ong.png")
plt.show()