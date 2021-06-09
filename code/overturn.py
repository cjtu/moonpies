import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lambda values at P=(10%, 50%, 99%) from Table 1 (Costello et al. 2018)
d = {
    'n': [1, 2, 3, 4, 6, 8, 10, 20, 30, 40, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 
          3e5, 1e6],
    'L10': [0.105, 0.530, 1.102, 1.742, 3.150, 4.655, 6.221, 14.53, 23.33, 
            32.11, 87.42, 2.78e2, 9.596e2, 2.93e3, 9.872e3, 2.978e4, 9.959e4, 
            2.993e5, 9.9687e5],
    'L50': [0.693, 1.678, 2.674, 3.672, 5.67, 7.67, 9.67, 19.67, 29.67, 39.67, 
            99.67, 2.997e2, 9.997e2, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6],
    'L99': [4.605, 6.638, 8.406, 10.05, 13.11, 16, 18.87, 31.85, 44.19, 56.16, 
            1.247e2, 3.418e2, 1.075e3, 3.129e3, 1.023e4, 3.041e4, 1.007e5, 
            3.013e5, 1.002e6]
}
df = pd.DataFrame(d)

# Model parameters from Table 2 (Costello et al. 2018)
c = 0.41
d = 1/3
g = 1.61e15

# Parameters specific to lunar primary craters
a = 6.3e-11  #
b = -2.7
vf = 5.7e11
rhom = 2500
K1 = 0.132
K2 = 0.26
Kr = 1.1
Kd = 0.6
mu = 0.41
Y = 0.01
rhot = 1500

# Defined variables from Table 3 (Costello et al. 2018)
alpha = K2 * (( (Y/rhot)*vf**2 )**(0.5*(2+mu)))
beta = (-3*mu)/(2+mu)
gamma = (K1*np.pi*rhom)/(6*rhot)
di = 2 * Kr
eps = (g/(2*vf**2))*(rhom/rhot)**(1/3)

def overturn_strength(lam, t):
    """Return reworking depth given lambda and time (strength regime)"""
    t1 = d * Kd/(2*Kr)
    t2_num = 4 * lam * (di*(gamma*alpha**beta)**(1/3))**b
    t2_den = a * (c**2) * np.pi * t
    exp = 1 / (b + 1)
    return t1 * (t2_num/t2_den)**exp

times = (1e2, 1e6, 1e9)  # [yrs]
for t, og in enumerate(('os100', 'os1M', 'os1G')):
    for lam in ('L10', 'L50', 'L99'):
        df[og+lam] = 10*overturn_strength(df[lam], times[t])

# Plot Figure 5 from Costello et al. (2018)
f, axs = plt.subplots(1, 2, figsize=(10, 5))

# Figure 5 a)
axs[0].set_title('Number of Turns as a function of $\lambda$')
axs[0].set_ylabel('Cumulative Number of Turns (n)')
axs[0].set_xlabel('Average Number of Events per Interval ($\lambda$)')
axs[0].loglog(df['L10'], df['n'], 'w:', label='10%')
axs[0].loglog(df['L50'], df['n'], 'w--', label='50%')
axs[0].loglog(df['L99'], df['n'], 'w-', label='99%')
axs[0].set_ylim(1e0, 1e4)
axs[0].set_xlim(0.5, 1e4)
axs[0].legend()


# Figure 5 b)
axs[1].set_title('Number of Turns as a function of reworking depth')
axs[1].set_ylabel('Cumulative Number of Turns (n)')
axs[1].set_xlabel('Reworking Depth (cm)')
axs[1].loglog(df['og100L10'], df['n'], 'c:', label='10%')
axs[1].loglog(df['og1ML10'], df['n'], 'g:', label='_nolegend_')
axs[1].loglog(df['og1GL10'], df['n'], 'm:', label='_nolegend_')
axs[1].loglog(df['og100L50'], df['n'], 'c--', label='50%')
axs[1].loglog(df['og1ML50'], df['n'], 'g--', label='_nolegend_')
axs[1].loglog(df['og1GL50'], df['n'], 'm--', label='_nolegend_')
axs[1].loglog(df['og100L99'], df['n'], 'c-', label='99%')
axs[1].loglog(df['og1ML99'], df['n'], 'g-', label='_nolegend_')
axs[1].loglog(df['og1GL99'], df['n'], 'm-', label='_nolegend_')
axs[1].set_ylim(1e0, 1e4)
axs[1].set_xlim(0.5e-4, 1.5e2)
axs[1].xaxis.set_ticks([1e-4, 1e-2, 1e0, 1e2])
l = axs[1].legend()
l.legendHandles[0].set_color('w')
l.legendHandles[1].set_color('w')
l.legendHandles[2].set_color('w')

plt.savefig('overturn.png', dpi=600)
