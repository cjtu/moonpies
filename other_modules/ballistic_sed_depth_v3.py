import numpy as np

G_MOON = 1.624  # [m s^-2]
# G_MOON = 9.8
R_MOON = 1737 * 1e3  # [m]
# R_MOON = 6356 * 1e3
M = 1.96 # ± 0.15
C = 0.022
E = 0.00 # ± 0.09
N = -0.14 # ± 0.03


def m2km(x):
    '''
    It converts m to km. 
    '''
    return x/1000

def sec_final_diam(d, range, M=1.96, C=0.022, E=0.00, N=-0.14):
    '''
    Returns secondary crater diameter given diameter of the primary crater and range away from the crater center. This function uses the generalized equation from Singer et al. 2020

    Parameters
    ----------
    d (num or array): diameter of primary crater [m]
    range (num or array): distance away from primary crater center[m]

    Returns
    -------
    sec_final_diam [m]: final secondary crater diameter
    
    '''

    d = m2km(d)
    range = m2km(range)

    a = C * (d**M)
    b = E - (np.log(d) * N)
    return ((a * (range**-b)) * 1000)

def sec_final_diam_copern(range, d, d_copern=93000, M = 1.96, a = 1.2e2, b = -.68):
    '''
    Returns secondary crater diameter given diameter of the primary crater and range away from the crater center. This function uses the Copernius derived equation from Singer et al. 2020
    The equation is then scaled by the primary crater. 

    Parameters
    ----------
    d (num or array): diameter of primary crater [m]
    range (num or array): distance away from primary crater center[m]

    Returns
    -------
    sec_final_diam [m]: final secondary crater diameter
    
    '''
    range = m2km(range)
    d = m2km(d)
    d_copern = m2km(d_copern)
    return ((a * (d/d_copern)**M * (range**(b*(-np.log(d/d_copern))))) * 1000)

def final2transient(diams, g=G_MOON, ds2c=18e3, gamma=1.25, eta=0.13):
    """
    Return transient crater diameters from final crater diams (Melosh 1989).

    Parameters
    ----------
    diams (num or array): final crater diameters [m]
    g (num): gravitational force of the target body [m s^-2]
    rho_t (num): target density (kg m^-3)

    Returns
    -------
    transient_diams (num or array): transient crater diameters [m]
    """
    # Scale simple to complex diameter (only if target is not Moon)
    # ds2c = simple2complex_diam(g)  # [m]
    diams = np.atleast_1d(diams)
    # diams < simple2complex == diam/gamma, else use complex scaling
    t_diams = np.copy(diams) / gamma
    t_diams[diams > ds2c] = (1 / gamma) * (
        diams[diams > ds2c] * ds2c ** eta
    ) ** (1 / (1 + eta))
    return t_diams

def ballistic_spherical(theta, d, g=G_MOON, rp=R_MOON):
    """
    Return ballistic speed (v) given ballistic range (d) and gravity of planet (g).
    Assumes perfectly spherical planet (Vickery, 1986).
    
    Parameters
    ----------
    d (num or array): ballistic range [m]
    g (num): gravitational force of the target body [m s^-2]
    theta (num): angle of impact [radians]
    rp (num): radius of the target body [m]
    
    Returns
    ------_
    v (num or array): ballistic speed [m s^-1]   
 
    """
    tan_phi = np.tan(d / (2 * rp))
    return np.sqrt((g * rp * tan_phi) / ((np.sin(theta) * np.cos(theta)) + (np.cos(theta)**2 * tan_phi)))



def excav_depth(R_at, v):
    '''
    Returns the excavation depth of a secondary crater

    Parameters
    ----------
    R_at (num): secondary transient apparent crater radius [m]
    v (num): incoming impactor velocity [m/s]

    Returns
    -------
    excav_depth (num): Excavation depth of a secondary crater [m]
    '''
    return .0134 * R_at * (v**0.38)

def excav_depth_eff(R_at, r, d_ex, C_ex = 3.5):
    '''
    Returns the effective excavation depth of a secondary crater at radius r away from center of secondary crater

    Parameters
    ----------
    R_at (num): secondary transient apparent crater radius [m]
    r (num): Distance from secondary crater center [m]

    Returns
    -------
    excav_depth (num): Effective excavation depth of a secondary crater [m]????
    '''
    if r <= R_at:
        return C_ex * d_ex * (1 - (r/R_at)**2)
    else:
        # Raise some error here
        return None 

def ballistic_sed_depth(d, range, theta, N_E_bound = 'std', b_val = 'std' ):
    '''
    Returns the excavation depth of secondary craters being ballistically scoured 

    Parameters
    ----------
    d (num or array): diameter of primary crater [m]
    range (num or array): distance away from primary crater center [m]
    theta (num): angle of impact [radians]

    '''
    # Choose which value within distribution to be used based on error bars. 
    if N_E_bound == 'low_N_high_E_bound':
        N = -0.17
        E = 0.09
    elif N_E_bound == 'high_N_low_E_bound':
        N = -0.11
        E = -0.09
    elif N_E_bound == 'std':
        N = -0.14 
        E = 0.00 

    if b_val == 'std':
        b = -.68
    elif b_val == 'low':
        b = -.74
    elif b_val =='high':
        b = -.62


    # Calcualte final secondary crater diameter:

    # d_sec_final = sec_final_diam(d, range, N=N, E=E)
    d_sec_final = sec_final_diam_copern(range, d, b=b)

    # Convert final diameter to transient
    d_sec_trans = final2transient(d_sec_final)


    v = ballistic_spherical(np.deg2rad(theta), range)

    d_ex = excav_depth(d_sec_trans/2,v)

    d_eff = excav_depth_eff(d_sec_trans/2, 0, d_ex) 

    # Multiply secondary transient crater diameter by .14 (Diameter/depth ratio from Singer et al. 2020)
    # return d_sec_trans * .14
    return d_eff



# Plotting 
excav_depths = []
excav_depths2 = []
excav_depths3 = []

copern_diam = 93000 
final_crater_diam = 26000 # [m]
# final_crater_diam = 50000
# final_crater_diam = 93000 # Orientale final diameter
x = np.arange((final_crater_diam/2/1000),60)
for i in x:
    excav_depths.append(ballistic_sed_depth(final_crater_diam, i*1000, 45))
    excav_depths2.append(ballistic_sed_depth(final_crater_diam, i*1000, 45, 'low_N_high_E_bound', 'low'))
    excav_depths3.append(ballistic_sed_depth(final_crater_diam, i*1000, 45, 'high_N_low_E_bound', 'high'))

# print(excav_depths)

import matplotlib.pyplot as plt


fig,ax = plt.subplots()
ax.plot(x, excav_depths, label='Depth curve (based off data from Singer et al. 2020)')
ax.plot(x, excav_depths2,'k--', label='Error bounds')
ax.plot(x, excav_depths3,'k--')

ax.set_xlabel('Distance from crater center [km]')
ax.set_ylabel('Excavation depth [m]')
ax.set_title('Excavation depth of secondary craters with distance from crater center\nPrimary crater diameter = 26 km')
bunte_points_x = [28.5, 25.5, 23, 19, 16.5, 27, 36.5, 35, 32, 32, 27]
bunte_points_y = [15, 52, 76, 34, 80, 47, 17, 28, 7, 21, 84]


ax.plot(bunte_points_x, bunte_points_y, 'ro', label='Bunte Breccia depths [m]')

ax.legend()
plt.show()

# ballistic_sed_depth(final_crater_diam, i*1000, 45)