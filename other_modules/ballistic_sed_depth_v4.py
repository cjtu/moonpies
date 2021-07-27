import numpy as np

G_MOON = 1.624  # [m s^-2]
# G_MOON = 9.8
R_MOON = 1737 * 1e3  # [m]
# R_MOON = 6356 * 1e3
from moonpies import default_config
cfg = default_config.Cfg()


def get_secondary_diam(dist, a, b):
    """
    Returns secondary crater diameter given distance from primary crater and 
    regression parameters.

    Parameters
    ----------
    dist (num): Distance from crater center [m]
    a (num): regression parameter [km]
    b (num): regression parameter

    Return
    ------
    Secondary crater diameter [m]
    """
    return km2m(a * m2km(dist) ** b)

def m2km(x):
    """Convert meters to kilometers."""
    return x / 1000

def km2m(x):
    """Convert kilometers to meters."""
    return x * 1000

def sec_final_diam(diam, dist, cfg):
    '''
    Returns secondary crater diameter given diameter of the primary crater and range away from the crater center.
    This function uses equations for Kepler, Copernicus, or Orientale, depending on diameter of the crater. 
    Regression parameters from Singer et al. 2020

    Parameters
    ----------
    diam (num or array): diameter of primary crater [m]
    dist (num or array): distance away from primary crater center[m]

    Returns
    -------
    sec_final_diam [m]: final secondary crater diameter
    
    '''
    if cfg.kepler_regime[0] < diam < cfg.kepler_regime[1]:
        cdiam = cfg.kepler_diam
        a = cfg.kepler_a
        b = cfg.kepler_b
    elif cfg.copernicus_regime[0] < diam < cfg.copernicus_regime[1]:
        cdiam = cfg.copernicus_diam
        a = cfg.copernicus_a
        b = cfg.copernicus_b
    elif cfg.orentale_regime[0] < diam < cfg.orientale_regime[1]:
        cdiam = cfg.orientale_diam
        a = cfg.orientale_a
        b = cfg.orientale_b 
    else:
        raise ValueError('Diam not in range.')
    dist_norm = (dist / cdiam) * diam
    return get_secondary_diam(dist_norm, a, b)

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
    t_diams[diams >= ds2c] = (1 / gamma) * ( # These should be >= ???
        diams[diams >= ds2c] * ds2c ** eta
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

def ballistic_sed_depth(d, range, theta, depth_mode ):
    '''
    Returns the excavation depth of secondary craters being ballistically scoured 

    Parameters
    ----------
    d (num or array): diameter of primary crater [m]
    range (num or array): distance away from primary crater center [m]
    theta (num): angle of impact [radians]

    '''

    # Calcualte final secondary crater diameter:
    d_sec_final = sec_final_diam(d, range, cfg)
    # d_sec_final = sec_final_diam_copern(range, d, b=b)

    # Convert final diameter to transient
    d_sec_trans = final2transient(d_sec_final)

    # Calculate imapct velocity of ejecta from primary crater center on a great sphere
    v = ballistic_spherical(np.deg2rad(theta), range)

    # Calculate excavation depth from Xie et al. 2020
    d_ex = excav_depth(d_sec_trans/2,v)

    # Calculate effective excavation depth from Xie et al. 2020 
    d_eff = excav_depth_eff(d_sec_trans/2, 0, d_ex) 


    if depth_mode == 1:
        return d_sec_trans/2 * .2 # Singer et al. 2020 transient crater diameter to depth ratio
    elif depth_mode == 2:
        return d_sec_final/2 * .125 # Singer et al. 2020 final crater diameter to depth ratio
    else:
        return d_ex # Xie et al. 2020 transient crater diameter to depth ratio






# Plotting 
excav_depths = []
excav_depths_singer_trans = []
excav_depths_singer_final = []

copern_diam = 93000 
# final_crater_diam = 26000 # [m]
# final_crater_diam = 50000
final_crater_diam = 93000 # Cpernicus final diameter
x = np.arange((final_crater_diam/2/1000),1000)
for i in x:
    excav_depths.append(ballistic_sed_depth(final_crater_diam, i*1000, 45, 3))
    excav_depths_singer_trans.append(ballistic_sed_depth(final_crater_diam, i*1000, 45, 1))
    excav_depths_singer_final.append(ballistic_sed_depth(final_crater_diam, i*1000, 45, 2))


# print(excav_depths)

import matplotlib.pyplot as plt


fig,ax = plt.subplots()
ax.plot(x, excav_depths, label='Xie')
ax.plot(x, excav_depths_singer_trans,'k--', label='Singer trans: 0.2')
ax.plot(x, excav_depths_singer_final,'r--', label='Singer final: 0.125')

ax.set_xlabel('Distance from crater center [km]')
ax.set_ylabel('Excavation depth [m]')
ax.set_title('Excavation depth of secondary craters with distance from crater center\nPrimary crater diameter = 93 km')
bunte_points_x = [28.5, 25.5, 23, 19, 16.5, 27, 36.5, 35, 32, 32, 27]
bunte_points_y = [15, 52, 76, 34, 80, 47, 17, 28, 7, 21, 84]


# ax.plot(bunte_points_x, bunte_points_y, 'ro', label='Bunte Breccia depths [m]')

ax.legend()
plt.show()

# ballistic_sed_depth(final_crater_diam, i*1000, 45)