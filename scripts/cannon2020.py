"""Modified from Cannon et al. (2020) Supplemental ds01.m and ds02.m"""
import numpy as np
import pandas as pd

# Constants
R_MOON = 1737  # [km]

diam_range = {
    # Regime: (rad_min, rad_max, step)
    'A': (0, 0.01, None),    # Micrometeorites (<1 mm)
    'B': (0.01, 3, 1e-4),    # Small impactors (1 mm - 1 m)
    'C': (100, 1.5e3, 1),    # Simple craters, steep branch (100 m - 1.5 km)
    'D': (1.5e3, 15e3, 1e2), # Simple craters, shallow branch (1.5 km - 15 km)
    'E': (15e3, 300e3, 1e3)  # Complex craters, shallow branch (15 km - 300 km)
}

sfd_slope = {
    'C': -3.82,
    'D': -1.8,
    'E': -1.8
}

# Functions
def diam2len(diams, speeds, regime):
    """
    Return size of impactor based on diam of crater.
    
    Regime C (Prieur et al., 2017)

    Regime D (Collins et al., 2005)

    Regime E (Johnson et al., 2016)
    """
    if regime == 'C':
        impactor_length = 0  # ToDo - Eq ??
    elif regime == 'D':
        impactor_length = 0  # ToDo - Eq 21?
    elif regime == 'E':
        impactor_length = 0  # ToDo - Eq 4?
    else:
        raise ValueError(f'Invalid regime {regime} in diam2len')
    return impactor_length


def mapAge(*args):
    return np.ones((nxy, nxy))

def ejS(n):
    return np.zeros((nxy, nxy))

def impact_ice(age, f24):
    return age

def ice_strat_model_cannon_2020(ejS, row, col, nxy=801, run_num=0):
    """
    Model the total ejecta and ice deposited in the South pole.


    """
    pct_area = 1.30e4
    hop_efficiency = 0.054  # From modified Kloos model with Hayne area - Cannon 2020
    timesteps = np.arange(425)[::-1] * 0.01  # Gyr, each step is 10 Myr
    
    # Read in Cannon 2020 crater list
    fcraters = '~/projects/ESSI_2021/data/cannon2020_crater_ages.csv'
    cols = ('name', 'lat', 'lon', 'diam', 'age', 'age_low', 'age_upp')
    df = pd.read_csv(fcraters, names=cols, header=0)


    x_scale = np.linspace(-400, 400, nxy)
    erosion_base = 0
    nt = len(timesteps)

    strat_ice_vul_x = np.zeros(nt)
    strat_ice_impact = np.zeros(nt)
    total_ice = np.zeros(nt)
    strat_ejecta_matrix = np.zeros((nxy, nxy, nt))
    total_ejecta_map = np.zeros((nxy, nxy))
    age_map = np.ones((nxy, nxy)) * 10  # age of youngest crater [Gyr]

    for t, timestep in enumerate(timesteps):
        age = round(timestep, 2)  # redundant?
        craters = df[df.age == age]

        # Ejecta step
        for i, crater in craters.iterrows():
            # Ejecta map (need to define ejS)
            crater_ejecta_map = ejS(crater)
            total_ejecta_map += crater_ejecta_map
            strat_ejecta_matrix[:, :, t] += crater_ejecta_map

            # Age map
            rad = 1000 * crater.age / 2  # [m]
            crater_age_map = mapAge(rad, crater.lat, crater.lon, nxy, x_scale, 'S', age)
            age_map[crater_age_map < age_map] = crater_age_map[crater_age_map < age_map]

        # Mass of Ice from volcanism [kg]
        ice_mass_vul_x = 0
        if 4 >= age >= 3.01:
            ice_mass_vul_x = 1e7*.75*3000*(1000**3)*(10/1e6)*(1e7/1e9)*hop_efficiency
        elif 3.01 > age > 2.01:
            ice_mass_vul_x = 1e7*.25*3000*(1000**3)*(10/1e6)*(1e7/1e9)*hop_efficiency

        strat_ice_vul_x[t] = ice_mass_vul_x
        
        # Mass of Ice from impacts (define impact_ice) [kg]
        ice_mass_impact = total_impact_ice(age) * hop_efficiency
        strat_ice_impact[t] = ice_mass_impact

        # Get ice thickness [m]
        total_ice_mass = ice_mass_vul_x + ice_mass_impact
        total_ice_vol = total_ice_mass / 934  # [m^3]
        total_ice[t] = total_ice_vol / (pct_area * 1e3 * 1e3)  # [m]

        if strat_ejecta_matrix[row, col, t] > 0.4:
            erosion_base = t

        layer = t
        ice_eroded = 0.1
        while ice_eroded > 0 and layer > erosion_base:
            if total_ice[layer] >= ice_eroded:
                total_ice[layer] = total_ice[t] - ice_eroded
                ice_eroded = 0
            else:
                ice_eroded = ice_eroded - total_ice[layer]
                total_ice[layer] = 0
                layer -= 1

    return total_ice, df.age.values, age_map, strat_ejecta_matrix


def latlon2xy(lat, lon, rp=R_MOON):
    """Return (x, y) from pole in units of rp from (lat, lon) [degrees]."""
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    y = rp * np.cos(lat) * np.cos(lon)
    x = rp * np.cos(lat) * np.sin(lon)
    return x, y


def probabalistic_round(x):
    """
    Randomly round positive float x up or down based on its distance to x + 1.

    E.g. 6.1 will round down ~90% of the time and round up ~10% of the time
    such that over many trials, the expected value is 6.1.
    """
    return int(x + np.random.random())


def impact_flux(t):
    """Return impact flux (Derivative of eqn. X in Ivanov)."""
    return 3.76992e-13 * (np.exp(6.93 * t)) + 8.38e-4


def neukum(diam, fit='1983'):
    """Return number of craters at diam (eqn. 2, Neukum 2001)."""
    a = {
        '1983': (-3.0768, -3.6269, 0.4366, 0.7935, 0.0865, -0.2649, -0.0664, 
                 0.0379, 0.0106, -0.0022, -5.18e-4, 3.97e-5),
        '2000': ()
    }
    log_n = 0
    for j, aj in enumerate(a[fit]):
        log_n += aj * np.log10(diam)**j
    return 10 ** log_n


def ice_micrometeorites(age):
    """
    Return ice from micrometeorites (Regime A, Cannon 2020).

    Grun et al. (2011) give 10**6 kg per year of < 1mm asteroid & comet grains.
    """
    # Multiply by years per timestep and assume 10% hydration
    scaling = 1e6 * 1e7 * 0.1
    return scaling * impact_flux(age) / impact_flux(0)


def ice_small_impactors(impactor_diams, impactors):
    """
    Return ice from small impactors (Regime B, Cannon 2020).

    Uses Jedicke et al. (2019) to compute water contents
    """
    # Calculate mass for each size bin
    impactor_masses = 1300*(4/3)*np.pi*(impactor_diams/2)**3
    # Multiply masses by number in each bin
    total_impactor_mass = sum(impactor_masses*impactors)
    # Averaged water contents (Jedicke et al. 2019)
    total_impactor_water = total_impactor_mass*0.36*(2/3)*0.1

    # Averaged retention from Ong et al., ballistic hopping
    return total_impactor_water * 0.165


def ice_small_craters(crater_diams, craters):
    """
    Return ice from simple craters, steep branch (Regime C, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams*1000, 20, regime)
    impactor_masses = 1300 * (4/3) * np.pi * (impactor_diams/2)**3
    total_impactor_mass = sum(impactor_masses * craters)
    total_impactor_water = total_impactor_mass * 0.36 * (2/3) * 0.1

    # Averaged retention from Ong et al., ballistic hopping
    return total_impactor_water * 0.165


def ice_large_craters(crater_diams, regime):
    """
    Return ice from simple craters, shallow branch (Regime D, Cannon 2020).
    """
    impactor_speeds = np.random.normal(20, 6, len(crater_diams))
    impactor_speeds[impactor_speeds < 2.38] = 2.38 # minimum is Vesc
    impactor_diams = diam2len(crater_diams*1000, impactor_speeds, regime)
    impactor_masses = 1300*(4/3)*np.pi*(impactor_diams/2)**3

    water_retained = np.ones(len(impactor_speeds)) * 0.5
    water_retained[impactor_speeds >= 10] = 36.26*np.exp(-0.3464*impactor_speeds)
    water_retained[water_retained < 0] = 0

    # Assuming 10% hydration
    water_masses = impactor_masses * water_retained * 0.1

    # This is a direct copy of the water array in Cannon 2020 - maybe there 
    # used to be a condieration for mass of water released vs ice mass?
    ice_masses = water_masses
    return sum(ice_masses)


def impact_ice_regime(age, regime):
    """Return ice at age from impactors in regime A - E (Cannon, 2020)."""
    if regime == 'A':
        # Micrometeorites
        return ice_micrometeorites(age)
    elif regime == 'B':
        # Small impactors
        impactor_diams, impactors = get_impactor_pop(regime)
        return ice_small_impactors(age, impactor_diams, impactors)
    elif regime == 'C':
        # Small simple craters (continuous)
        crater_diams, craters = get_crater_pop(age, regime)
        return ice_small_craters(crater_diams, craters, regime)
    else:
        # Large simple / complex craters (stochastic)
        crater_diams = get_crater_pop(age, regime)
        return ice_large_craters(crater_diams, regime)



def get_diam_array(regime):
    """Return array of diameters based on diameters in diam_range."""
    dmin, dmax, step = diam_range[regime]
    n = int((dmax - dmin) / step)
    return np.linspace(dmin, dmax, n + 1)


def get_impactor_pop(age, regime):
    """
    Return population of impactors and number in regime B.

    Use constants and eqn. 3 from Brown et al. (2002) to compute N craters. 
    """
    impactor_diams = get_diam_array(regime)

    # Get number of impactors given min and max diam (Brown et al. 2002)
    c = 1.568 # constant (Brown et al. 2002)
    d = 2.7 # constant (Brown et al. 2002)
    n_impactors_gt_low = 10**(c - d * np.log10(impactor_diams[0]))  # [yr^-1]
    n_impactors_gt_high = 10**(c - d * np.log10(impactor_diams[-1]))  # [yr^-1]
    n_impactors = n_impactors_gt_low - n_impactors_gt_high
    
    # Scale for timestep, Earth-Moon ratio (Mazrouei 2019) and impact flux
    n_impactors *= 1e7 * (1 / 22.5) * impact_flux(age) / impact_flux(0)
    
    # Scale by size-frequency distribution
    sfd = impactor_diams**sfd_slope[regime]
    impactors = sfd * (n_impactors / sum(sfd))

    return impactor_diams, impactors


def get_crater_pop(age, regime):
    """
    Return population of crater diameters and number (regimes C - E).

    Weight small simple craters by size-frequency distribution.

    Randomly resample large simple & complex crater diameters.
    """
    crater_diams = get_diam_array(regime)
    n_craters = neukum(crater_diams[0]) - neukum(crater_diams[-1])
    # Scale for timestep, surface area and impact flux
    n_craters *= (1e7/1e9) * 3.79e7 * impact_flux(age) / impact_flux(0)
    sfd = crater_diams ** sfd_slope[regime]
    if regime == 'C':
        # Steep branch of sfd (simple)
        craters = sfd * (n_craters / sum(sfd))
        return crater_diams, craters

    # Regimes D and E: shallow branch of sfd (simple / complex)
    n_craters = probabalistic_round(n_craters)

    # Resample crater diameters with replacement, weighted by sfd
    crater_diams = np.random.choice(crater_diams, n_craters, p=sfd)
    
    # Randomly include only hydrated carbonaceous impacts
    # Assume 36 % are C-type (Jedicke et al., 2018)
    # Assume 2/3 are hydrated (Rivkin, 2012)
    rand_arr = np.random.rand(len(crater_diams))
    crater_diams[rand_arr >= 0.36 * (2/3)] = np.nan
    
    return crater_diams


def total_impact_ice(age):
    """Return total impact ice from regimes A-E (Cannon 2020)."""
    total_ice = 0
    for regime in ('A', 'B', 'C', 'D', 'E'):
        total_ice += impact_ice_regime(age, regime)
    return total_ice
