"""
# From Jupyter, make sure to os.chdir to location of this file:
    use import mixing to run any function in this file e.g. mixing.main()

import os
os.chdir('/home/cjtu/projects/moonpies/moonpies')
import mixing as mm
ej_cols, ice_cols, run_meta = mixing.main()

# Run with gnuparallel (6 s/run normal, 0.35 s/run 48 cores)
    parallel -P-1 uses all cores except 1

conda activate moonpies
seq 10000 | parallel -P-1 python mixing.py

# Code Profiling (pip install snakeviz)

python -m cProfile -o .model.prof moonpies.py 1
snakeviz .model.prof

time poetry run python -m cProfile -o mst_210709.prof make_shade_table.py
snakeviz mst_210709.prof
"""
import numpy as np
import pandas as pd
import default_config
CFG = default_config.Cfg()

"""
Config
    # Compute depths of secondary craters using singer or xie
    secondary_depth_mode: str = 'singer'  # ['singer', 'xie']
    secondary_depth_eff: float = True  # Convert secondary max depth to effective depth (Equation 17, Xie et al., 2020)
    sec_depth_eff_rad_frac: float = 0.5 # [0-1 crater radii] Distance from secondary to compute depth (Equation 17, Xie et al., 2020)
    sec_depth_eff_c_ex: float = 3.5  # Equation 17

    ## Singer mode - compute secondary crater diam from observed secondaries
    depth_to_diam_sec = 0.125  # Value used in Singer et al. (2020)

    ## Xie mode - compute secondary crater depth from ballistic velocity
    ## Equations from (Xie et al., 2020)
    xie_a_simple: float = 0.0094  # pre-exponential term for simple secondaries, Equation 15 
    xie_a_complex: float = 0.0134  # pre-exponential term for complex secondaries, Equation 16  
    xie_b: float = 0.38  # velocity exponential, Equation 14
"""

# Old treatment of ballistic sedimentatino depth using Singer / Xie secondaries
def ballistic_sed_depths_time(time_arr, df, cfg=CFG):
    """
    Return ballistic sedimentation depth vs time for each coldtrap.
    """
    if not cfg.ballistic_sed:
        return np.zeros((len(time_arr), len(cfg.coldtrap_names)), cfg.dtype)
    dist = get_coldtrap_dists(df, cfg)
    diam = df.rad.values[:, np.newaxis] * 2  # Diameter column vector

    # Ballistic sed depth from all craters to all coldtraps
    bsed_depths = ballistic_sed_depth(dist, diam, cfg)

    # Check if ballistic sed happend with teq
    # if cfg.ballistic_teq:
    #     bsed_depths = check_teq_ballistic_sed(bsed_depths, cfg)
    
    # Scale ballistic sed by volume fraction deposited
    vol_frac = get_volume_frac(dist, diam / 2, cfg)
    bsed_depths *= vol_frac   
   
    # Convert to time array shape: (Ncrater, Ncoldtrap) -> (Ntime, Ncoldtrap)
    ages = df.age.values
    bsed_depth_t = ages2time(time_arr, ages, bsed_depths, np.nanmax, 0)
    return bsed_depth_t


def ballistic_sed_depth(dist, diam, cfg=CFG):
    """
    Return ballistic sedimentation mixing depths for each crater.
    """
    # Get secondary crater (distances within primarty diameter excluded)
    secondary_diam_vec = np.vectorize(secondary_diam)
    sec_diam_dist = secondary_diam_vec(diam, dist, cfg)
    if cfg.secondary_depth_mode == "singer":
        depth = secondary_depth_singer(sec_diam_dist, cfg)
    elif cfg.secondary_depth_mode == "xie":
        depth = secondary_depth_xie(sec_diam_dist, dist, cfg)
    else:
        msg = f"Invalid secondary depth mode {cfg.secondary_depth_mode}."
        raise ValueError(msg)

    # Convert to effective depth mixed by ballistic sed
    if cfg.secondary_depth_eff:
        # Convert to effective depth (Xie et al. 2020)
        # TODO: covert to secondary transient radius?
        depth_eff = secondary_excavation_depth_eff(depth, sec_diam_dist/2, cfg)
    else:
        # Excavation = 0.5*crater depth (Melosh 1989 via Singer et al. 2020)
        depth_eff = depth / 2
    return depth_eff


def get_volume_frac(ej_distances, rad, cfg=CFG):
    """
    Return volume fraction of target/ejecta material in [0, 1]. 
    
    Ex. 0.9 = 90% target, 10% ejecta; 0.5 = 50% target, 50% ejecta.
    
    Parameters
    ----------
    ej_distances (2D array): distances between each crater and each cold trap [m]
    rad (1D array): radius of each crater [m]
    
    Returns
    -------
    vf (2D array): volume fraction of ballistic sedimentation mixing region for
        each crater into each cold trap [fraction]
    """
    vf_a, vf_b = cfg.ballistic_sed_vf_a, cfg.ballistic_sed_vf_b
    volume_frac = np.zeros((len(rad), len(cfg.coldtrap_names)), cfg.dtype)
    for i in range(0,len(rad)):
        for j in range(0, len(cfg.coldtrap_names)):
            # TODO: bug? do a, b agree with units of rad? [m]
            volume_frac[i,j] = vf_a * (ej_distances[i,j] / rad[i])**vf_b
    return volume_frac


# Singer mode (cfg.secondary_crater_mode == 'singer')
def secondary_depth_singer(diam_secondary, cfg=CFG):
    """
    Returns excavation depth of ballistically sedimentation from secondary
    depth (Singer et al. 2020).

    Parameters
    ----------
    diam_secondary (num or array): diameter of secondary crater [m]
    """
    # Convert secondary diameter to depth with depth to diam
    depth_sec_final = diam_secondary * cfg.depth_to_diam_sec
    return depth_sec_final


def secondary_diam(diam, dist, cfg=CFG):
    """
    Return secondary crater diameter given diam of primary crater and dist from
    primary.

    Uses secondary diam vs dist fits for Kepler, Copernicus and Orientale
    from Singer et al. 2020 and regimes set in cfg.

    Parameters
    ----------
    diam (num or array): diameter of primary crater [m]
    dist (num or array): distance away from primary crater center[m]
    cfg (config): Config object

    Returns
    -------
    sec_final_diam [m]: final secondary crater diameter
    """
    if cfg.kepler_regime[0] < diam < cfg.kepler_regime[1]:
        cdiam = cfg.kepler_diam
        a = cfg.kepler_a
        b = cfg.kepler_b
    elif cfg.copernicus_regime[0] < diam < cfg.copernicus_regime[1]:
        cdiam = cfg.copernicus_diam
        a = cfg.copernicus_a
        b = cfg.copernicus_b
    elif cfg.orientale_regime[0] < diam < cfg.orientale_regime[1]:
        cdiam = cfg.orientale_diam
        a = cfg.orientale_a
        b = cfg.orientale_b
    else:
        raise ValueError("Diam not in range.")
    dist_norm = (dist / cdiam) * diam

    out = secondary_diam_at_dist(dist_norm, a, b)
    return out


def secondary_diam_at_dist(dist, a, b):
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

# Xie mode (cfg.secondary_crater_mode == 'xie')
def secondary_depth_xie(diam_secondary, dist, cfg=CFG):
    """
    Returns the depth of a secondary crater (Xie et al. 2020).

    Parameters
    ----------
    diam_secondary (num or array): diameter of secondary crater [m]
    dist (num or array): distance from primary crater center [m]
    cfg (Config): config object
    """
    # Convert final diameter to transient
    t_rad = final2transient(diam_secondary) / 2
    v = ballistic_speed(dist, cfg)

    # Compute secondary depth from transient radius and speed
    depth = np.zeros_like(t_rad)
    ixs = diam_secondary < cfg.simple2complex
    depth[ixs] = cfg.xie_a_simple * t_rad[ixs] * (v[ixs] ** cfg.xie_b)
    depth[~ixs] = cfg.xie_a_complex * t_rad[~ixs] * (v[~ixs] ** cfg.xie_b)
    return depth


def secondary_excavation_depth_eff(depth, t_rad, cfg=CFG):
    """
    Return the effective excavation depth of a secondary crater at radius r
    from center of secondary crater.

    Parameters
    ----------
    depth (num): Secondary crater depth [m]
    t_rad (num): Secondary transient crater radius [m]
    cfg (Config): Config object must contain xie_sec_radial_frac
        xie_sec_radial_frac (num): Fraction of t_rad to compute effective depth
            at (in range [0, 1])

    Returns
    -------
    excav_depth (num): Effective excavation depth of a secondary crater [m]????
    """
    if cfg.sec_depth_eff_rad_frac > 1:
        raise ValueError("Config: sec_depth_eff_rad_frac must be in range [0, 1]")
    radial_dist = t_rad * cfg.sec_depth_eff_rad_frac
    return cfg.sec_depth_eff_c_ex * depth * (1 - (radial_dist / t_rad) ** 2)


def ballistic_speed(dist, cfg=CFG):
    """
    Return ballistic speed at given dist, angle of impact theta gravity and
    radius of planet.

    Assumes planet is spherical (Vickery, 1986).

    Parameters
    ----------
    dist (num or array): ballistic range [m]
    theta (num): angle of impact [degrees]
    g (num): gravitational force of the target body [m s^-2]
    rp (num): radius of the target body [m]

    Returns
    -------
    speed (num or array): ballistic speed [m s^-1]
    """
    theta = np.deg2rad(cfg.impact_angle)
    tan_phi = np.tan(dist / (2 * cfg.rad_moon))
    numer = cfg.grav_moon * cfg.rad_moon * tan_phi
    denom = (np.sin(theta) * np.cos(theta)) + (np.cos(theta) ** 2 * tan_phi)
    return np.sqrt(numer / denom)


# Old method of determining if ballistic sed is applied or not
def check_teq_ballistic_sed(ballistic_depths, cfg=CFG):
    """
    Check whether ballistic sedimentation has energy to melt local ice.

    See thermal_eq.py for more details.
    """
    teq_df = read_teqs(cfg.teq_csv_in)
    # Find all craters with final eq temp high enough to melt local ice
    for i, cname_src in enumerate(teq_df.index):
        for j, cname_dst in enumerate(cfg.coldtrap_names):
            try:
                if teq_df.loc[cname_src, cname_dst] < cfg.coldtrap_max_temp:
                    ballistic_depths[i, j] = 0
            except KeyError:
                ballistic_depths[i, j] = 0
    return ballistic_depths


def read_teqs(teq_csv):
    """
    Read equilibrium temperatures for ballistic sed module.

    See thermal_eq.py.
    """
    df = pd.read_csv(teq_csv, header=0, index_col=0).T
    return df


# Potter scaling for basins (needs debugging)
def diam2len_potter(
    t_diam, v=20e3, rho_i=1300, rho_t=1500, g=1.62, dtype=None
):
    """
    Return impactor length from final crater diam using Potter et al. (2015)
    method.

    Note: Interpolates impactor lengths from the forward Potter impactor length
    to transient crater diameter equation.

    Parameters
    ----------
    diam (num or array): crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)

    Returns
    -------
    impactor_length (num): impactor diameter [m]
    """
    i_lengths = np.linspace(100, 1e6, 1000, dtype=dtype)
    pi2 = 3.22 * g * (i_lengths / 2) / v ** 2
    piD = 1.6 * pi2 ** -0.22
    t_diams = (
        piD
        * ((rho_i * (4 / 3) * np.pi * (i_lengths / 2) ** 3) / rho_t) ** 0.33
    )
    impactor_length = np.interp(t_diam, t_diams, i_lengths)
    return impactor_length



# Scale crater regimes by gravity of other bodies (not needed we only model the Moon)
def simple2complex_diam(
    gravity,
    density=2700,
    s2c_moon=18e3,
    g_moon=1.62,
    rho_moon=2700,
):
    """
    Return simple-to-complex transition diameter given gravity of body [m s^-2]
    and density of target [kg m^-3] (Melosh 1989).

    Parameters
    ----------
    gravity [num] : Gravity of a planetary body [m s^-2]
    density [num] : Density of the target material [kg m^-3]
    s2c_moon [num] : Lunar simple-to-complex transition diamter [m]
    g_moon [num] : Gravity of the Moon [m s^-2]
    rho_moon [num] : Bulk density of the Moon [kg m^-3]

    Return
    ------
    simple2complex (num): Simple-to-complex transition diameter [m]
    """
    return g_moon * rho_moon * s2c_moon / (gravity * density)


def complex2peakring_diam(
    gravity,
    density,
    c2pr_moon=cfg.complex2peakring,
    g_moon=cfg.grav_moon,
    rho_moon=cfg.bulk_density,
):
    """
    Return complex-to-peak ring basin transition diameter given gravity of
    body [m s^-2] and density of target [kg m^-3] (Melosh 1989).

    Parameters
    ----------
    gravity [num] : Gravity of a planetary body [m s^-2]
    density [num] : Density of the target material [kg m^-3]
    c2pr_moon [num] : Lunar complex crater to peak ring transition diamter [m]
    g_moon [num] : Gravity of the Moon [m s^-2]
    rho_moon [num] : Bulk density of the Moon [kg m^-3]

    Return
    ------
    complex2peakring (num): Complex-to-peak ring basin transition diameter [m]
    """
    return g_moon * rho_moon * c2pr_moon / (gravity * density)


# Grid helpers
def get_grd_ind(x, y, grdx, grdy):
    """Return index of val in monotonic grid_axis."""
    xidx = np.searchsorted(grdx[0, :], x) - 1
    yidx = grdy.shape[0] - (np.searchsorted(-grdy[:, 0], y) - 1)
    return (xidx, yidx)

def grid_interp(x, y, grdvalues, grdx, grdy):
    """Return interpolated value x, y in 2D grid."""
    ix, iy = get_grd_ind(x, y, grdx, grdy)
    gx, gy = grdx.flatten(), grdy.flatten()

    dy = (y - gy[iy]) / (gy[iy + 1] - gy[iy])
    interp_y = (1 - dy) * grdvalues[iy] + dy * grdvalues[iy + 1]

    dx = (x - gx[ix]) / (gx[ix + 1] - gx[ix])
    interp = (1 - dx) * interp_y[ix] + dx * interp_y[ix + 1]
    return interp


# Ballistic speed with no curvature (we use ballistic_spherical instead)
def ballistic_planar(theta, d, g=1.62):
    """
    Return ballistic speed (v) given ballistic range (d) and gravity of planet (g).
    Assumes planar surface (d << R_planet).

    Parameters
    ----------
    d (num or array): ballistic range [m]
    g (num): gravitational force of the target body [m s^-2]
    theta (num): angle of impaact [radians]

    Returns
    -------
    v (num or array): ballistic speed [m s^-1]

    """
    return np.sqrt((d * g) / np.sin(2 * theta))


# Unused thermal melting module for ballistic sed
def ice_melted(
    ke,
    t_surf=cfg.coldtrap_max_temp,
    cp=cfg.regolith_cp,
    ice_rho=cfg.ice_density,
    ice_frac=cfg.ice_frac,
    heat_frac=cfg.heat_frac,
    heat_ret=cfg.heat_retained,
    lat_heat=cfg.ice_latent_heat,
    t_melt=cfg.ice_melt_temp,
):
    """
    Return mass of ice [kg] melted by input kinetic energy.

    Parameters
    ----------
    ke (num or array): kinetic energy of ejecta blanket [J m^-2]
    t_surf (num): surface temperature [K]
    cp (num): heat capacity for regolith [J kg^-1 K^-1]
    ice_rho (num): ice density [kg m^-3]
    ice_frac (num): fraction ice vs regolith (default: 5.6%; Colaprete 2010)
    heat_frac (num): fraction KE used in heating vs mixing (default 50%)
    heat_ret (num): fraction of heat retained (10-30%; Stopar 2018)
    lat_heat (num): latent heat of ice [J/kg]
    t_melt (num): melting point of ice [K]

    Returns
    -------
    ice_mass (num or array): mass of ice melted due to ejecta [kg]
    """
    heat = ice_frac * heat_ret * heat_frac * ke  # [J m^-2]
    delta_t = t_melt - t_surf  # heat to go from t_surf to melting point
    ice_mass = heat / (lat_heat + cp * delta_t)  # [kg]
    # ice_depth = ice_mass / (ice_frac * ice_rho)  # [m]
    return ice_mass


# Sublimation module not implemented (implicit in ballistic sed and impact gardening)
def get_sublimation_rate(timestep=cfg.timestep, temp=cfg.coldtrap_max_temp):
    """
    Return ice lost due to sublimation at temp each timestep.

    Compute surface residence time (Langmuir 1916, Kloos 2019), invert to
    num H2O molecules lost in timestep, then convert num H2O to ice thickness.
    """
    vu0 = 2e12  # [s^-1] (vibrational frequency of water)
    Ea = 0.456  # [eV] (activation energy)
    kb = 8.6e-5  # [ev K^-1] (Boltzmann constant, in eV units)
    tau = (1 / vu0) * np.exp(Ea / (kb * temp))  # [s], surface residence time
    # TODO: covert tau to our units, get num H2O out/s, multiply by timestep
    # convert num H2O to thickness of ice
    return 0


# Conversions
def mps2kmph(v):
    """
    Return v in km/hr, given v in m/s

    Parameters
    ----------
    v (num or array): velocity [m s^-1]

    Returns
    -------
    v (num or array): velocity [km hr^-1]
    """

    return 3600 * v / 1000


def thick2mass(thick, density):
    """
    Convert an ejecta blanket thickness to kg per meter squared, default
    density of the ejecta blanket from Carrier et al. 1991.
    Density should NOT be the bulk density of the Moon!

    Parameters
    ----------
    thick (num or array): ejecta blanket thickness [m]
    density (num): ejecta blanket density [kg m^-3]

    Returns
    -------
    mass (num or array): mass of the ejecta blanket [kg]
    """
    return thick * density


def mps2KE(speed, mass):
    """
    Return kinetic energy [J m^-2] given mass [kg], speed [m s^-1].

    Parameters
    ----------
    speed (num or array): ballistic speed [m s^-1]
    mass (num or array): mass of the ejecta blanket [kg]

    Returns
    -------
    KE (num or array): Kinetic energy [J m^-2]
    """
    return 0.5 * mass * speed ** 2.0

