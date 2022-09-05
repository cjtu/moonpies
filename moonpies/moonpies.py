"""Moon Polar Ice and Ejecta Stratigraphy model (MoonPIES).

This module contains the main functions for running the MoonPIES model. This
model simulates the evolution of polar ice and ejecta stratigraphy on the Moon
over geologic time.


:Authors: C.J. Tai Udovicic, K.R. Frizzell, K.M. Luchsinger, A. Madera, T.G. Paladino
:Acknowledgements: This model was developed as part of the 2021 Exploration
    Science Summer Intern Program hosted by the Lunar and Planetary Institute
    with support from the Center for Lunar Science and Exploration node of the 
    NASA Solar System Exploration Research Virtual Institute.
"""
import os
import gc
from functools import lru_cache, _lru_cache_wrapper
import numpy as np
from scipy import stats
import pandas as pd
from moonpies import config

CFG = config.Cfg()  # Default Cfg object

def main(cfg=CFG):
    """Run MoonPIES model and return stratigraphy columns.

    Runs full ejecta and ice straitigraphy model for time, cold traps and
    model parameters specified in cfg.

    Parameters
    ----------
    cfg : moonpies.config.Cfg
        MoonPIES config object.

    Returns
    -------
    strat_dict : dict
        Dict of {str:pandas.DataFrame} of coldtrap names and stratigraphy
        tables (rows: layers, cols: ice, ejecta, source, time, depth, ice%)

    See Also
    --------
    moonpies.config.Cfg

    Examples
    --------
    >>> import moonpies as mp
    >>> ej_df, ice_df, strat_dfs = mp.main()
    >>> print(strat_dfs['Faustini'].head())
    """
    # Setup phase
    vprint(cfg, "Initializing run...")
    clear_cache()
    rng = get_rng(cfg)

    # Setup time and crater list
    time_arr = get_time_array(cfg)
    df = get_crater_basin_list(cfg, rng)
    if not cfg.ejecta_basins:
        df[~df.isbasin].reset_index(drop=True)

    # Init strat columns dict based for all cfg.coldtrap_names
    ej_dists = get_coldtrap_dists(df, cfg)  # Crater -> coldtrap distances (2D)
    strat_cols = init_strat_columns(time_arr, df, ej_dists, cfg, rng)

    # Get gardening and bsed time arrays
    bsed_depth, bsed_frac = get_bsed_depth(time_arr, df, ej_dists, cfg)
    overturn = overturn_depth_time(time_arr, cfg)

    # Main loop over time
    vprint(cfg, "Starting main loop...")
    strat_cols = update_strat_cols(
        strat_cols, overturn, bsed_depth, bsed_frac, cfg
    )

    # Format and save outputs
    return format_save_outputs(strat_cols, time_arr, df, cfg)


def get_time_array(cfg=CFG):
    """Return model time array [yr].

    Parameters
    ----------
    cfg : moonpies.config.Cfg, optional
        cfg.timestart : int
            Start time [yr].
        cfg.timeend : int
            End time [yr].
        cfg.timestep : int
            Time step [yr].

    Returns
    -------
    time_arr : np.ndarray
    """    
    n = int((cfg.timestart - cfg.timeend) / cfg.timestep)
    return np.linspace(cfg.timestart, cfg.timestep, n, dtype=cfg.dtype)


# Strat column functions
def init_strat_columns(time_arr, df, ej_dists, cfg=CFG, rng=None):
    """Return initialized model stratigraphy columns.

    Initialize dict of list (ice, ej, ej_sources) for each cold trap. Computes
    the ice and ejecta delivered in each model timestep to each cold trap.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater and basin DataFrame.
    ej_dists : numpy.ndarray
        2D crater to cold trap distance array [m] (Ncrater, Ncoldtrap).
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.
    rng : int or numpy.random.Generator, optional
        Random seed or random number generator, by default None.

    Returns
    -------
    dict
        Dict of {str : list of array} of coldtrap names to [ice, ej, ej_src].

    See Also
    --------
    get_ejecta_thickness_time, get_ice_cols_time
    """
    ej_cols, ej_srcs = get_ejecta_thickness_time(time_arr, df, ej_dists, cfg)
    ice_cols = get_ice_cols_time(time_arr, df, cfg, rng)

    # Get cold trap crater names (corresponds to columns in ej_cols, ice_cols)
    ctraps = cfg.coldtrap_names

    # Build strat columns as {cname: ice_col, ej_col, ej_src}
    strat_columns = {
        coldtrap: [ice_cols[:, i], ej_cols[:, i], ej_srcs[:, i]]
        for i, coldtrap in enumerate(ctraps)
    }
    return strat_columns


def update_ice_col(ice_col, ej_col, t, overturn_d, bsed_d, bsed_frac, cfg=CFG):
    """Return ice_column updated with all processes applicable at time t.

    Apply effects of impact gardening (overturn) and ballistic sedimentation 
    (bsed) at time t to ice_col. Modifies ice_col in place.

    Parameters
    ----------
    ice_col : umpy.ndarray
        Ice column array.
    ej_col : numpy.ndarray
        Ejecta column array.
    t : int
        Current time index.
    overturn_d : float
        Overturn depth [m] at timestep t.
    bsed_d : float
        Ballistic sedimentation depth [m] at timestep t.
    bsed_frac : float
        Ballistic sedimentation fraction at timestep t.
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.

    Returns
    -------
    ice_col : numpy.ndarray
        Updated ice column arr.

    See Also
    --------
    garden_ice_column, remove_ice_overturn
    """    
    # Ballistic sed gardens column before any ice gain (timestep t-1)
    ice_col = garden_ice_column(ice_col, ej_col, t - 1, bsed_d, bsed_frac)

    # Ice "gained" by column (already pre-computed in ice_col[t])

    # Ice gardened at end of timestep, i.e. after ice gain (timestep t)
    ice_col = remove_ice_overturn(ice_col, ej_col, t, overturn_d, cfg)
    return ice_col


def update_strat_cols(strat_cols, overturn, bsed_depth, bsed_frac, cfg=CFG):
    """Return strat_cols dict updated with ice modification at each timestep.

    Parameters
    ----------
    strat_cols : dict
        Dict of {str : list of array} of coldtrap names to [ice, ej, ej_src].
    overturn : numpy.ndarray
        Overturn depth array [m] (Ntime).
    bsed_depth : numpy.ndarray
        Ballistic sedimentation depth array [m] (Ntime).
    bsed_frac : numpy.ndarray
        Ballistic sedimentation fraction array (Ntime).
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.

    Returns
    -------
    strat_cols : dict
        Updated strat_cols dict.

    See Also
    --------
    update_ice_col
    """    
    # Loop through all timesteps
    for t, overturn_t in enumerate(overturn):
        # Update all coldtrap ice_cols
        for i, coldtrap in enumerate(cfg.coldtrap_names):
            ice_col, ej_col, _ = strat_cols[coldtrap]
            ice_col = update_ice_col(
                ice_col, 
                ej_col,
                t,
                overturn_t,
                bsed_depth[t, i],
                bsed_frac[t, i],
                cfg,
            )
            strat_cols[coldtrap][0] = ice_col  # Redundant (updated in place)
    return strat_cols


# Import data
def get_crater_basin_list(cfg=CFG, rng=None):
    """Return dataframe of craters with ages randomized.

    Parameters
    ----------
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.
    rng : int or numpy.random.Generator, optional
        Random seed or random number generator, by default None.

    Returns
    -------
    pandas.DataFrame
        Crater and basin DataFrame.

    See Also
    --------
    random_icy_basins, randomize_crater_ages
    """
    df_craters = read_crater_list(cfg)
    df_craters["isbasin"] = False
    df_craters["icy_impactor"] = "no"

    df_basins = read_basin_list(cfg)
    df_basins["isbasin"] = True
    df_basins = random_icy_basins(df_basins, cfg, rng)

    # Combine DataFrames and randomize ages
    df = pd.concat([df_craters, df_basins])
    df = randomize_crater_ages(df, cfg.timestep, rng)
    return df


def random_icy_basins(df, cfg=CFG, rng=None):
    """Randomly assign each basin an impactor type determining hydration. 
    
    Probability of hydration and comet based on cfg. Possible icy_impactor 
    types are {"hyd_ctype", "comet", or "no"}.

    Parameters
    ----------
    df : pandas.DataFrame
        Basin DataFrame.
    cfg : moonpies.config.Cfg, optional
        cfg.impact_ice_basins : bool
            Whether to allow basins to be icy.
        cfg.impact_ice_comets : bool
            Whether to allow comet impactors.
    rng : int or numpy.random.Generator, optional
        Random seed or random number generator, by default None.

    Returns
    -------
    pandas.DataFrame
        Basin DataFrame with icy_impactor column added.

    See Also
    --------
    get_random_hydrated_basins
    """
    df["icy_impactor"] = "no"
    ctype, comet = get_random_hydrated_basins(len(df), cfg, rng)
    if cfg.impact_ice_basins:
        df.loc[ctype, "icy_impactor"] = "hyd_ctype"
        if cfg.impact_ice_comets:
            df.loc[comet, "icy_impactor"] = "comet"
    return df


def read_crater_list(cfg=CFG):
    """Return DataFrame of craters from cfg.crater_csv_in.

    Required columns, names and units in cfg.crater_csv_in:
        - 'lat': Latitude [deg]
        - 'lon': Longitude [deg]
        - 'diam': Diameter [km]
        - 'age': Crater age [Gyr]
        - 'age_low': Age error residual, lower (e.g., age - age_low) [Gyr]
        - 'age_upp': Age error residual, upper (e.g., age + age_upp) [Gyr]

    Optional columns in cfg.crater_csv_in:
        - 'psr_area': Permanently shadowed area of crater [km^2]

    Parameters
    ----------
    cfg : moonpies.config.Cfg, optional
        cfg.crater_csv_in : str
            Path to crater CSV file.
        cfg.crater_cols : tuple of str
            Tuple of column names in cfg.crater_csv_in.

    Returns
    -------
    pandas.DataFrame
        Crater DataFrame.
    """
    df = pd.read_csv(cfg.crater_csv_in, names=cfg.crater_cols, header=0)

    # Convert units, mandatory columns
    df["diam"] = df["diam"] * 1000  # [km -> m]
    df["rad"] = df["diam"] / 2
    df["age"] = df["age"] * 1e9  # [Gyr -> yr]
    df["age_low"] = df["age_low"] * 1e9  # [Gyr -> yr]
    df["age_upp"] = df["age_upp"] * 1e9  # [Gyr -> yr]

    # Define optional columns
    if "psr_area" in df.columns:
        df["psr_area"] = df.psr_area * 1e6  # [km^2 -> m^2]
    else:
        # Estimate psr area as 90% of crater area
        df["psr_area"] = 0.9 * np.pi * df.rad**2
    return df


def read_basin_list(cfg=CFG):
    """Return dataframe of craters from basin_csv path with columns names.

    Required columns, names and units in cfg.basin_csv_in:
        - 'lat': Latitude [deg]
        - 'lon': Longitude [deg]
        - 'diam': Diameter [km]
        - 'age': Crater age [Gyr]
        - 'age_low': Age error residual, lower (e.g., age - age_low) [Gyr]
        - 'age_upp': Age error residual, upper (e.g., age + age_upp) [Gyr]

    Parameters
    ----------
    cfg : moonpies.config.Cfg, optional
        cfg.basin_csv_in : str
            Path to basin CSV file.
        cfg.basin_cols : tuple of str
            Tuple of column names in cfg.basin_csv_in.

    Returns
    -------
    pandas.DataFrame
        Basin DataFrame.
    """
    df = pd.read_csv(cfg.basin_csv_in, names=cfg.basin_cols, header=0)

    # Convert units, mandatory columns
    df["diam"] = df["diam"] * 1000  # [km -> m]
    df["rad"] = df["diam"] / 2
    df["age"] = df["age"] * 1e9  # [Gyr -> yr]
    df["age_low"] = df["age_low"] * 1e9  # [Gyr -> yr]
    df["age_upp"] = df["age_upp"] * 1e9  # [Gyr -> yr]
    return df


def read_volcanic_species(cfg=CFG):
    """Return DataFrame of time, species mass from Table S3 [1]_.

    Parameters
    ----------
    cfg : moonpies.config.Cfg, optional
        cfg.nk_csv : str
            Path to Table S3 [1] CSV file.
        cfg.nk_cols : tuple of str
            Tuple of column names in cfg.nk_csv.
        cfg.species : str
            Species name (must be in nk_cols).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns "time" and species.
    
    References
    ----------

    .. [1] Needham, D. H., & Kring, D. A. (2017). "Lunar volcanism produced a 
       transient atmosphere around the ancient Moon." Earth and Planetary 
       Science Letters, 478, 175-178. 
       https://doi.org/10.1016/j.epsl.2017.09.002

    """
    df = pd.read_csv(cfg.nk_csv, names=cfg.nk_cols, header=4)
    df = df[["time", cfg.species]]
    df["time"] = df["time"] * 1e9  # [Gyr -> yr]
    df[cfg.species] = df[cfg.species] * 1e-3  # [g -> kg]
    df = df.sort_values("time", ascending=False).reset_index(drop=True)
    return df


def read_solar_luminosity(cfg=CFG):
    """Return DataFrame of time, solar luminosity from Table 2 [1]_.

    Parameters
    ----------
    cfg : moonpies.config.Cfg, optional
        cfg.bahcall_csv_in : str
            Path to CSV of Table 2 [1].

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns "time" and "luminosity".
    
    References
    ----------
    .. [1] Bahcall, J. N., Pinsonneault, M. H., & Basu, S. (2001). "Solar 
       Models: Current Epoch and Time Dependences, Neutrinos, and 
       Helioseismological Properties." The Astrophysical Journal, 555(2), 
       990-1012. https://doi.org/10/dv2q29

    """
    cols = ["time", "luminosity"]
    df = pd.read_csv(cfg.bahcall_csv_in, names=cols, header=1)
    df.loc[:, "time"] = (4.57 - df.loc[:, "time"]) * 1e9  # [Gyr -> yr]
    return df


@lru_cache(2)
def read_ballistic_melt_frac(mean=True, cfg=CFG):
    """Return DataFrame of ballistic sedimentation melt fractions.

    Parameters
    ----------
    mean : bool, optional
        If True, return mean melt fraction. If False, return standard dev
    cfg : moonpies.config.Cfg, optional
        cfg.bsed_frac_mean_in : str
            CSV of ballistic sedimentation melt fractions.

    Returns
    -------
    pandas.DataFrame
        Melt fractions (index: mixing ratio, cols: ejecta temperature).
    """
    csv = cfg.bsed_frac_mean_in if mean else cfg.bsed_frac_std_in
    df = pd.read_csv(csv, index_col=0, dtype=cfg.dtype)
    df.columns = df.columns.astype(cfg.dtype)
    return df

# Pre-compute grid functions
def get_coldtrap_dists(df, cfg=CFG):
    """Return great circle distance between all craters and coldtraps. 
    
    Returns a 2D array (rows: craters from df, cols: cold traps from 
    cfg.coldtrap_names, in order). Elements are NaN for distance from crater 
    to its own cold trap or distance further than ejecta_threshold radii.


    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    cfg : moonpies.config.Cfg, optional
        cfg.coldtrap_names : tuple of str
            Tuple of cold trap names.
        cfg.ejecta_threshold : float
            Continuous ejecta threshold distance [crater radii].
        cfg.basin_ej_threshold : float
            Continuous ejecta threshold distance for basins [basin radii].

    Returns
    -------
    numpy.ndarray
        2D array of distances from craters to cold traps (Ncrater, Ncoldtrap).
    """
    dist = np.zeros((len(df), len(cfg.coldtrap_names)), dtype=cfg.dtype)
    for i, row in df.iterrows():
        src_lon = row.lon
        src_lat = row.lat
        src_rad = row.rad
        if row.isbasin:
            ej_threshold = cfg.basin_ej_threshold
        else:
            ej_threshold = cfg.ej_threshold
        if ej_threshold < 0:
            ej_threshold = np.inf
        for j, cname in enumerate(cfg.coldtrap_names):
            if row.cname == cname:
                dist[i, j] = np.nan
                continue
            dst_lon = df[df.cname == cname].psr_lon.values
            dst_lat = df[df.cname == cname].psr_lat.values
            d = gc_dist(src_lon, src_lat, dst_lon, dst_lat)
            # Only keep distances outside crater radius and less than thresh
            if src_rad < d < ej_threshold * src_rad:
                dist[i, j] = d
    dist[dist <= 0] = np.nan  # Set zeros to NaN
    return dist


def get_ejecta_thickness(distance, radius, cfg=CFG):
    """Return ejecta thickness as a function of distance given crater radius.

    Parameters
    ----------
    distance : numpy.ndarray
        Distance from crater center [m].
    radius : float
        Crater radius [m].
    cfg : moonpies.config.Cfg, optional
        cfg.simple2complex : float
            Simple to complex transition diameter [m]
        cfg.complex2peakring : float
            Complex to peak ring basin transition diameter [m]

    Returns
    -------
    thickness : numpy.ndarray
        Ejecta thickness [m] at distance.

    See Also
    --------
    get_ej_thick_simple, get_ej_thick_complex, get_ej_thick_basin
    """
    dist_v = np.atleast_1d(distance)
    rad_v = np.broadcast_to(radius, dist_v.shape)
    thick = np.zeros_like(dist_v * rad_v)

    # Simple craters
    is_s = rad_v < cfg.simple2complex / 2
    thick[is_s] = get_ej_thick_simple(dist_v[is_s], rad_v[is_s], cfg)

    # Complex craters
    is_c = (cfg.simple2complex / 2 < rad_v) & (
        rad_v < cfg.complex2peakring / 2
    )
    thick[is_c] = get_ej_thick_complex(dist_v[is_c], rad_v[is_c], cfg)

    # Basins
    is_pr = rad_v >= cfg.complex2peakring / 2
    t_radius = final2transient(rad_v * 2, cfg) / 2
    thick[is_pr] = get_ej_thick_basin(dist_v[is_pr], t_radius[is_pr], cfg)
    thick[np.isnan(thick)] = 0
    if np.ndim(distance) == 0 and len(thick) == 1:
        thick = float(thick)  # Return scalar if passed a scalar
    return thick


def get_ej_thick_simple(distance, radius, cfg=CFG):
    """Return ejecta thickness with distance from a simple crater [1]_.

    Parameters
    ----------
    distance : numpy.ndarray
        Distance from crater center [m].
    radius : float
        Crater final radius [m].
    cfg : moonpies.config.Cfg, optional
        cfg.ej_thickness_exp : float
            Ejecta thickness exponent, default: -3.

    Returns
    -------
    thickness : numpy.ndarray
        Ejecta thickness [m] at distance.

    References
    ----------
    .. [1] Kring, D. A. (1995). "The dimensions of the Chicxulub impact crater
       and impact melt sheet." Journal of Geophysical Research, 100(E8),
       16979. https://doi.org/10/fphq8j

    """    
    return 0.04 * radius * (distance / radius) ** cfg.ej_thickness_exp


def get_ej_thick_complex(distance, radius, cfg=CFG):
    """Return ejecta thickness with distance from a complex crater [1]_.

    Parameters
    ----------
    distance : numpy.ndarray
        Distance from crater center [m].
    radius : float
        Crater final radius [m].
    cfg : moonpies.config.Cfg, optional
        cfg.ej_thickness_exp : float
            Ejecta thickness exponent, default: -3.

    Returns
    -------
    thickness : numpy.ndarray
        Ejecta thickness [m] at distance.

    References
    ----------
    .. [1] Kring, D. A. (1995). "The dimensions of the Chicxulub impact crater
       and impact melt sheet." Journal of Geophysical Research, 100(E8),
       16979. https://doi.org/10/fphq8j
    """
    return 0.14 * radius**0.74 * (distance / radius) ** cfg.ej_thickness_exp


def get_ej_thick_basin(distance, t_radius, cfg=CFG):
    """Return ejecta thickness with distance from a basin [1]_.

    Uses Pike (1974) and Haskin et al. (2003) correction for curvature as
    described in [1].

    Parameters
    ----------
    distance : numpy.ndarray
        Distance from crater center [m].
    t_radius : float
        Basin transient crater radius [m].
    cfg : moonpies.config.Cfg, optional
        cfg.ej_thickness_exp : float
            Ejecta thickness exponent, default: -3.
        cfg.rad_moon : float
            Lunar radius [m].
        cfg.grav_moon : float
            Lunar gravity [m/s^2].
        cfg.impact_angle : float
            Impact angle [deg].

    Returns
    -------
    thickness : numpy.ndarray
        Ejecta thickness [m] at distance.

    References
    ----------
    .. [1] Zhang, X., Xie, M., & Xiao, Z. (2021). "Thickness of orthopyroxene-
      rich materials of ejecta deposits from the south pole-Aitken basin."
      Icarus, 358, 114214. https://doi.org/10/gmfc53
    """
    def r_flat(r_t, R, g, theta):
        """Return distance on flat surface, r' (eq S3, Zhang et al., 2021)"""
        v = ballistic_velocity(r_t * 1e3) / 1e3
        return R + (v**2 * np.sin(2 * np.deg2rad(theta))) / g

    def r_soi(r, R, R0):
        """Return distance to SOI (eq S4/S5, Zhang et al., 2021)."""
        tanr = np.tan((r - R) / (2 * R0))
        return R + (2 * R0 * tanr) / (tanr + 1)

    def s_flat(r_inner_sph, r_outer_sph):
        """Return area of flat SOI (eq S6, Zhang et al., 2021)."""
        return np.pi * (r_outer_sph**2 - r_inner_sph**2)

    def s_spherical(r_inner, r_outer, R0):
        """Return area of spherical SOI (eq S7, Zhang et al., 2021)."""
        return (
            2 * np.pi * R0**2 * (np.cos(r_inner / R0) - np.cos(r_outer / R0))
        )

    def area_corr(distance, radius, R0):
        """Return correction for area of SOI (eq S8, Zhang et al., 2021)."""
        r_inner = distance - radius / 8
        r_outer = distance + radius / 8
        area_sph = s_spherical(r_inner, r_outer, R0)
        r_inner_sph = r_soi(r_inner, radius, R0)
        r_outer_sph = r_soi(r_outer, radius, R0)
        area_flat = s_flat(r_inner_sph, r_outer_sph)
        return area_flat / area_sph

    # Compute dist and area corrections and return thickness
    exp = cfg.ej_thickness_exp
    radius = t_radius * 1e-3  # [m] -> [km]
    distance *= 1e-3  # [m] -> [km]
    R0 = cfg.rad_moon * 1e-3  # [m] -> [km]
    grav = cfg.grav_moon * 1e-3  # [m] -> [km]
    theta = cfg.impact_angle  # [deg]
    r_t = distance - radius
    rflat = r_flat(r_t, radius, grav, theta)
    with np.errstate(divide="ignore", over="ignore"):
        acorr = area_corr(distance, radius, R0)
        thickness = 0.033 * radius * np.power(rflat / radius, exp) * acorr
    return thickness * 1e3  # [km] -> [m]


def get_ejecta_thickness_time(time_arr, df, ej_distances, cfg=CFG):
    """Return ejecta thickness and ejecta sources at each time. 
    
    Ejecta layers smaller than cfg.thickness_min are set to 0.
    
    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater DataFrame.
    ej_distances : numpy.ndarray
        2D array of crater to cold trap distances [m].
    cfg : moonpies.config.Cfg, optional
        cfg.thickness_min : float
            Minimum ejecta layer thickness [m].

    Returns
    -------
    thickness_time : numpy.ndarray
        2D array ejecta thickness [m] at each time (Ntime, Ncoldtrap).
    sources_time : numpy.ndarray of str
        2D array of ejecta source(s) at each time (Ntime, Ncoldtrap).

    See Also
    --------
    get_ejecta_thickness_matrix
    """
    ej_thick = get_ejecta_thickness_matrix(df, ej_distances, cfg)
    ej_ages = df.age.values
    ej_thick_t = ages2time(time_arr, ej_ages, ej_thick, np.nansum, 0)

    # Label sources above threshold
    has_ej = ej_thick > cfg.thickness_min
    ej_srcs = (df.cname.values[:, np.newaxis] + ",") * has_ej
    ej_sources_t = ages2time(time_arr, ej_ages, ej_srcs, np.sum, "", object)
    ej_sources_t = np.char.rstrip(ej_sources_t.astype(str), ",")
    return ej_thick_t, ej_sources_t


def get_ejecta_thickness_matrix(df, ej_dists, cfg=CFG):
    """Return ejecta thickness from each crater to each cold trap.

    Rows are craters in df, columns are cold traps from cfg.coldtrap_names.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    ej_dists : numpy.ndarray
        2D array of crater to cold trap distances [m] (Ncrater, Ncoldtrap).

    Returns
    -------
    thickness : numpy.ndarray
        2D array of ejecta thickness [m] (Ncrater, Ncoldtrap).
    
    See Also
    --------
    get_ejecta_thickness_time, get_ejecta_thickness
    """
    # Get radius array same shape as ej_dists (radii as rows)
    rad = np.tile(df.rad.values[:, np.newaxis], (1, ej_dists.shape[1]))
    return get_ejecta_thickness(ej_dists, rad, cfg)


def ages2time(
    time_arr, ages, values, agg=np.nansum, fillval=0, dtype=None, cfg=CFG
):
    """Return values shaped like time_arr given corresponding ages.

    Rounds ages to nearest time in time_arr. If multiple values have the same 
    age, they are combined using agg. Times with no ages are set to fillval.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    ages : numpy.ndarray
        Age of each value [yr].
    values : numpy.ndarray
        Values to reshape (must be same length as ages).
    agg : callable, optional
        Function to combine multiple values of same age, by default np.nansum.
    fillval : int, optional
        Value to use as default (all times lacking a value), by default 0.
    dtype : data-type, optional
        Data type of output, by default same as values.
    cfg : moonpies.config.Cfg, optional
        cfg.rtol : float
            Relative tolerance for time comparison to avoid rounding error.

    Returns
    -------
    values_time : numpy.ndarray
        Values reshaped like time_arr.
    """
    if dtype is None:
        dtype = values.dtype
    # Remove ages and associated values not in time_arr
    tmin, tmax = minmax_rtol(time_arr, cfg.rtol)  # avoid rounding error
    isin = np.where((ages >= tmin) & (ages <= tmax))
    ages = ages[isin]
    values = values[isin]

    # Minimize distance between ages and times to handle rounding error
    time_idx = np.abs(time_arr[:, np.newaxis] - ages).argmin(axis=0)
    shape = [len(time_arr), *values.shape[1:]]
    values_time = np.full(shape, fillval, dtype=dtype)

    for i, t_idx in enumerate(time_idx):
        # Sum here in case more than one crater formed at t_idx
        values_time[t_idx] = agg([values_time[t_idx], values[i]], axis=0)
    return values_time

# Polar ice deposition
def get_ice_cols_time(time_arr, df, cfg=CFG, rng=None):
    """Return ice thickness from all sources at all times for each cold trap.

    Combines polar ice and volcanic-derived ice and applies ballistic hop 
    efficiency to determine amount of ice that reaches each cold trap.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater DataFrame.
    cfg : moonpies.config.Cfg, optional
        cfg.coldtrap_names : list of str
            Names of cold traps.
        cfg.ballistic_hop_moores : bool
            Use Moores et al. (2016) ballistic hop model for ice deposition.

    rng : int or numpy.random.Generator, optional
        Random seed or random number generator, by default None.

    Returns
    -------
    ice_time : numpy.ndarray
        2D array of ice thickness [m] at each time (Ntime, Ncoldtrap).

    See Also
    --------
    get_polar_ice, get_volcanic_ice
    """
    # Get column vectors of polar and volc ice
    ice_polar = get_polar_ice(time_arr, df, cfg, rng)[:, None]
    ice_volcanic = get_volcanic_ice(time_arr, cfg)[:, None]

    # TODO: refactor into get_volcanic ice, need to move ballistic hop too
    if cfg.use_volc_dep_effcy:
        # Rescale by volc dep effcy, apply evenly to all coldtraps
        ice_volcanic *= cfg.volc_dep_effcy / cfg.ballistic_hop_effcy
    else:
        # Treat as ballistically hopping polar ice
        ice_polar += ice_volcanic
        ice_volcanic *= 0

    # Rescale by ballistic hop efficiency per coldtrap
    if cfg.ballistic_hop_moores:
        bhops = get_ballistic_hop_coldtraps(list(cfg.coldtrap_names), cfg)
        bhops /= cfg.ballistic_hop_effcy
        ice_polar = bhops * ice_polar  # row * col -> 2D arr
    else:
        ice_polar = np.tile(ice_polar, len(cfg.coldtrap_names))
    ice_cols_time = ice_polar + ice_volcanic
    return ice_cols_time


def get_polar_ice(time_arr, df, cfg=CFG, rng=None):
    """Return total polar ice at all timesteps from all sources.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater DataFrame.
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.
    rng : int or numpy.random.Generator, optional
        Random seed or random number generator, by default None.

    Returns
    -------
    ice_time : numpy.ndarray
        Total ice delivered to the pole [m] at each time.

    See Also
    --------
    get_impact_ice, get_impact_ice_comet, get_solar_wind_ice
    """    
    impact_ice = get_impact_ice(time_arr, df, cfg, rng)
    comet_ice = get_impact_ice_comet(time_arr, df, cfg, rng)
    if cfg.impact_ice_comets:
        # comet_ice is run every time for repro, but only add if needed
        impact_ice += comet_ice
    solar_wind_ice = get_solar_wind_ice(time_arr, cfg)
    return impact_ice + solar_wind_ice


def get_ice_thickness(global_ice_mass, cfg=CFG):
    """Return polar ice thickness [m] given globally delivered ice mass [kg].

    Assumes ice is uniformly distributed over polar coldtrap area and has a 
    constant ice density. Scale by amount of global ice that reaches poles.

    Parameters
    ----------
    global_ice_mass : np.ndarray
        Ice delivered globally [kg].
    cfg : moonpies.config.Cfg, optional
        cfg.ice_species : str
            Ice species {"H2O", "CO2"}.
        cfg.coldtrap_area_H2O, cfg.coldtrap_area_CO2 : float
            Area of cold traps [m^2].
        cfg.ballistic_hop_effcy : float
            Ballistic hop efficiency (fraction of ice that reaches poles).
        cfg.ice_density : float
            Ice density [kg/m^3].

    Returns
    -------
    ice_thickness : np.ndarray
        Polar ice thickness [m].
    """    
    if cfg.ice_species == "H2O":
        coldtrap_area = cfg.coldtrap_area_H2O
    elif cfg.ice_species == "CO2":
        coldtrap_area = cfg.coldtrap_area_CO2
    polar_ice_mass = global_ice_mass * cfg.ballistic_hop_effcy  # [kg]
    ice_volume = polar_ice_mass / cfg.ice_density  # [m^3]
    ice_thickness = ice_volume / coldtrap_area
    return ice_thickness


# Solar wind module
def get_solar_wind_ice(time_arr, cfg=CFG):
    """Return thickness of solar wind ice at all times.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.

    Returns
    -------
    ice_time : numpy.ndarray
        Thickness of solar wind ice delivered [m] at each time.
    
    See Also
    --------
    solar_wind_ice_mass, get_ice_thickness
    """
    sw_ice_t = np.zeros_like(time_arr)
    if cfg.solar_wind_ice:
        sw_ice_mass = solar_wind_ice_mass(time_arr, cfg)
        sw_ice_t = get_ice_thickness(sw_ice_mass, cfg)
    return sw_ice_t


def solar_wind_ice_mass(time_arr, cfg=CFG):
    """Return global solar wind ice mass [kg] deposited at each time.

    H2O supply rate is given by cfg.solar_wind_mode ("Benna": 2g/s [1]_,
    "Lucey-Hurley": H2 supply of 30 g/s, H2 -> H2O at 1 ppt [2]_). If 
    cfg.faint_young_sun, scale solar wind H2O supply rate by relative solar
    luminosity at each time [3]_.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.solar_wind_mode : str
            Solar wind ice mode {"Benna", "Lucey-Hurley"}.
        cfg.faint_young_sun : bool
            Use faint young sun model.
        
    Returns
    -------
    ice_mass : numpy.ndarray
        Global solar wind ice mass [kg] deposited at each time.

    References
    ----------
    .. [1] Benna, M., Hurley, D. M., Stubbs, T. J., Mahaffy, P. R., & Elphic,
       R. C. (2019). "Lunar soil hydration constrained by exospheric water
       liberated by meteoroid impacts." Nature Geoscience, 12(5), 333-338.
       https://doi.org/10/c4nz

    .. [2] Lucey, P. G., Costello, E. S., Hurley, D. M., Prem, P., Farrell,
       W. M., Petro, N., & Cable, M. L. (2020). Relative Magnitudes of Water
       Sources to the Lunar Poles. LPS LI, Abstract # 2319.
       Retrieved from https://www.hou.usra.edu/meetings/lpsc2020/pdf/2319.pdf

    .. [3] Bahcall, J. N., Pinsonneault, M. H., & Basu, S. (2001). "Solar 
       Models: Current Epoch and Time Dependences, Neutrinos, and 
       Helioseismological Properties." The Astrophysical Journal, 555(2), 
       990-1012. https://doi.org/10/dv2q29
    """ 
    if cfg.solar_wind_mode.lower() == "benna":
        # Benna et al. 2019; Arnold, 1979; Housley et al. 1973
        volatile_supply_rate = 2 * 1e-3  # [g/s -> kg/s]
    elif cfg.solar_wind_mode.lower() == "lucey-hurley":
        # Lucey et al. 2020, Hurley et al. 2017 (assume 1 ppt H2 -> H2O)
        volatile_supply_rate = 30 * 1e-3 * 1e-3  # [g/s - kg/s] * [1 ppt]
    else:
        msg = 'cfg.solar_wind_mode not one of {"Benna", "Lucey-Hurley"}'
        raise ValueError(msg)
    # convert to kg per timestep
    supply_rate_ts = volatile_supply_rate * 60 * 60 * 24 * 365 * cfg.timestep

    sw_ice_mass = np.ones_like(time_arr) * supply_rate_ts
    if cfg.faint_young_sun:
        # Import historical solar luminosity (Bahcall et al. 2001)
        df_lum = read_solar_luminosity(cfg)

        # Interpolate to time_arr, scale by solar luminosity at each time
        lum_time = np.interp(-time_arr, -df_lum.time, df_lum.luminosity)
        sw_ice_mass *= lum_time
    return sw_ice_mass


# Volcanic ice delivery module
def get_volcanic_ice(time_arr, cfg=CFG):
    """Return ice thickness [m] delivered to pole by volcanics at each time.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.volc_mode : str
            Volcanic ice mode {"NK", "Head"}.

    Returns
    -------
    ice_time : numpy.ndarray
        Thickness of volcanic ice delivered [m] at each time.

    See Also
    --------
    volcanic_ice_nk, volcanic_ice_head
    """
    if cfg.volc_mode == "NK":
        volc_ice_mass = volcanic_ice_nk(time_arr, cfg)
    elif cfg.volc_mode == "Head":
        volc_ice_mass = volcanic_ice_head(time_arr, cfg)
    else:
        raise ValueError(f"Invalid mode {cfg.volc_mode}.")

    volc_ice_t = get_ice_thickness(volc_ice_mass, cfg)
    return volc_ice_t


def volcanic_ice_nk(time_arr, cfg=CFG):
    """Return global ice mass [kg] deposited at each time using [1]_.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.volc_mode : str

    Returns
    -------
    ice_mass : numpy.ndarray
        Global volcanic ice mass [kg] deposited at each time.
    """    
    df_volc = read_volcanic_species(cfg)
    tmax = time_arr.max()
    df_volc = df_volc[df_volc.time < tmax + rtol(tmax, cfg.rtol)]

    rounded_time = np.rint(time_arr / cfg.timestep)
    rounded_ages = np.rint(df_volc.time.values / cfg.timestep)
    time_idx = np.searchsorted(-rounded_time, -rounded_ages)

    # Compute volc ice mass at each time in time_arr
    #   Divide each ice mass by time between timesteps in df_volc
    volc_ice_mass = np.zeros_like(time_arr)
    for i, t_idx in enumerate(time_idx[:-1]):
        # Sum here in case more than one crater formed at t_idx
        next_idx = time_idx[i + 1]
        dt = (time_arr[t_idx] - time_arr[next_idx]) / cfg.timestep
        volc_ice_mass[t_idx : next_idx + 1] = (
            df_volc.iloc[i][cfg.nk_species] / dt
        )
    return volc_ice_mass


def volcanic_ice_head(time_arr, cfg=CFG):
    """Return global ice [kg] deposited at each time using Head et al. (2020).

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.volc_total_vol : float
            Total volcanics volume [km^3].
        cfg.volc_magma_density : float
            Magma density [kg/m^3].
        cfg.volc_ice_ppm : float
            Volcanic volatile concentration [ppm].
        cfg.volc_early : tuple of float
            Early volcanism time period (start, end) [yr].
        cfg.volc_early_pct : float
            Percent of total volcanics delivered during volc_early.
        cfg.volc_late : tuple of float
            Late volcanism time period (start, end) [yr].
        cfg.volc_late_pct : float
            Percent of total volcanics delivered during volc_late.

    Returns
    -------
    ice_mass : numpy.ndarray
        Global volcanic ice mass [kg] deposited at each time.
    """
    # Global ice deposition (Head et al. 2020)
    volc_mass = cfg.volc_total_vol * cfg.volc_magma_density
    ice_total = volc_mass * cfg.volc_ice_ppm * 1e-6  # [ppm yr^-1 -> yr^-1]

    # Ice deposited per epoch
    ice_early = ice_total * cfg.volc_early_pct * cfg.timestep
    ice_late = ice_total * cfg.volc_late_pct * cfg.timestep

    # Set ice at times in each epoch
    volc_ice_mass = np.zeros_like(time_arr)
    emax, emin = cfg.volc_early
    lmax, lmin = cfg.volc_late
    volc_ice_mass[(time_arr <= emax) & (time_arr > emin)] = ice_early
    volc_ice_mass[(time_arr <= lmax) & (time_arr > lmin)] = ice_late
    return volc_ice_mass


# Ballistic sedimentation module
def get_melt_frac(ejecta_temps, mixing_ratios, cfg=CFG):
    """Return fraction of ice melted at given ejecta_temps and vol_fracs.

    Parameters
    ----------
    ejecta_temps : numpy.ndarray
        Ejecta temperatures [K].
    mixing_ratios : numpy.ndarray
        Mixing ratios (target:ejecta).
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.

    Returns
    -------
    melt_frac : numpy.ndarray
        Fraction of ice melted at each ejecta_temp and mixing ratio.

    See Also
    --------
    read_ballistic_melt_frac
    """
    def insert_unique_in_range(src, target):
        """Return sorted target with unique values in src inserted."""
        uniq = np.unique(src[~np.isnan(src)])
        # Ensure in range of target
        uniq = uniq[(target.min() < uniq) & (uniq < target.max())]
        arr = np.sort(np.unique(np.concatenate([target, uniq])))
        return arr

    mdf = read_ballistic_melt_frac(True, cfg)
    temps = insert_unique_in_range(ejecta_temps, mdf.columns.to_numpy())
    mrs = insert_unique_in_range(mixing_ratios, mdf.index.to_numpy())
    mdf = mdf.reindex(index=mrs, columns=temps)
    minterp = mdf.interpolate(axis=0).interpolate(axis=1)

    # Interpolate melt_frac at each non-nan ejecta_temp, mixing_ratio
    inds = np.argwhere(~np.isnan(mixing_ratios))
    melt_frac = np.zeros_like(mixing_ratios)
    for i, j in inds:
        melt_frac[i, j] = minterp.loc[mixing_ratios[i, j], ejecta_temps[i, j]]
    return melt_frac


def kinetic_energy(mass, velocity):
    """Return kinetic energy [J] given mass [kg] and velocity [m/s]."""
    return 0.5 * mass * velocity**2


def ejecta_temp(df, shape=None, cfg=CFG):
    """Return ejecta temperature [K].

    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    shape : tuple, optional
        Shape of output, by default (len(df),)
    cfg : moonpies.config.Cfg, optional
        cfg.polar_ejecta_temp_init : float
            Initial polar ejecta temperature [K].
        cfg.basin_ejecta_temp_warm : bool
            If True, use warm ejecta temperatures for basins, otherwise cold.
        cfg.basin_ejecta_temp_init_warm : float
            Initial warm basin ejecta temperature [K].
        cfg.basin_ejecta_temp_init_cold : float
            Initial cold basin ejecta temperature [K].

    Returns
    -------
    ejecta_temp : numpy.ndarray
        Ejecta temperature [K] for each impact in df.
    """
    if shape is None:
        shape = (len(df),)
    ejecta_t = np.ones(shape) * cfg.polar_ejecta_temp_init

    if cfg.basin_ejecta_temp_warm:
        basin_t = cfg.basin_ejecta_temp_init_warm
    else:
        basin_t = cfg.basin_ejecta_temp_init_cold
    ejecta_t[df.isbasin] = basin_t
    return ejecta_t


def ballistic_velocity(dist, cfg=CFG):
    """Return ballistic velocity given distance of travel on a sphere [1]_.

    Parameters
    ----------
    dist : numpy.ndarray
        Distance of travel [m].
    cfg : moonpies.config.Cfg, optional
        cfg.grav_moon : float
            Lunar gravity [m/s^2].
        cfg.rad_moon : float
            Lunar radius [m].
        cfg.impact_angle : float
            Impact angle [deg].

    Returns
    -------
    velocity : numpy.ndarray
        Ballistic velocity [m/s].

    References
    ----------
    .. [1] Vickery, A. (1986). "Size-velocity distribution of large ejecta 
       fragments." Icarus, 67(2), 224-236. https://doi.org/10/bjjk93

    """    
    theta = np.radians(cfg.impact_angle)
    g = cfg.grav_moon
    rp = cfg.rad_moon
    tan_phi = np.tan(dist / (2 * rp))
    num = g * rp * tan_phi
    denom = (np.sin(theta) * np.cos(theta)) + (np.cos(theta) ** 2 * tan_phi)
    return np.sqrt(num / denom)


def get_bsed_depth(time_arr, df, ej_dists, cfg=CFG):
    """Return ballistic sedimentation depth over time [1]_.

    If more than one ballistic sedimentation event occurs in a timestep, take
    the maximum depth and melt fraction. TODO: make sequential.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater DataFrame.
    ej_dists : numpy.ndarray
        2D array of ejecta distances [m] (Ncrater, Ncoldtrap).
    cfg : moonpies.config.Cfg, optional
        

    Returns
    -------
    depths : numpy.ndarray
        2D array of ballistic sedimentation depths [m] (Ntime, Ncoldtrap).
    melt_fracs : numpy.ndarray
        2D array of melt fractions (Ntime, Ncoldtrap).

    See Also
    --------
    get_mixing_ratio_oberbeck, ejecta_temp, get_melt_frac

    References
    ----------
    .. [1] Zhang, X., Xie, M., & Xiao, Z. (2021). "Thickness of orthopyroxene-
       rich materials of ejecta deposits from the south pole-Aitken basin." 
       Icarus, 358, 114214. https://doi.org/10/gmfc53
    """
    if not cfg.ballistic_sed:
        depths = np.zeros((len(time_arr), len(cfg.coldtrap_names)), cfg.dtype)
        fracs = np.zeros((len(time_arr), len(cfg.coldtrap_names)), cfg.dtype)
        return depths, fracs

    # Get distance, mixing_ratio, volume_frac for each crater to each coldtrap
    mixing_ratio = get_mixing_ratio_oberbeck(ej_dists, cfg)
    ej_temp = ejecta_temp(df, mixing_ratio.shape, cfg)
    melt_frac = get_melt_frac(ej_temp, mixing_ratio, cfg)

    # Convert to time array shape: (Ncrater, Ncoldtrap) -> (Ntime, Ncoldtrap)
    ej_thick_t, _ = get_ejecta_thickness_time(time_arr, df, ej_dists, cfg)
    ages = df.age.values
    mixing_ratio_t = ages2time(time_arr, ages, mixing_ratio, np.nanmax, 0)
    bsed_depths_t = ej_thick_t * mixing_ratio_t  # Petro and Pieters (2004)
    melt_frac_t = ages2time(time_arr, ages, melt_frac, np.nanmax, 0)
    return bsed_depths_t, melt_frac_t


def get_mixing_ratio_oberbeck(ej_distances, cfg=CFG):
    """Return mixing ratio of target to ejecta material [1]_.

    If cfg.mixing_ratio_petro, use the adjustment for large mr from [2].

    Parameters
    ----------
    ej_distances : numpy.ndarray
        2D array of ejecta distances [m] (Ncrater, Ncoldtrap).
    cfg : moonpies.config.Cfg, optional
        cfg.mixing_ratio_a : float
            Mixing ratio factor a [1].
        cfg.mixing_ratio_b : float
            Mixing ratio exponent b [2].
        cfg.mixing_ratio_petro : bool
            If True, adjust large mixing ratios using [2]

    Returns
    -------
    mixing_ratio : numpy.ndarray
        2D array of mixing ratios (target:ejecta) (Ncrater, Ncoldtrap).

    References
    ----------
    .. [1] Oberbeck, V. R. (1975). "The role of ballistic erosion and 
       sedimentation in lunar stratigraphy." Reviews of Geophysics, 13(2), 
       337-362. https://doi.org/10/bcxqfb
    .. [2] Petro, N. E., & Pieters, C. M. (2006). "Modeling the provenance of 
       the Apollo 16 regolith." Journal of Geophysical Research, 111(E9), 
       E09005. https://doi.org/10/dbwzmv
    """
    dist = ej_distances * 1e-3  # [m -> km]
    mr = cfg.mixing_ratio_a * dist**cfg.mixing_ratio_b
    if cfg.mixing_ratio_petro:
        mrv = np.atleast_1d(mr)
        mrv[mrv > 5] = 0.5 * mrv[mrv > 5] + 2.5
        mr = mrv if mrv.ndim > 0 else float(mrv)
    return mr


# Impact gardening module (remove ice by impact overturn)
def remove_ice_overturn(ice_col, ej_col, t, depth, cfg=CFG):
    """Return ice_col gardened to depth below time t.

    Parameters
    ----------
    ice_col : numpy.ndarray
        Ice column [m].
    ej_col : numpy.ndarray
        Ejecta column [m].
    t : int
        Time index.
    depth : float
        Impact gardening depth [m].
    cfg : moonpies.config.Cfg, optional
        cfg.impact_gardening_costello : bool
            If True, use Costello impact gardening model, else use Cannon.

    Returns
    -------
    ice_col : numpy.ndarray
        Ice column [m] with impact gardening applied.

    See Also
    --------
    garden_ice_column, erode_ice_cannon
    """    
    if cfg.impact_gardening_costello:
        ice_col = garden_ice_column(ice_col, ej_col, t, depth)
    else:
        ice_col = erode_ice_cannon(ice_col, ej_col, t, depth)
    return ice_col


def erode_ice_cannon(ice_col, ej_col, t, erosion_depth=0.1, ej_shield=0.4):
    """Return ice_col gardened to erosion_depth below time t [1]_.

    Translated directly from Cannon et al. (2020) MATLAB code [1].

    Parameters
    ----------
    ice_col : numpy.ndarray
        Ice column [m].
    ej_col : numpy.ndarray
        Ejecta column [m].
    t : int
        Time index.
    erosion_depth : float, optional
        Erosion depth [m], by default 0.1
    ej_shield : float, optional
        Ejecta shield thickness [m], by default 0.4

    Returns
    -------
    ice_col : numpy.ndarray
        Ice column [m] with impact gardening applied.
    
    References
    ----------
    .. [1] Cannon, K. M., Deutsch, A. N., Head, J. W., & Britt, D. T. (2020). 
       "Stratigraphy of Ice and Ejecta Deposits at the Lunar Poles."
       Geophysical Research Letters, 47(21). https://doi.org/10/gksdcc

    """    
    # Possible bug: erosion base never updated for adjacent ejecta layers
    # Erosion base is most recent time when ejecta column was > ejecta_shield
    erosion_base = -1
    erosion_base_idx = np.where(ej_col[: t + 1] > ej_shield)[0]
    if len(erosion_base_idx) > 0:
        erosion_base = erosion_base_idx[-1]

    # Garden from top of ice column until ice_to_erode amount is removed
    layer = t
    while erosion_depth > 0 and layer >= 0:
        # Possible bug: t > erosion_base should be layer > erosion_base.
        # - loop doesn't end if we reach erosion base while eroding
        # - loop only ends in else block if we started at erosion_base
        if t > erosion_base:
            ice_in_layer = ice_col[layer]
            if ice_in_layer >= erosion_depth:
                ice_col[layer] -= erosion_depth
            else:
                ice_col[layer] = 0
            erosion_depth -= ice_in_layer
            layer -= 1
        else:
            break
    return ice_col


def garden_ice_column(ice_column, ejecta_column, t, depth, eff=1):
    """Return ice_column gardened to depth, but preserved by ejecta_column.

    Ejecta deposited on last timestep preserves ice. Loop through ice_col and
    ejecta_col until overturn_depth and remove all ice that is encountered.

    Parameters
    ----------
    ice_column : numpy.ndarray
        Ice column [m].
    ejecta_column : numpy.ndarray
        Ejecta column [m].
    t : int
        Time index.
    depth : float
        Impact gardening depth [m].
    eff : float, optional
        Fraction to remove from each layer in [0, 1], by default 1 (all).

    Returns
    -------
    ice_column : numpy.ndarray
        Ice column [m] with impact gardening applied.
    """
    # Travese ice and ejecta column from t down, removing ice, skipping ejecta
    # Loop until we hit the bottom or have gone down depth meters
    # - If ejecta[t] > depth, no ice is removed.
    # Double i so i//2 is current index to garden (odd: ejecta, even: ice)
    i = (2 * t) + 1
    d = 0  # current depth
    while i >= 0 and d < depth:  # and < 2 * len(ice_column):
        if i % 2:
            # Odd i (ejecta): do nothing, add ejecta layer to depth, d
            d += ejecta_column[i // 2]
        else:
            # Even i (ice): remove ice*eff from layer
            removed = ice_column[i // 2] * eff

            # Removing more ice than depth, only remove enough to reach depth
            if (d + removed) > depth:
                removed = depth - d
            ice_column[i // 2] -= removed
            d += ice_column[i // 2]  # Count all ice in layer towards depth
        i -= 1
    return ice_column


# Impact gardening module (Costello et al. 2018, 2020)
def overturn_depth_time(time_arr, cfg=CFG):
    """Return overturn depth [m] at each time.


    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.impact_gardening_costello : bool
            If True, get overturn over time, else use cfg.ice_erosion_rate.
        cfg.ice_erosion_rate : float
            Constant ice erosion rate [m/10 Ma].
        cfg.timestep : float
            Timestep [yr] to scale cfg.ice_erosion_rate.

    Returns
    -------
    overturn_depth : numpy.ndarray
        Overturn depth [m] at each time.

    See Also
    --------
    overturn_depth_costello_time
    """
    if cfg.impact_gardening_costello:
        overturn_t = overturn_depth_costello_time(time_arr, cfg)
    else:
        # Cannon mode assume ice_erosion_rate 0.1 m / Ma gardening at all times
        t_scaling = cfg.timestep / 1e7  # scale from 10 Ma rate
        overturn_t = cfg.ice_erosion_rate * t_scaling * np.ones_like(time_arr)
    return overturn_t


def overturn_depth_costello_time(time_arr, cfg=CFG):
    """Return regolith overturn depth at each time [1]_.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.overturn_ancient_t0 : float
            Time when ancient overturn rate becomes present rate [yr].
        cfg.overturn_depth_present : float
            Present-day overturn depth per cfg.timestep [m].
        cfg.overturn_ancient_slope : float
            Slope of ancient overturn depth [m/yr].

    Returns
    -------
    overturn_depth : numpy.ndarray
        Overturn depth [m] at each time.
    
    References
    ----------
    .. [1] Costello, E. S., Ghent, R. R., Hirabayashi, M., & Lucey, P. G. 
       (2020). "Impact Gardening as a Constraint on the Age, Source, and 
       Evolution of Ice on Mercury and the Moon." Journal of Geophysical 
       Research: Planets, 125(3). https://doi.org/10/gk56kr

    """    
    t_anc = cfg.overturn_ancient_t0
    depth = np.ones_like(time_arr) * cfg.overturn_depth_present
    d_scaling = cfg.overturn_ancient_slope * (time_arr - t_anc) + 1
    depth[time_arr > t_anc] *= d_scaling[time_arr > t_anc]
    return depth


# Impact-delivered ice module
def get_impact_ice(time_arr, df, cfg=CFG, rng=None):
    """Return polar ice thickness [m] from global impacts at each time.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater dataframe.
    cfg : moonpies.config.Cfg, optional
        cfg.impact_ice_basins : bool
            If True, add basin ice contributions
    rng : int or numpy.random.Generator, optional
        Random number generator or seed.

    Returns
    -------
    ice_time : numpy.ndarray
        Polar ice thickness [m] at each time.
    """
    impact_ice_t = np.zeros_like(time_arr)
    impact_ice_t += get_micrometeorite_ice(time_arr, cfg)
    impact_ice_t += get_small_impactor_ice(time_arr, cfg)
    impact_ice_t += get_small_simple_crater_ice(time_arr, cfg)
    impact_ice_t += get_large_simple_crater_ice(time_arr, cfg, rng)
    impact_ice_t += get_complex_crater_ice(time_arr, cfg, rng)
    impact_ice_basins_t = get_basin_ice(time_arr, df, cfg, rng)
    if cfg.impact_ice_basins:
        # get_basin_ice is run every time for repro, but only add if needed
        impact_ice_t += impact_ice_basins_t
    return impact_ice_t


def get_comet_cfg(cfg):
    """Return config with impact parameters updated to comet values.

    Sets cfg.is_comet, cfg.hydrated_wt_pct, cfg.impactor_density, 
    cfg.impact_mass_retained to comet counterparts.

    Parameters
    ----------
    cfg : moonpies.config.Cfg
        MoonPIES config object.

    Returns
    -------
    comet_cfg : moonpies.config.Cfg
        MoonPIES config object with comet parameters set.
    """    
    # TODO: Explicitly handle comets instead of cludge with impact cfg params.
    # Define comet_cfg with comet parameters to supply to get_impact_ice
    comet_cfg_dict = cfg.to_dict()
    comet_cfg_dict["is_comet"] = True  # Use comet impact speeds
    comet_cfg_dict["hydrated_wt_pct"] = cfg.comet_hydrated_wt_pct
    comet_cfg_dict["impactor_density"] = cfg.comet_density
    comet_cfg_dict["impact_mass_retained"] = cfg.comet_mass_retained
    return config.from_dict(comet_cfg_dict)


def get_impact_ice_comet(time_arr, df, cfg, rng=None):
    """Return polar ice thickness [m] from comet impacts at each time.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater dataframe.
    cfg : moonpies.config.Cfg
        MoonPIES config object.
    rng : int or numpy.random.Generator, optional
        Random number generator or seed.

    Returns
    -------
    ice_time : numpy.ndarray
        Comet polar ice thickness [m] at each time.
    """    
    # Repeat get_impact_ice with comet cfg
    comet_cfg = get_comet_cfg(cfg)
    comet_ice_t = get_impact_ice(time_arr, df, comet_cfg, rng)
    return comet_ice_t


def get_micrometeorite_ice(time_arr, cfg=CFG):
    """
    Return ice thickness [m] delivered to pole due to micrometeorites vs time.

    Returns
    -------
    mm_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    mm_ice_mass = ice_micrometeorites(time_arr, cfg)
    mm_ice_t = get_ice_thickness(mm_ice_mass, cfg)
    return mm_ice_t


def get_small_impactor_ice(time_arr, cfg=CFG):
    """
    Return ice thickness [m] delivered to pole due to small impactors vs time.

    Returns
    -------
    si_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    impactor_diams, impactors = get_small_impactor_pop(time_arr, cfg)
    si_ice_mass = ice_small_impactors(impactor_diams, impactors, cfg)
    si_ice_t = get_ice_thickness(si_ice_mass, cfg)
    return si_ice_t


def get_small_simple_crater_ice(time_arr, cfg=CFG):
    """
    Return ice thickness [m] delivered to pole by small simple crater impacts.

    Returns
    -------
    ssc_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    crater_diams, n_craters_t, sfd_prob = get_crater_pop(time_arr, "c", cfg)
    n_craters = n_craters_t * sfd_prob
    ssc_ice_mass = ice_small_craters(crater_diams, n_craters, "c", cfg)
    ssc_ice_t = get_ice_thickness(ssc_ice_mass, cfg)
    return ssc_ice_t


def get_large_simple_crater_ice(time_arr, cfg=CFG, rng=None):
    """
    Return ice thickness [m] delivered to pole by large simple crater impacts.

    Returns
    -------
    lsc_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    lsc_ice_mass = get_ice_stochastic(time_arr, "d", cfg, rng)
    lsc_ice_t = get_ice_thickness(lsc_ice_mass, cfg)
    return lsc_ice_t


def get_complex_crater_ice(time_arr, cfg=CFG, rng=None):
    """
    Return ice thickness [m] delivered to pole by complex crater impacts.

    Returns
    -------
    cc_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    cc_ice_mass = get_ice_stochastic(time_arr, "e", cfg, rng)
    cc_ice_t = get_ice_thickness(cc_ice_mass, cfg)
    return cc_ice_t


def get_basin_ice(time_arr, df, cfg=CFG, rng=None):
    """
    Return ice thickness [m] delivered to pole by basin impacts vs time.

    Returns
    -------
    b_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    b_ice_mass = ice_basins(df, time_arr, cfg, rng)
    b_ice_t = get_ice_thickness(b_ice_mass, cfg)
    return b_ice_t


def get_ice_stochastic(time_arr, regime, cfg=CFG, rng=None):
    """
    Return ice mass [kg] delivered to pole at each time from stochastic 
    delivery by crater-forming impactors.

    Parameters
    ----------
    time_arr (arr): Array of times [yr] to calculate ice thickness at.
    regime (str): Regime of ice production.
    cfg (Cfg): Config object.
    rng (int or numpy.random.Generator): Seed or random number generator.

    Returns
    -------
    ice_t (arr): Ice mass [kg] delivered to pole at each time.
    """
    rng = _rng(rng)
    diams, num_craters_t, sfd_prob = get_crater_pop(time_arr, regime, cfg)

    # Round to integer number of craters at each time
    num_craters_t = probabilistic_round(num_craters_t, rng=rng)

    # Get ice thickness for each time, randomizing crater diam and impact speed
    ice_mass_t = np.zeros_like(time_arr)
    for i, num_craters in enumerate(num_craters_t):
        # Randomly resample crater diams from sfd prob with replacement
        rand_diams = rng.choice(diams, num_craters, p=sfd_prob)

        # Randomly subset impacts to only hydrated
        hyd = get_random_hydrated_craters(len(rand_diams), cfg, rng)
        hyd_diams = rand_diams[hyd]

        speeds = get_random_impactor_speeds(len(hyd_diams), cfg, rng)
        ice_mass_t[i] = ice_large_craters(hyd_diams, speeds, regime, cfg)
    return ice_mass_t


def read_ballistic_hop_csv(bhop_csv):
    """
    Return dict of ballistic hop efficiency of each coldtrap in bhop_csv.
    """
    return pd.read_csv(bhop_csv, header=0, index_col=0)


def get_ballistic_hop_coldtraps(coldtraps, cfg=CFG):
    """
    Return ballistic hop efficiency as row vector of coldtraps [1,N].

    Parameters
    ----------
    coldtraps (list of str): Desired coldtraps.
    cfg (Cfg): Configuration object.

    Returns
    -------
    bhop (arr): Ballistic hop efficiency of coldtraps.
    """
    bhop = read_ballistic_hop_csv(cfg.bhop_csv_in)
    return bhop.loc[coldtraps].values.T


def ice_micrometeorites(time, cfg=CFG):
    """
    Return ice from micrometeorites (Regime A, Cannon 2020).

    Multiply total_mm_mass / yr by timestep and scale by assumed hydration %
    and scale by ancient flux relative to today.

    Unlike larger impactors, we DO NOT assume ctype composition and fraction of
    hydrated ctypes and also DO NOT scale by asteroid retention rate.
    TODO: are these reasonable assumptions?
    """
    # Scale by impact flux relative to today
    mm_mass_t = cfg.mm_mass_rate * impact_flux_scaling(time)

    if cfg.is_comet:
        # When comet micrometeorites (100% comets)
        ice_ret = cfg.comet_hydrated_wt_pct * cfg.comet_mass_retained
    elif cfg.impact_ice_comets:
        # When asteroid micrometeorites but comets included
        ice_ret = 0
    else:
        # When asteroid micrometeorites but comets excluded (100% asteroids)
        ice_ret = cfg.hydrated_wt_pct * cfg.impact_mass_retained
    mm_ice_t = mm_mass_t * ice_ret * cfg.timestep
    return mm_ice_t


def ice_small_impactors(diams, impactors_t, cfg=CFG):
    """
    Return ice mass [kg] from small impactors (Regime B, Cannon 2020) given
    impactor diams and number of impactors over time.
    """
    impactor_masses = diam2vol(diams) * cfg.impactor_density
    impactor_mass_t = np.sum(impactors_t * impactor_masses, axis=1)
    impactor_ice_t = impactor_mass2water(impactor_mass_t, cfg)
    return impactor_ice_t


def ice_small_craters(
    crater_diams,
    ncraters,
    regime,
    cfg=CFG,
):
    """
    Return ice from simple craters, steep branch (Regime C, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams, cfg.impact_speed_mean, regime, cfg)
    impactor_masses = diam2vol(impactor_diams) * cfg.impactor_density  # [kg]
    total_impactor_mass = np.sum(impactor_masses * ncraters, axis=1)
    total_impactor_water = impactor_mass2water(total_impactor_mass, cfg)
    return total_impactor_water


def ice_large_craters(crater_diams, impactor_speeds, regime, cfg=CFG):
    """
    Return ice from simple/complex craters, shallow branch of sfd
    (Regime D-E, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams, impactor_speeds, regime, cfg)
    impactor_masses = diam2vol(impactor_diams) * cfg.impactor_density  # [kg]

    # Find ice mass assuming hydration wt% and retention based on speed
    ice_retention = ice_retention_factor(impactor_speeds, cfg)
    ice_masses = impactor_masses * cfg.hydrated_wt_pct * ice_retention
    return np.sum(ice_masses)


def ice_basins(df, time_arr, cfg=CFG, rng=None):
    """Return ice mass [kg] from basin impacts vs time."""
    # Get basin ice from hydrated ctypes and cometary basins
    itype = "hyd_ctype"  # icy_impactor flag (see get_random_icy_basins)
    if cfg.is_comet:
        itype = "comet"

    basin_ice_t = np.zeros_like(time_arr)

    bdf = df[df.isbasin & (df.icy_impactor == itype)].reset_index(drop=True)
    basin_diams = 2 * bdf.rad.values
    impactor_speeds = get_random_impactor_speeds(len(bdf), cfg, rng)

    # Get ice mass of each basin impactor
    ice_masses = np.zeros_like(basin_diams)
    for i, (diam, speed) in enumerate(zip(basin_diams, impactor_speeds)):
        ice_masses[i] = ice_large_craters(diam, speed, "f", cfg)

    # Insert each basin ice mass to its position in time_arr
    ages = bdf.age.values
    basin_ice_t = ages2time(time_arr, ages, ice_masses, np.sum, 0)
    return basin_ice_t


def ice_retention_factor(speeds, cfg):
    """
    Return ice retained in impact, given impactor speeds (Ong et al. 2010).

    For speeds < 10 km/s, retain 50% (Svetsov & Shuvalov 2015 via Cannon 2020).
    For speeds >= 10 km/s, use fit Fig 2 (Ong et al. 2010 via Cannon 2020)
    """
    speeds = speeds * 1e-3  # [m/s] -> [km/s]
    retained = np.ones_like(speeds)
    if cfg.mode == "cannon":
        # Cannon et al. (2020) ds02
        retained[speeds >= 10] = 36.26 * np.exp(-0.3464 * speeds[speeds >= 10])
    elif cfg.mode == "moonpies":
        # Fit to Fig 2. (Ong et al. 2010), negligible retention > 45 km/s
        retained[speeds >= 10] = 1.66e4 * speeds[speeds >= 10] ** -4.16
    if not cfg.is_comet:
        retained[speeds < 10] = 0.5  # Cap at 50% retained for asteroids
    retained[retained < 0] = 0
    return retained


def impactor_mass2water(impactor_mass, cfg):
    """
    Return water [kg] from impactor mass [kg] using assumptions of Cannon 2020:
        - 36% of impactors are C-type (Jedicke et al., 2018)
        - 2/3 of C-types are hydrated (Rivkin, 2012)
        - Hydrated impactors are 10% water by mass (Cannon et al., 2020)
        - 16% of asteroid mass retained on impact (Ong et al., 2010)
     for comets:
        - assume 50% of comet hydrated
        - 6.5% of comet mass retained on impact (Ong et al., 2010)
    """
    if cfg.is_comet:
        mass_h2o = (
            cfg.comet_hydrated_wt_pct * impactor_mass * cfg.comet_mass_retained
        )
    else:
        mass_h2o = (
            cfg.ctype_frac
            * cfg.ctype_hydrated
            * cfg.hydrated_wt_pct
            * impactor_mass
            * cfg.impact_mass_retained
        )
    return mass_h2o


# Stochasticity and randomness
def _rng(rng):
    """Shorthand wrapper for np.random.default_rng."""
    return np.random.default_rng(rng)


def get_rng(cfg=CFG):
    """Return numpy random number generator from given seed in rng."""
    return _rng(cfg.seed)


def randomize_crater_ages(df, timestep, rng=None):
    """
    Return ages randomized with a truncated Gaussian between age_low, age_upp.

    Parameters
    ----------
    df (DataFrame): Crater dataframe with age, age_low, age_upp columns.
    timestep (int): Timestep [yr].
    rng (int or numpy.random.Generator): Seed or random number generator.
    """
    rng = _rng(rng)

    # Set standard deviation and get scaling params for truncnorm
    sig = df[["age_low", "age_upp"]].mean(axis=1) / 2
    a = -df.age_low / sig
    b = df.age_upp / sig

    # Truncated normal, returns vector of randomized ages
    new_ages = stats.truncnorm.rvs(a, b, df.age, sig, random_state=rng)
    df["age"] = round_to_ts(new_ages, timestep)
    df = df.sort_values("age", ascending=False).reset_index(drop=True)
    return df


def get_random_hydrated_craters(n, cfg=CFG, rng=None):
    """
    Return crater diams of hydrated craters from random distribution.

    Parameters
    ----------
    n (int): Number of craters.
    cfg (Cfg): Config object.
    rng (int or numpy.random.Generator): Seed or random number generator.
    """
    rng = _rng(rng)
    rand_arr = rng.random(size=n)
    if cfg.is_comet:
        # Get fraction of cometary impactors (assumed always hydrated)
        hydration_prob = cfg.comet_ast_frac
    elif cfg.impact_ice_comets:
        # Get fraction of asteroids, in runs that contain comets
        ast_frac = 1 - cfg.comet_ast_frac
        hydration_prob = ast_frac * cfg.ctype_frac * cfg.ctype_hydrated
    else:
        # Get fraction of ctype asteroids time prob of being hydrated
        hydration_prob = cfg.ctype_frac * cfg.ctype_hydrated
    hydrated_inds = rand_arr < hydration_prob
    return hydrated_inds


def get_random_hydrated_basins(n, cfg=CFG, rng=None):
    """
    Return indicess of hydrated basins from random distribution with asteroids
    and comets mutually exclusive.

    Parameters
    ----------
    n (int): Number of basins.
    cfg (Cfg): Config object.
    rng (int or numpy.random.Generator): Seed or random number generator.
    """
    rng = _rng(rng)
    rand_arr = rng.random(size=n)
    # Fraction of ctype hydrated basin impactors
    if cfg.impact_ice_comets:
        ast_frac = 1 - cfg.comet_ast_frac
        hydration_prob = ast_frac * cfg.ctype_frac * cfg.ctype_hydrated
    else:
        hydration_prob = cfg.ctype_frac * cfg.ctype_hydrated
    ctype_inds = rand_arr < hydration_prob

    # Fraction of cometary basins (use same rand_arr and pick unique inds)
    hydration_prob = cfg.comet_ast_frac
    comet_inds = rand_arr > (1 - hydration_prob)
    return ctype_inds, comet_inds


def get_random_impactor_speeds(n, cfg=CFG, rng=None):
    """
    Return n impactor speeds from normal distribution about mean, sd.

    Parameters
    ----------
    n (int): Number of impactors.
    cfg (Cfg): Config object.
    rng (int or numpy.random.Generator): Seed or random number generator.
    """
    # Randomize impactor speeds with Gaussian around mean, sd
    rng = _rng(rng)
    rv = comet_speed_rv(cfg) if cfg.is_comet else asteroid_speed_rv(cfg)
    return rv.rvs(size=n, random_state=rng)


@lru_cache(maxsize=1)
def asteroid_speed_rv(cfg=CFG):
    """
    Return asteroid speed random variable object from scipy.stats.truncnorm.
    """
    # Set scaling param for stats.truncnorm (min impact speed is escape vel)
    a = (cfg.escape_vel - cfg.impact_speed_mean) / cfg.impact_speed_sd
    b = np.inf
    return stats.truncnorm(a, b, cfg.impact_speed_mean, cfg.impact_speed_sd)


@lru_cache(maxsize=1)
def comet_speed_rv(cfg=CFG):
    """
    Return comet speed random variable object from scipy.stats.truncnorm.
    """
    # Set scaling param for stats.truncnorm
    a = -cfg.comet_speed_min / cfg.jfc_speed_sd
    b = cfg.comet_speed_max / cfg.jfc_speed_sd
    jfc = stats.truncnorm(a, b, loc=cfg.jfc_speed_mean, scale=cfg.jfc_speed_sd)
    a = -cfg.comet_speed_min / cfg.lpc_speed_sd
    b = cfg.comet_speed_max / cfg.lpc_speed_sd
    lpc = stats.truncnorm(a, b, loc=cfg.lpc_speed_mean, scale=cfg.lpc_speed_sd)
    w = [cfg.jfc_frac, cfg.lpc_frac]
    return rv_mixture([jfc, lpc], weights=w)


# Crater/impactor size-frequency helpers
@lru_cache(6)
def neukum(diam, cfg=CFG):
    """
    Return number of craters per m^2 per yr at diam [m] (eqn. 2, Neukum 2001).

    Eqn 2 expects diam [km], returns N [km^-2 Ga^-1].

    Parameters
    ----------
    diam (float): Crater diameter [m].
    cfg (Cfg): Config object.
    """
    if cfg.neukum_pf_new:
        a_vals = cfg.neukum_pf_a_2001
    else:
        a_vals = cfg.neukum_pf_a_1983
    diam = diam * 1e-3  # [m] -> [km]
    j = np.arange(len(a_vals))
    ncraters = 10 ** np.sum(a_vals * np.log10(diam) ** j)  # [km^-2 Ga^-1]
    return ncraters * 1e-6 * 1e-9  # [km^-2 Ga^-1] -> [m^-2 yr^-1]


def get_impactors_brown(mindiam, maxdiam, timestep, cfg=CFG):
    """
    Return number of impactors per yr in range mindiam, maxdiam (Brown et al.
    2002) and scale by Earth-Moon impact ratio (Mazrouei et al. 2019).
    """
    c0, d0 = cfg.brown_c0, cfg.brown_d0
    n_impactors_gt_low = 10 ** (c0 - d0 * np.log10(mindiam))  # [yr^-1]
    n_impactors_gt_high = 10 ** (c0 - d0 * np.log10(maxdiam))  # [yr^-1]
    n_impactors_earth_yr = n_impactors_gt_low - n_impactors_gt_high
    n_impactors_moon = n_impactors_earth_yr * timestep / cfg.earth_moon_ratio
    if cfg.is_comet:
        n_impactors_moon = n_impactors_moon * cfg.comet_ast_frac
    elif cfg.impact_ice_comets:
        n_impactors_moon = n_impactors_moon * (1 - cfg.comet_ast_frac)
    return n_impactors_moon


def get_crater_pop(time_arr, regime, cfg=CFG):
    """
    Return crater population assuming continuous (non-stochastic, regime c).
    """
    mindiam, maxdiam, step, slope = cfg.impact_regimes[regime]
    diams, sfd_prob = get_crater_sfd(mindiam, maxdiam, step, slope, cfg.dtype)
    num_craters_t = num_craters_chronology(mindiam, maxdiam, time_arr, cfg)
    num_craters_t = np.atleast_1d(num_craters_t)[:, np.newaxis]  # make col vec
    if cfg.is_comet:
        num_craters_t = num_craters_t * cfg.comet_ast_frac
    elif cfg.impact_ice_comets:
        num_craters_t = num_craters_t * (1 - cfg.comet_ast_frac)
    return diams, num_craters_t, sfd_prob


def num_craters_chronology(mindiam, maxdiam, time, cfg=CFG):
    """
    Return number of craters [m^-2 yr^-1] mindiam and maxdiam at each time.
    """
    # Compute number of craters from neukum pf
    fmax = neukum(mindiam, cfg)
    fmin = neukum(maxdiam, cfg)
    count = (fmax - fmin) * cfg.sa_moon * cfg.timestep

    # Scale count by impact flux relative to present day flux
    num_craters = count * impact_flux_scaling(time)
    return num_craters


def get_small_impactor_pop(time_arr, cfg=CFG):
    """
    Return population of impactors and number in regime B.

    Use constants and eqn. 3 from Brown et al. (2002) to compute N craters.
    """
    min_d, max_d, step, sfd_slope = cfg.impact_regimes["b"]
    diams, sfd_prob = get_crater_sfd(min_d, max_d, step, sfd_slope, cfg.dtype)
    n_impactors = get_impactors_brown(min_d, max_d, cfg.timestep, cfg)

    # Scale n_impactors by historical impact flux, csfd shape: (NT, Ndiams)
    flux_scaling = impact_flux_scaling(time_arr)
    n_impactors_t = n_impactors * flux_scaling[:, None] * sfd_prob
    return diams, n_impactors_t


@lru_cache(4)
def get_crater_sfd(dmin, dmax, step, sfd_slope, dtype=None):
    """
    Return diam_array and sfd_prob. This func makes it easier to cache both.
    """
    diam_array = get_diam_array(dmin, dmax, step, dtype)
    sfd_prob = get_sfd_prob(diam_array, sfd_slope)
    return diam_array, sfd_prob


def get_sfd_prob(diams, sfd_slope):
    """
    Return size-frequency distribution probability given diams, sfd slope.
    """
    sfd = diams**sfd_slope
    return sfd / np.sum(sfd)


def get_diam_array(dmin, dmax, step, dtype=None):
    """Return array of diameters based on diameters from dmin, dmax, dstep."""
    n = int((dmax - dmin) / step)
    return np.linspace(dmin, dmax, n + 1, dtype=dtype)


def impact_flux_scaling(time):
    """
    Return the factor to scale impact flux by in the past vs. present day.

    Take ratio of historical and present day impact fluxes from Ivanov 2008).
    Parameters
    ----------
    time (num or arr): Time [year] before present.

    Returns
    -------
    scaling_factor (num): Impact flux scaling factor.
    """
    scaling_factor = impact_flux(time) / impact_flux(0)
    return scaling_factor


def impact_flux(time):
    """Return impact flux at time [yrs] (Derivative of eqn. 1, Ivanov 2008)."""
    time = time * 1e-9  # [yrs -> Ga]
    flux = 6.93 * 5.44e-14 * (np.exp(6.93 * time)) + 8.38e-4  # [n/Ga]
    return flux * 1e-9  # [Ga^-1 -> yrs^-1]


# Crater-impactor scaling laws
def diam2len(
    diams,
    speeds,
    regime,
    cfg=CFG,
):
    """
    Return size of impactors based on crater diams and impactor speeds.

    Different crater regimes are scaled via the following scaling laws:
    - regime=='c': (Prieur et al., 2017)
    - regime=='d': (Collins et al., 2005)
    - regime=='e': (Johnson et al., 2016)

    Parameters
    ----------
    diams (arr): Crater diameters [m].
    speeds (arr): Impactor speeds [m/s].
    regime (str): Crater scaling regime ('c', 'd', or 'e').

    Returns
    -------
    lengths (arr): Impactor diameters [m].
    """
    t_diams = final2transient(diams, cfg)
    if regime == "c":
        impactor_length = diam2len_prieur(
            tuple(t_diams),
            speeds,
            cfg.impactor_density,
            cfg.target_density,
            cfg.grav_moon,
            cfg.dtype,
        )
    elif regime == "d":
        impactor_length = diam2len_collins(
            t_diams,
            speeds,
            cfg.impactor_density,
            cfg.target_density,
            cfg.grav_moon,
            cfg.impact_angle,
        )
    elif regime in ("e", "f"):
        impactor_length = diam2len_johnson(
            t_diams,
            speeds,
            cfg.impactor_density,
            cfg.target_density,
            cfg.grav_moon,
            cfg.impact_angle,
            cfg.simple2complex,
        )
    # elif regime == "f": TODO: Implement Potter basin scaling
    else:
        raise ValueError(f"Invalid regime {regime} in diam2len")
    return impactor_length


def final2transient(diam, cfg=CFG):
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
    # Init arrays and transition diams
    diam = np.atleast_1d(diam)
    t_diam = np.zeros_like(diam)
    ds2c = cfg.simple2complex
    dc2pr = cfg.complex2peakring

    # Simple craters
    mask = diam <= ds2c
    t_diam[mask] = f2t_melosh(diam[mask], simple=True)

    # Complex craters
    mask = (diam > ds2c) & (diam <= dc2pr)
    t_diam[mask] = f2t_melosh(diam[mask], simple=False)

    # Basins
    mask = diam > dc2pr
    t_diam[mask] = f2t_croft(diam[mask])  # TODO: make potter?
    return t_diam


def f2t_melosh(diam, simple=True, ds2c=18e3, gamma=1.25, eta=0.13):
    """
    Return transient crater diameter(s) from final crater diam (Melosh 1989).

    Parameters
    ----------
    diams (num or array): final crater diameters [m]
    simple (bool): if True, use simple scaling law (Melosh 1989).
    """
    if simple:  # Simple crater scaling
        t_diam = diam / gamma
    else:  # Complex crater scaling
        t_diam = (1 / gamma) * (diam * ds2c**eta) ** (1 / (1 + eta))
    return t_diam


def f2t_croft(diam, ds2c=18.7e3, eta=0.18):
    """
    Return transient crater diameter(s) fom final crater diam (Croft 1985).
    """
    return (diam * ds2c**eta) ** (1 / (1 + eta))


@lru_cache(1)
def diam2len_prieur(
    t_diam,
    v=20e3,
    rho_i=1300,
    rho_t=1500,
    g=1.62,
    dtype=None,
):
    """
    Return impactor length from input diam using Prieur et al. (2017) method.

    Note: Interpolates impactor lengths from the forward Prieur impactor length
    to transient crater diameter equation.

    Parameters
    ----------
    t_diam (num or array): transient crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)

    Returns
    -------
    impactor_length (num): impactor diameter [m]
    """
    i_lengths = np.linspace(t_diam[0] / 100, t_diam[-1], 1000, dtype=dtype)
    i_masses = rho_i * diam2vol(i_lengths)
    # Prieur impactor len to crater diam equation
    numer = 1.6 * (1.61 * g * i_lengths / v**2) ** -0.22
    denom = (rho_t / i_masses) ** (1 / 3)
    t_diams = numer / denom

    # Interpolate to back out impactor len from diam
    impactor_length = np.interp(t_diam, t_diams, i_lengths)
    return impactor_length


def diam2len_collins(
    t_diam,
    v=20e3,
    rho_i=1300,
    rho_t=1500,
    g=1.62,
    theta=45,
):
    """
    Return impactor length from input diam using Collins et al. (2005) method.

    Parameters
    ----------
    t_diam (num or array): transient crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)
    theta (num): impact angle (degrees)

    Returns
    -------
    impactor_length (num): impactor diameter [m]
    """
    cube_root_theta = np.sin(np.deg2rad(theta)) ** (1 / 3)
    denom = (
        1.161
        * (rho_i / rho_t) ** (1 / 3)
        * v**0.44
        * g**-0.22
        * cube_root_theta
    )
    impactor_length = (t_diam / denom) ** (1 / 0.78)
    return impactor_length


def diam2len_johnson(
    diam,
    v=20e3,
    rho_i=1300,
    rho_t=1500,
    g=1.62,
    theta=45,
    ds2c=18e3,
):
    """
    Return impactor length from final crater diam using Johnson et al. (2016).

    Parameters
    ----------
    diam (num or array): crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)
    theta (num): impact angle (degrees)

    Returns
    -------
    impactor_length (num): impactor diameter [m]
    """
    sin_theta = np.sin(np.deg2rad(theta))
    denom = (
        1.52
        * (rho_i / rho_t) ** 0.38
        * v**0.5
        * g**-0.25
        * ds2c**-0.13
        * sin_theta**0.38
    )
    impactor_length = (diam / denom) ** (1 / 0.88)
    return impactor_length


# Thermal module
def specific_heat_capacity(T, cfg=CFG):
    """
    Return specific heat capacity [J/kg/K] at temperature T [K] for lunar
    regolith (Hayne et al., 2017).

    Parameters
    ----------
    T (num or array): temperature [K]
    cfg (Cfg): Config object with specific_heat_coeffs
    """
    c0, c1, c2, c3, c4 = cfg.specific_heat_coeffs
    return c0 + c1 * T + c2 * T**2 + c3 * T**3 + c4 * T**4


# Gridded outputs
def get_grid_outputs(df, grdx, grdy, cfg=CFG):
    """Return gridded age of most recent impact and ejecta thickness.

    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    grdx : numpy.ndarray
        Grid x-coordinates [m].
    grdy : numpy.ndarray
        Grid y-coordinates [m].
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.

    Returns
    -------
    age_grid : numpy.ndarray
        Gridded ages of last impact [yr] (Ny, Nx).
    thickness_grid : numpy.ndarray
        Gridded ejecta thickness [m] (Ny, Nx, Ncoldtraps).

    See Also
    --------
    get_age_grid, get_gc_dist_grid, get_ejecta_thickness_matrix
    """    
    # Age of most recent impact (2D array: NX, NY)
    age_grid = get_age_grid(df, grdx, grdy, cfg)

    # Great circle distance from each crater to grid (3D array: NX, NY, NC)
    dist_grid = get_gc_dist_grid(df, grdx, grdy, cfg.dtype)

    # Ejecta thickness produced by each crater on grid (3D array: NX, NY, NC)
    ej_thick_grid = get_ejecta_thickness_matrix(df, dist_grid, cfg)
    # TODO: ballistic sed depth, kinetic energy, etc
    return age_grid, ej_thick_grid


def get_gc_dist_grid(df, grdx, grdy, cfg=CFG):
    """Return gridded great circle distance from each crater

    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    grdx : numpy.ndarray
        Grid x-coordinates [m].
    grdy : numpy.ndarray
        Grid y-coordinates [m].
    cfg : moonpies.config.Cfg, optional
        

    Returns
    -------
    distance_grid : numpy.ndarray
        Gridded from each crater in df [m] (Ny, Nx, Ncrater).

    See Also
    --------
    gc_dist
    """
    ny, nx = grdy.shape[0], grdx.shape[1]
    lat, lon = xy2latlon(grdx, grdy, cfg.rad_moon)
    grd_dist = np.zeros((nx, ny, len(df)), dtype=cfg.dtype)
    for i in range(len(df)):
        clon, clat = df.iloc[i][["lon", "lat"]]
        grd_dist[:, :, i] = gc_dist(clon, clat, lon, lat)
    return grd_dist


def get_age_grid(df, grdx, grdy, cfg=CFG):
    """Return final surface age of each grid point after all craters formed.

    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    grdx : numpy.ndarray
        Grid x-coordinates [m].
    grdy : numpy.ndarray
        Grid y-coordinates [m].
    cfg : moonpies.config.Cfg, optional
        cfg.timestart : float
            Start time [yr].
        cfg.rad_moon : float
            Lunar radius [m].

    Returns
    -------
    age_grid : numpy.ndarray
        Gridded ages of last impact [yr] (Ny, Nx).
    """
    def update_age(age_grid, crater, grdx, grdy, rp):
        """Return new age grid, update points interior to crater with age."""
        x, y = latlon2xy(crater.lat, crater.lon, rp)
        rad = crater.rad
        crater_mask = (np.abs(grdx - x) < rad) * (np.abs(grdy - y) < rad)
        age_grid[crater_mask] = crater.age
        return age_grid
    ny, nx = grdy.shape[0], grdx.shape[1]
    age_grid = cfg.timestart * np.ones((ny, nx), dtype=cfg.dtype)
    for _, crater in df.iterrows():
        age_grid = update_age(age_grid, crater, grdx, grdy, cfg.rad_moon)
    return age_grid


# Format and export model results
def make_strat_col(time, ice, ejecta, ej_srcs, age, cfg=CFG):
    """
    Return stratigraphy column of ice and ejecta.

    Parameters
    ----------
    time (array): Time [yr]
    ice (array): Ice thickness [m]
    ejecta (array): Ejecta thickness [m]
    ej_srcs (array): Labels for ejecta layer sources
    age (num): Age at base of strat col [yr]

    Returns
    -------
    ej_col (DataFrame): Ejecta columns vs. time
    ice_col (array): Ice columns vs. time
    strat_cols (dict of DataFrame)
    """
    # Label rows by ice (no ejecta) or ejecta_source
    label = np.empty(len(time), dtype=object)
    label[ice > cfg.thickness_min] = "Ice"
    label[ejecta > cfg.thickness_min] = ej_srcs[ejecta > cfg.thickness_min]
    label[label == ""] = "Minor source(s)"

    # Make stratigraphy DataFrame
    data = np.array([time, ice, ejecta]).T
    cols = ["time", "ice", "ejecta"]
    sdf = pd.DataFrame(data, columns=cols, dtype=time.dtype)
    sdf["label"] = label

    # Exclude layers before age of crater
    if cfg.strat_after_age:
        sdf = sdf[sdf.time < age + rtol(age, cfg.rtol)].reset_index(drop=True)
        # Insert empty row labelling formation age of strat column
        sdf = pd.concat([
            pd.DataFrame([[sdf.iloc[0].time, 0, 0, "Formation age"]
            ], columns=sdf.columns), sdf], ignore_index=True)

    # Remove rows with no label (no ice or ejecta)
    sdf = sdf.dropna(subset=["label"])

    # Combine adjacent rows with same label into layers
    strat = merge_adjacent_strata(sdf)

    # Compute depth of each layer (reverse cumulative sum of ice and ejecta)
    strat["depth"] = np.cumsum((strat.ice + strat.ejecta)[::-1])[::-1]

    # Add ice / ejecta percentage of each layer
    strat["icepct"] = np.round(100 * strat.ice / (strat.ice + strat.ejecta), 4)
    return strat


def merge_adjacent_strata(strat, agg=None):
    """Return strat_col with adjacent rows with same label merged."""
    isadj = (strat.label != strat.label.shift()).cumsum()
    if agg is None:
        agg = {"ice": "sum", "ejecta": "sum", "label": "last", "time": "last"}
    return strat.groupby(["label", isadj], as_index=False, sort=False).agg(agg)


def format_csv_outputs(strat_cols, time_arr, df, cfg=CFG):
    """
    Return all formatted model outputs and write to outpath, if specified.

    Returns
    -------
    ejecta: pandas.DataFrame
        Ejecta thickness [m] (rows: time, cols: cold traps)
    ice: pandas.DataFrame
        Ice thickness [m] (rows: time, cols: cold traps)
    strat_dict: dict of str:pandas.DataFrame
        Dict of stratigraphy column DataFrames; keys: coldtrap names, values:
        dataframe (rows: layers, cols: ice, ejecta, source, time, depth, ice%)
    """
    ej_dict = {"time": time_arr}
    ice_dict = {"time": time_arr}
    strat_dfs = {}
    ej_source_all = []
    cdf = df.set_index('cname').loc[list(cfg.coldtrap_names), 'age']
    for cname, age in cdf.iteritems():
        ice_t, ej_t, ej_srcs = strat_cols[cname]
        ej_dict[cname] = ej_t
        ice_dict[cname] = ice_t
        strat_dfs[cname] = make_strat_col(time_arr, ice_t, ej_t, ej_srcs, age, cfg)
        ej_source_all.append(ej_srcs)

    # Convert to DataFrames
    ej_cols_df = pd.DataFrame(ej_dict)
    ejecta_labels = get_all_labels(np.stack(ej_source_all, axis=1))
    ej_cols_df.insert(1, "ej_sources", ejecta_labels)
    ice_cols_df = pd.DataFrame(ice_dict)

    return ej_cols_df, ice_cols_df, strat_dfs


def get_all_labels(label_array):
    """Return all unique labels from label_array."""
    all_labels = []
    for label_col in label_array:
        all_labels_str = ",".join(label_col)
        unique_labels = set(all_labels_str.split(","))
        all_labels.append(",".join(unique_labels).strip(","))
    return all_labels


def format_save_outputs(strat_cols, time_arr, df, cfg=CFG):
    """
    Format dataframes and save outputs based on write / write_npy in cfg.
    """
    vprint(cfg, "Formatting outputs")
    ejdf, icedf, strat_dfs = format_csv_outputs(strat_cols, time_arr, df, cfg)
    if cfg.write:
        # Save config file
        vprint(cfg, f"Saving outputs to {cfg.out_path}")
        save_outputs([cfg], [cfg.config_py_out])

        # Save coldtrap strat column dataframes
        fnames = []
        dfs = []
        for coldtrap, strat in strat_dfs.items():
            fnames.append(os.path.join(cfg.out_path, f"strat_{coldtrap}.csv"))
            dfs.append(strat)
        save_outputs(dfs, fnames)

        # Save raw ice and ejecta column vs time dataframes
        save_outputs([ejdf, icedf], [cfg.ej_t_csv_out, cfg.ice_t_csv_out])
        print(f"Outputs saved to {cfg.out_path}")

    if cfg.write_npy:
        # Note: Only compute these on demand (expensive to do every run)
        # Age grid is age of most recent impact (2D array: NX, NY)
        vprint(cfg, "Computing gridded outputs...")
        grdy, grdx = get_grid_arrays(cfg)
        grd_outputs = get_grid_outputs(df, grdx, grdy, cfg)
        npy_fnames = (cfg.agegrd_npy_out, cfg.ejmatrix_npy_out)
        vprint(cfg, f"Saving npy outputs to {cfg.out_path}")
        save_outputs(grd_outputs, npy_fnames)
    return strat_dfs


def save_outputs(outputs, fnames):
    """
    Save outputs to files in fnames in directory outpath.
    """
    for out, fout in zip(outputs, fnames):
        outpath = os.path.dirname(fout)
        if not os.path.exists(outpath):
            print(f"Creating new directory: {outpath}")
            os.makedirs(outpath)
        if isinstance(out, pd.DataFrame):
            out.to_csv(fout, index=False)
        elif isinstance(out, np.ndarray):
            np.save(fout, out)
        elif isinstance(out, config.Cfg):
            out.to_py(fout)


# Geospatial helpers
def get_grid_arrays(cfg=CFG):
    """
    Return sparse meshgrid (grdy, grdx) from sizes [m], steps [m] and dtype.

    Grid is centered on South pole with y in (ysize, -ysize) and c in
    (-xsize, xsize) therefore total gridsize is (2*xsize)*(2*ysize).
    """
    ysize, ystep = cfg.grdysize, cfg.grdstep
    xsize, xstep = cfg.grdxsize, cfg.grdstep
    grdy, grdx = np.meshgrid(
        np.arange(ysize, -ysize, -ystep, dtype=cfg.dtype),
        np.arange(-xsize, xsize, xstep, dtype=cfg.dtype),
        sparse=True,
        indexing="ij",
    )
    return grdy, grdx


def latlon2xy(lat, lon, rp=1737.4e3):
    """
    Return (x, y) [m] South Polar stereo coords from (lat, lon) [deg].

    Parameters
    ----------
    lat (num or arr): Latitude(s) [deg]
    lon (num or arr): Longitude(s) [deg]
    rp (num): Radius of the planet or moon [m]

    Returns
    -------
    x (num or arr): South Pole stereo x coordinate(s) [m]
    y (num or arr): South Pole stereo y coordinate(s) [m]
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = rp * np.cos(lat) * np.sin(lon)
    y = rp * np.cos(lat) * np.cos(lon)
    return x, y


def xy2latlon(x, y, rp=1737.4e3):
    """
    Return (lat, lon) [deg] from South Polar stereo coords (x, y) [m].

    Parameters
    ----------
    x (num or arr): South Pole stereo x coordinate(s) [m]
    y (num or arr): South Pole stereo y coordinate(s) [m]
    rp (num): Radius of the planet or moon [m]

    Returns
    -------
    lat (num or arr): Latitude(s) [deg]
    lon (num or arr): Longitude(s) [deg]
    """
    z = np.sqrt(rp**2 - x**2 - y**2)
    lat = np.rad2deg(-np.arcsin(z / rp))
    lon = np.rad2deg(np.arctan2(x, y))
    return lat, lon


def gc_dist(lon1, lat1, lon2, lat2, rp=1737.4e3):
    """
    Return great circle distance [m] from (lon1, lat1) - (lon2, lat2) [deg].

    Uses the Haversine formula adapted from C. Veness
    https://www.movable-type.co.uk/scripts/latlong.html

    Parameters
    ----------
    lon1 (num or arr): Longitude [deg] of start point
    lat1 (num or arr): Latitude [deg] of start point
    lon2 (num or arr): Longitude [deg] of end point
    lat2 (num or arr): Latitude [deg] of end point
    rp (num): Radius of the planet or moon [m]

    Returns
    -------
    gc_dist (num or arr): Great circle distance(s) in meters [m]
    """
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    sin2_dlon = np.sin((lon2 - lon1) / 2) ** 2
    sin2_dlat = np.sin((lat2 - lat1) / 2) ** 2
    a = sin2_dlat + np.cos(lat1) * np.cos(lat2) * sin2_dlon
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = rp * c
    return dist


# General helpers
def probabilistic_round(x, rng=None):
    """
    Randomly round float x up or down weighted by its distance to x + 1.

    E.g. 6.1 will round down ~90% of the time and round up ~10% of the time
    such that over many trials the expected value is 6.1.

    Modified from C. Locke https://stackoverflow.com/a/40921597/8742181

    Parameters
    ----------
    x (float): Any float (works on positive and negative values).

    Returns
    -------
    x_rounded (int): Either floor(x) or ceil(x), rounded probabalistically
    """
    rng = _rng(rng)
    x = np.atleast_1d(x)
    random_offset = rng.random(x.shape)
    x_rounded = np.floor(x + random_offset).astype(int)
    return x_rounded


def round_to_ts(values, timestep):
    """Return values rounded to nearest timestep."""
    return np.around(values / timestep) * timestep


def rtol(num, tol=CFG.rtol):
    """Return relative tolerance where value is equal to num at tol."""
    return np.abs(num) * tol


def minmax_rtol(arr, tol=CFG.rtol):
    """Return min and max of arr -/+ relative tolerance tol."""
    amin = np.min(arr)
    amax = np.max(arr)
    return amin - rtol(amin, tol), amax + rtol(amax, tol)


def diam2vol(diameter):
    """Return volume of sphere [m^3] given diameter [m]."""
    return (4 / 3) * np.pi * (diameter / 2) ** 3


def m2km(x):
    """Convert meters to kilometers."""
    return x / 1000


def km2m(x):
    """Convert kilometers to meters."""
    return x * 1000


def vprint(cfg, *args, **kwargs):
    """Print if verbose."""
    if cfg.verbose:
        print(*args, **kwargs)


def clear_cache():
    """Reset lru_cache."""
    # All objects collected
    for obj in gc.get_objects():
        if isinstance(obj, _lru_cache_wrapper):
            obj.cache_clear()


# Classes
class rv_mixture(stats.rv_continuous):
    """Return mixture of scipy.stats.rv_continuous models."""

    def __init__(self, submodels, *args, weights=None, **kwargs):
        """
        Return mixture of scipy.stats rv submodels weighted by weights.

        Submodels must be instance of scipy.stats.rv_continuous. May not work
        for all stats.rv_continuous submodels. Tested on norm and truncnorm.

        Parameters
        ----------
        submodels (list of scipy.stats.rv_continuous): List of submodels
        weights (list of float): List of weights for submodels
        """
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise ValueError(
                f"Got {len(weights)} weights. Expected {len(submodels)}."
            )
        self.weights = [w / sum(weights) for w in weights]

    def _weighted_sum(self, vals):
        """Return weighted sum of vals (must be same len as self.weights)."""
        return np.sum(vals * np.array(self.weights, ndmin=2).T, axis=0)

    def _pdf(self, x, *args):
        pdfs = [submodel.pdf(x, *args) for submodel in self.submodels]
        return self._weighted_sum(np.array(pdfs))

    def _sf(self, x, *args):
        sfs = [submodel.sf(x, *args) for submodel in self.submodels]
        return self._weighted_sum(np.array(sfs))

    def _cdf(self, x, *args):
        cdfs = [submodel.cdf(x, *args) for submodel in self.submodels]
        return self._weighted_sum(np.array(cdfs))

    def _rvs(self, *args, size=None, random_state=None):
        size = 1 if size is None else size
        rng = np.random.default_rng() if random_state is None else random_state
        rv_choices = rng.choice(len(self.submodels), size=size, p=self.weights)
        rv_samples = [
            rv.rvs(*args, size=size, random_state=rng) for rv in self.submodels
        ]
        rvs = np.choose(rv_choices, rv_samples)
        return rvs
