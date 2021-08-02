"""
Moon Polar Ice and Ejecta Stratigraphy module
Date: 07/06/21
Authors: CJ Tai Udovicic, K Frizzell, K Luchsinger, A Madera, T Paladino
Acknowledgements: This model is largely updated from Cannon et al. (2020)
"""
import argparse
import os
from functools import lru_cache
import numpy as np
import pandas as pd

try:
    import default_config
except ModuleNotFoundError:
    from moonpies import default_config

CACHE = {}


def main(cfg=default_config.Cfg()):
    """Run mixing model with options in cfg (see default_config.py)."""
    # Setup phase
    vprint = print if cfg.verbose else lambda *a, **k: None
    vprint("Initializing run...")

    rng = get_rng(cfg.seed)
    time_arr = get_time_array(cfg)
    df = get_crater_list(cfg, cfg.ejecta_basins, rng)

    # Init strat columns dict based for all cfg.coldtrap_names
    strat_cols = init_strat_columns(time_arr, df, cfg)

    # Main loop over time
    vprint("Starting main loop...")
    for t in range(len(time_arr)):
        strat_cols = update_ice_cols(strat_cols, time_arr, t, cfg, rng)

    # Format and save outputs
    outputs = format_save_outputs(strat_cols, time_arr, df, cfg, vprint)
    return outputs


# Import data
def get_crater_list(cfg, basins=False, rng=None):
    """
    Return dataframe of craters considered in the model.
    """
    global CACHE
    if 'crater_list' not in CACHE:
        df_craters = read_crater_list(cfg)
        df_craters['isbasin'] = False
        
        df_basins = read_basin_list(cfg)
        df_basins['isbasin'] = True

        # Combine DataFrames and randomize ages
        df = pd.concat([df_craters, df_basins])
        df = randomize_crater_ages(df, cfg.timestep, cfg.dtype, rng)
        CACHE['crater_list'] = df
    df = CACHE['crater_list']
    if not basins:
        df = df[df.isbasin == False].reset_index(drop=True)
    return df


def read_crater_list(cfg):
    """
    Return dataframe of craters from crater_csv path with columns names.

    Mandatory columns and naming convention:
        - 'lat': Latitude [deg]
        - 'lon': Longitude [deg]
        - 'diam': Diameter [km]
        - 'age': Crater age [Gyr]
        - 'age_low': Age error residual, lower (e.g., age - age_low) [Gyr]
        - 'age_upp': Age error residual, upper (e.g., age + age_upp) [Gyr]

    Optional columns and naming conventions:
        - 'psr_area': Permanently shadowed area of crater [km^2]

    Parameters
    ----------
    crater_csv (str): Path to crater list csv
    columns (list of str): List of names of all columns in crater_csv

    Returns
    -------
    df (DataFrame): Crater DataFrame read and updated from crater_csv
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
        df["psr_area"] = 0.9 * np.pi * df.rad ** 2
    return df


def read_basin_list(cfg):
    """
    Return dataframe of craters from basin_csv path with columns names.

    Mandatory columns and naming convention:
        - 'lat': Latitude [deg]
        - 'lon': Longitude [deg]
        - 'diam': Diameter [km]
        - 'age': Crater age [Gyr]
        - 'age_low': Age error residual, lower (e.g., age - age_low) [Gyr]
        - 'age_upp': Age error residual, upper (e.g., age + age_upp) [Gyr]

    Parameters
    ----------
    basin_csv (str): Path to crater list csv
    columns (list of str): List of names of all columns in basin_csv

    Returns
    -------
    df (DataFrame): Crater DataFrame read and updated from basin_csv
    """
    df = pd.read_csv(cfg.basin_csv_in, names=cfg.basin_cols, header=0)

    # Convert units, mandatory columns
    df["diam"] = df["diam"] * 1000  # [km -> m]
    df["rad"] = df["diam"] / 2
    df["age"] = df["age"] * 1e9  # [Gyr -> yr]
    df["age_low"] = df["age_low"] * 1e9  # [Gyr -> yr]
    df["age_upp"] = df["age_upp"] * 1e9  # [Gyr -> yr]
    return df


def read_volcanic_species(nk_csv, nk_cols, species):
    """
    Return DataFrame of time, species mass (Table S3, Needham & Kring 2017).
    """
    df = pd.read_csv(nk_csv, names=nk_cols, header=4)
    df = df[["time", species]]
    df["time"] = df["time"] * 1e9  # [Gyr -> yr]
    df[species] = df[species] * 1e-3  # [g -> kg]
    df = df.sort_values("time", ascending=False).reset_index(drop=True)
    return df


def read_teqs(teq_csv):
    """
    Read equilibrium temperatures for ballistic sed module.

    See thermal_eq.py.
    """
    df = pd.read_csv(teq_csv, header=0, index_col=0).T
    return df


@lru_cache(1)
def read_lambda_table(costello_csv):
    """
    Return DataFrame of lambda, probabilities (Table 1, Costello et al. 2018).
    """
    df = pd.read_csv(costello_csv, header=1)
    return df


def read_solar_luminosity(bahcall_csv):
    """
    Return DataFrame of time, solar luminosity (Table 2, Bahcall et al. 2001).
    """
    df = pd.read_csv(bahcall_csv, names=("age", "luminosity"), header=1)
    df.loc[:, "age"] = (4.57 - df.loc[:, "age"]) * 1e9  # [Gyr -> yr]
    return df


# Format and export model results
def make_strat_col(time, ice, ejecta, ejecta_sources, thresh=1e-6):
    """Return stratigraphy column of ice and ejecta"""
    # Label rows by ice (no ejecta) or ejecta_source
    label = np.empty(len(time), dtype=object)
    label[ice > thresh] = "Ice"
    label[ejecta > thresh] = ejecta_sources[ejecta > thresh]

    # Make stratigraphy DataFrame
    data = [time, ice, ejecta]
    cols = ["time", "ice", "ejecta"]
    sdf = pd.DataFrame(np.array(data).T, columns=cols, dtype=time.dtype)

    # Remove empty rows from strat col
    sdf["label"] = label
    sdf = sdf.dropna(subset=["label"])

    # Combine adjacent rows with same label into layers
    isadj = (sdf.label != sdf.label.shift()).cumsum()
    agg = {k: "sum" for k in ("ice", "ejecta")}
    agg = {
        "label": "last",
        "time": "last",
        **agg,
    }
    strat = sdf.groupby(["label", isadj], as_index=False, sort=False).agg(agg)

    # Add depth and ice vs ejecta % of each layer
    # Add depth (reverse cumulative sum)
    strat["depth"] = np.cumsum(strat.ice[::-1] + strat.ejecta[::-1])[::-1]

    # Add ice / ejecta percentage of each layer
    strat["icepct"] = np.round(100 * strat.ice / (strat.ice + strat.ejecta), 4)
    return strat


def format_csv_outputs(strat_cols, time_arr):
    """
    Return all formatted model outputs and write to outpath, if specified.
    """
    ej_dict = {"time": time_arr}
    ice_dict = {"time": time_arr}
    strat_dfs = {}
    ej_source_all = []
    for cname, (ice_t, ej_t, ej_sources) in strat_cols.items():
        ej_dict[cname] = ej_t
        ice_dict[cname] = ice_t
        strat_dfs[cname] = make_strat_col(time_arr, ice_t, ej_t, ej_sources)
        ej_source_all.append(ej_sources)

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


def format_save_outputs(strat_cols, time_arr, df, cfg, vprint=print):
    """
    Format dataframes and save outputs based on write / write_npy in cfg.
    """
    vprint("Formatting outputs")
    ej_df, ice_df, strat_dfs = format_csv_outputs(strat_cols, time_arr)
    if cfg.write:
        # Save config file
        vprint(f"Saving outputs to {cfg.outpath}")
        save_outputs([cfg], [cfg.config_py_out])

        # Save coldtrap strat column dataframes
        fnames = []
        dfs = []
        for coldtrap, strat in strat_dfs.items():
            fnames.append(f"{os.path.join(cfg.outpath, coldtrap)}_strat.csv")
            dfs.append(strat)
        save_outputs(dfs, fnames)

        # Save raw ice and ejecta column vs time dataframes
        save_outputs([ej_df, ice_df], [cfg.ej_t_csv_out, cfg.ice_t_csv_out])
        print(f"Outputs saved to {cfg.outpath}")

    if cfg.write_npy:
        # Note: Only compute these on demand (expensive to do every run)
        # Age grid is age of most recent impact (2D array: NX, NY)
        vprint("Computing gridded outputs...")
        grdy, grdx = get_grid_arrays(cfg)
        grd_outputs = get_grid_outputs(df, grdx, grdy, cfg)
        npy_fnames = (cfg.agegrd_npy_out, cfg.ejmatrix_npy_out)
        vprint(f"Saving npy outputs to {cfg.outpath}")
        save_outputs(grd_outputs, npy_fnames)
    return ej_df, ice_df, strat_dfs


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
        elif isinstance(out, default_config.Cfg):
            out.to_py(fout)


# Pre-compute grid functions
def get_coldtrap_dists(df, cfg):
    """
    Return 2D array of great circle dist between all craters in df. Distance
    from a crater to itself or outside ejecta_threshold set to nan.

    Mandatory
        - df : Read in crater_list.csv as a DataFrame with defined columns
        - df : Required columns defined 'lat' and 'lon'
        - See 'read_crater_list' function

    Parameters
    ----------
    df (DataFrame): Crater DataFrame, e.g., read by read_crater_list
    coldtraps (arr of str): Coldtrap crater names (must be in df.cname)
    ej_threshold (num): Number of crater radii to limit distant to else np.nan

    Returns
    -------
    dist (2D array): great circle distances from df craters to coldtraps
    """
    ej_threshold = cfg.ej_threshold
    if ej_threshold < 0:
        ej_threshold = np.inf

    dist = np.zeros((len(df), len(cfg.coldtrap_names)), dtype=cfg.dtype)
    for i, row in df.iterrows():
        src_lon = row.lon
        src_lat = row.lat
        for j, cname in enumerate(cfg.coldtrap_names):
            if row.cname == cname:
                dist[i, j] = np.nan
                continue
            dst_lon = df[df.cname == cname].psr_lon.values
            dst_lat = df[df.cname == cname].psr_lat.values
            d = gc_dist(src_lon, src_lat, dst_lon, dst_lat)
            dist[i, j] = d
            # Cut off distances > ej_threshold crater radii
            if dist[i, j] > (ej_threshold * df.iloc[i].rad):
                dist[i, j] = np.nan
    dist[dist <= 0] = np.nan
    return dist


def get_ejecta_thickness(distance, radius, ds2c=18e3, order=-3):
    """
    Return ejecta thickness as a function of distance given crater radius.

    Complex craters McGetchin 1973
    """
    exp_complex = 0.74  # McGetchin 1973, simple craters exp=1
    exp = np.ones_like(radius)
    exp[radius * 2 > ds2c] = exp_complex
    thickness = 0.14 * radius ** exp * (distance / radius) ** order
    thickness[np.isnan(thickness)] = 0
    return thickness


def get_ejecta_thickness_matrix(time_arr, df, cfg):
    """
    Return ejecta_matrix of thickness [m] at each time in time_arr given
    triangular matrix of distances between craters in df.

    Return
    ------
    ejecta_thick_time (3D array): Ejecta thicknesses (shape: NY, NX, NT)
    """
    # Distance from all craters to all coldtraps shape: (Ncrater, Ncoldtrap)
    ej_distances = get_coldtrap_dists(df, cfg)

    # Ejecta thickness [m] deposited in each coldtrap from each crater
    rad = df.rad.values[:, np.newaxis]  # Pass radii as column vector
    ej_thick = get_ejecta_thickness(
        ej_distances,
        rad,
        cfg.simple2complex,
        cfg.ejecta_thickness_order,
    )
    ej_ages = df.age.values
    ej_thick_t = ages2time(time_arr, ej_ages, ej_thick, np.nansum, 0)

    # Label sources above threshold
    has_ej = ej_thick > cfg.thickness_threshold
    ej_sources = (df.cname.values[:, np.newaxis] + ",") * has_ej
    ej_sources_t = ages2time(time_arr, ej_ages, ej_sources, np.sum, "", object)
    ej_sources_t = np.char.rstrip(ej_sources_t.astype(str), ",")
    return ej_thick_t, ej_sources_t


def ages2time(time_arr, age_arr, values, agg=np.nansum, fillval=0, dtype=None):
    """
    Return values filled into array of len(time_arr) filled to nearest ages.
    """
    if dtype is None:
        dtype = values.dtype
    # Remove ages and associated values not in time_arr
    isin = np.where((age_arr >= time_arr.min()) & (age_arr <= time_arr.max()))
    age_arr = age_arr[isin]
    values = values[isin]

    # Minimize distance between ages and times to handle rounding error
    time_idx = np.abs(time_arr[:, np.newaxis] - age_arr).argmin(axis=0)
    shape = [len(time_arr), *values.shape[1:]]
    values_time = np.full(shape, fillval, dtype=dtype)

    for i, t_idx in enumerate(time_idx):
        # Sum here in case more than one crater formed at t_idx
        values_time[t_idx] = agg([values_time[t_idx], values[i]], axis=0)
    return values_time


def get_grid_outputs(df, grdx, grdy, cfg):
    """
    Return matrices of interest computed on the grid of shape: (NY, NX, (NC)).
    """
    # Age of most recent impact (2D array: NX, NY)
    age_grid = get_age_grid(df, grdx, grdy, cfg)

    # Great circle distance from each crater to grid (3D array: NX, NY, NC)
    dist_grid = get_gc_dist_grid(df, grdx, grdy, cfg.dtype)

    # Ejecta thickness produced by each crater on grid (3D array: NX, NY, NC)
    ej_thick_grid = get_ejecta_thickness(
        dist_grid,
        df.rad.values[:, np.newaxis],
        cfg.simple2complex,
        cfg.ejecta_thickness_order,
    )
    # TODO: ballistic sed depth, kinetic energy, etc
    return age_grid, ej_thick_grid


def get_gc_dist_grid(df, grdx, grdy, cfg):
    """
    Return 3D array of great circle dist between all craters in df and every
    point on the grid.

    Parameters
    ----------
    df (DataFrame):
    grdx (arr):
    grdy (arr):

    Return
    ------
    grd_dist (3D arr: NX, NY, Ndf):
    """
    ny, nx = grdy.shape[0], grdx.shape[1]
    lat, lon = xy2latlon(grdx, grdy, cfg.rad_moon)
    grd_dist = np.zeros((nx, ny, len(df)), dtype=cfg.dtype)
    for i in range(len(df)):
        clon, clat = df.iloc[i][["lon", "lat"]]
        grd_dist[:, :, i] = gc_dist(clon, clat, lon, lat)
    return grd_dist


# Polar ice deposition
def get_polar_ice(time_arr, t, cfg, rng=None):
    """
    Return total polar ice at timestep t from all sources.

    Parameters
    ----------
    time_arr (array): Model time array [yrs].
    t (int): Index of current timestep in the model.
    cfg (Cfg): Config object, must contain:
        ...

    Return
    ------
    polar_ice (float): Ice thickness [m] delivered to the pole vs time.
    """
    global CACHE
    if "polar_ice_time" not in CACHE:
        CACHE["polar_ice_time"] = np.sum(
            [
                get_solar_wind_ice(time_arr, cfg),
                get_volcanic_ice(time_arr, cfg),
                get_impact_ice(time_arr, cfg, rng),
            ],
            axis=0,
        )
    return CACHE["polar_ice_time"][t]


# Solar wind module
def get_solar_wind_ice(time_arr, cfg):
    """
    Return solar wind ice over time if mode is mpies.
    """
    sw_ice_t = np.zeros_like(time_arr)
    if cfg.solar_wind_ice:
        sw_ice_mass = solar_wind_ice(time_arr, cfg)
        sw_ice_t = get_ice_thickness(sw_ice_mass, cfg)
    return sw_ice_t


def solar_wind_ice(time_arr, cfg):
    """
    Return ice mass [kg] deposited globally by solar wind vs time in time_arr.

    Does not account for movement of ice, only deposition given a supply rate.

    cfg.solar_wind_mode determines the H2O supply rate:
      "Benna": H2O supply rate 2 g/s (Benna et al. 2019; Arnold, 1979;
        Housley et al. 1973)
      "Lucey-Hurley": H2 supply rate 30 g/s, converted to water assuming 1 part
        per thousand (Lucey et al. 2020)

    Parameters
    ----------
    time_arr (array): Model time array.
    cfg (Cfg): Config object, must contain:
      - solar_wind_mode (str): Solar wind mode to use.
      - faint_young_sun (bool): Scale by luminosity of faint young sun
      - cfg.bahcall_csv_in (str): Path to solar wind data.

    Return
    ------
    sw_ice_mass (1D array): Ice mass deposited by solar wind at each time
    """
    if cfg.solar_wind_mode == "Benna":
        # Benna et al. 2019; Arnold, 1979; Housley et al. 1973
        volatile_supply_rate = 2 * 1e-3  # [g/s -> kg/s]
    elif cfg.solar_wind_mode == "Lucey-Hurley":
        # Lucey et al. 2020, Hurley et al. 2017 (assume 1 ppt H2 -> H2O)
        volatile_supply_rate = 30 * 1e-3 * 1e-3  # [g/s - kg/s] * [1 ppt]
    else:
        msg = 'Solar wind mode not recognized. Accepts "Benna", "Lucey-Hurley"'
        raise ValueError(msg)
    # convert to kg per timestep
    supply_rate_ts = volatile_supply_rate * 60 * 60 * 24 * 365 * cfg.timestep

    sw_ice_mass = np.ones_like(time_arr) * supply_rate_ts
    if cfg.faint_young_sun:
        # Import historical solar luminosity (Bahcall et al. 2001)
        df_lum = read_solar_luminosity(cfg.bahcall_csv_in)

        # Interpolate to time_arr, scale by solar luminosity at each time
        lum_time = np.interp(-time_arr, -df_lum["age"], df_lum["luminosity"])
        sw_ice_mass *= lum_time
    return sw_ice_mass


# Volcanic ice delivery module
def get_volcanic_ice(time_arr, cfg):
    """
    Return ice thickness [m] delivered to pole by volcanic outgassing vs time.

    Return
    ------
    volc_ice_t (arr): Ice thickness [m] delivered at the pole at each time.
    """
    if cfg.volc_mode == "NK":
        volc_ice_mass = volcanic_ice_nk(time_arr, cfg)
    elif cfg.volc_mode == "Head":
        volc_ice_mass = volcanic_ice_head(time_arr, cfg)
    else:
        raise ValueError(f"Invalid mode {cfg.volc_mode}.")

    volc_ice_t = get_ice_thickness(volc_ice_mass, cfg)
    if not cfg.volc_ballistic:
        volc_ice_t = get_ice_thickness(volc_ice_mass, cfg)
        # rescale by volcanic deposition % instead of ballistic hop %
        volc_ice_t *= cfg.volc_dep_effcy / cfg.ballistic_hop_effcy
    return volc_ice_t


def volcanic_ice_nk(time_arr, cfg):
    """
    Return global ice [kg] deposited vs time using Needham & Kring (2017).

    Values from supplemental spreadsheet S3 (Needham and Kring, 2017)
    transient atmosphere data. Scale by % material delievered to the pole.

    Return
    ------
    volc_ice_mass (arr): Ice mass [kg] deposited at the pole at each time.
    """
    df_volc = read_volcanic_species(cfg.nk_csv_in, cfg.nk_cols, cfg.nk_species)
    df_volc = df_volc[df_volc.time < time_arr.max()]

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


def volcanic_ice_head(time_arr, cfg):
    """
    Return global ice [kg] deposited vs time using Head et al. (2020).

    Return
    ------
    volc_ice_mass (arr): Ice mass [kg] deposited vs time.
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
def get_ballistic_sed_depths(time_arr, t, cfg):
    """
    Return ballistic sedimentation depth for each coldtrap at t.
    """
    global CACHE
    if "bsed_depths" not in CACHE:
        df = CACHE['crater_list']  # Must exist before calling this function
        CACHE["bsed_depths"] = ballistic_sed_depths_time(time_arr, df, cfg)
    return CACHE["bsed_depths"][t]


def ballistic_sed_depths_time(time_arr, df, cfg):
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
    if cfg.ballistic_teq:
        bsed_depths = check_teq_ballistic_sed(bsed_depths, cfg)

    # Convert to time array shape: (Ncrater, Ncoldtrap) -> (Ntime, Ncoldtrap)
    ages = df.age.values
    bsed_depth_t = ages2time(time_arr, ages, bsed_depths, np.nanmax, 0)
    return bsed_depth_t


def ballistic_sed_depth(dist, diam, cfg):
    """
    Return ballistic sedimentation mixing depths for each crater.
    """
    # Get secondary crater diameter excluded
    secondary_diam_vec = np.vectorize(secondary_diam)
    diam_secondary = secondary_diam_vec(diam, dist, cfg)
    if cfg.secondary_depth_mode == "singer":
        depth = ballistic_sed_depth_singer(diam_secondary, cfg)
    elif cfg.secondary_depth_mode == "xie":
        depth = ballistic_sed_depth_xie(diam_secondary, dist, cfg)
    else:
        msg = f"Invalid secondary depth mode {cfg.secondary_depth_mode}."
        raise ValueError(msg)
    return depth


def check_teq_ballistic_sed(ballistic_depths, cfg):
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


# Singer mode (cfg.secondary_crater_mode == 'singer')
def ballistic_sed_depth_singer(diam_secondary, cfg):
    """
    Returns excavation depth of ballistically sedimentation from secondary
    depth (Singer et al. 2020).

    Parameters
    ----------
    diam_secondary (num or array): diameter of secondary crater [m]
    """
    # Convert secondary diameter to depth with depth to diam
    depth_sec_final = diam_secondary * cfg.depth_to_diam_sec

    # Excavation depth is half of crater depth (Melosh 1989)
    return depth_sec_final / 2


def secondary_diam(diam, dist, cfg):
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
def ballistic_sed_depth_xie(diam_secondary, dist, cfg):
    """
    Returns excavation depth of ballistically sedimentation from secondary
    depth (Xie et al. 2020).

    Parameters
    ----------
    diam_secondary (num or array): diameter of primary crater [m]
    dist (num or array): distance away from primary crater center [m]
    theta (num): angle of impact [radians]
    cfg (Config): config object
    """
    # Convert final diameter to transient
    t_rad = final2transient(diam_secondary) / 2
    speed = ballistic_speed(dist, cfg)
    depth = secondary_excavation_depth_xie(t_rad, speed, cfg.xie_depth_eff)
    return depth


def ballistic_speed(dist, cfg):
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


def secondary_excavation_depth_xie(t_rad, speed, cfg):
    """
    Returns the excavation depth of a secondary crater (Xie et al. 2020).

    Parameters
    ----------
    t_rad (num): secondary transient apparent crater radius [m]
    speed (num): incoming impactor velocity [m/s]

    Returns
    -------
    excav_depth (num): Excavation depth of a secondary crater [m]
    """
    depth = cfg.xie_depth_rad * t_rad * (speed ** cfg.xie_vel_exp)
    if cfg.xie_depth_eff:
        depth = secondary_excavation_depth_eff(depth, t_rad, cfg)
    return depth


def secondary_excavation_depth_eff(depth, t_rad, cfg):
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
    if cfg.xie_sec_radial_frac > 1:
        raise ValueError("Config: xie_sec_radial_frac must be in range [0, 1]")
    radial_dist = t_rad * cfg.xie_sec_radial_frac
    return cfg.xie_c_ex * depth * (1 - (radial_dist / t_rad) ** 2)


# Strat column functions
def init_strat_columns(time_arr, df, cfg):
    """
    Return initialized stratigraphy columns (ice, ej, ej_sources)
    """
    ej_cols, ej_sources = get_ejecta_thickness_matrix(time_arr, df, cfg)
    strat_columns = make_strat_columns(ej_cols, ej_sources, cfg)
    return strat_columns


def make_strat_columns(ej_cols, ej_sources, cfg):
    """
    Return dict of ice and ejecta columns for cold trap craters in df.

    Currently init at start of time_arr, but possibly:
    - TODO: init after crater formed? Or destroy some pre-existing ice here?
    - TODO: init only after PSR stable (Siegler 2015)?

    Return
    ------
    strat_columns_dict[cname] = [ice_col, ej_col]
    """
    # Get cold trap crater names and their locations in df
    ctraps = cfg.coldtrap_names

    # Build strat columns with cname: ccol, ice_col, ej_col
    ice_cols = np.zeros_like(ej_cols)  # shape: NT, Ncoldtrap
    strat_columns = {
        coldtrap: [ice_cols[:, i], ej_cols[:, i], ej_sources[:, i]]
        for i, coldtrap in enumerate(ctraps)
    }
    return strat_columns


def update_ice_col(cols, t, new_ice, overturn_depth, bsed_depth, cfg):
    """
    Return ice_column updated with all processes applicable at time t.
    """
    ice_col, ej_col, _ = cols
    # Ballistic sed gardens first, if crater was formed
    ice_col = garden_ice_column(ice_col, ej_col, t - 1, bsed_depth)

    # Ice gained by column
    ice_col[t] = new_ice

    # Ice eroded in column
    ice_col = remove_ice_overturn(ice_col, ej_col, t, overturn_depth, cfg)
    return ice_col


def update_ice_cols(strat_cols, time_arr, t, cfg, rng=None):
    """
    Update ice_cols new ice added and ice eroded.
    """
    # Get ice modification for this timestep
    ballistic_sed_d = get_ballistic_sed_depths(time_arr, t, cfg)
    polar_ice = get_polar_ice(time_arr, t, cfg, rng)
    overturn_d = get_overturn_depth(time_arr, t, cfg)

    # Update all coldtrap strat_cols
    for i, (coldtrap, cols) in enumerate(strat_cols.items()):
        bsed_d = ballistic_sed_d[i]
        new_ice = get_ice_coldtrap(polar_ice, coldtrap, cfg)
        ice_col = update_ice_col(cols, t, new_ice, overturn_d, bsed_d, cfg)
        strat_cols[coldtrap][0] = ice_col
    return strat_cols


def get_ice_thickness(global_ice_mass, cfg):
    """
    Return ice thickness assuming global_ice_mass delived balistically.

    Converts:
        global_ice_mass to polar_ice_mass with cfg.ballistic_hop_effcy,
        polar_ice_mass to volume with cfg.ice_density,
        volume to thickness with coldtrap_area of cfg.ice_species at cfg.pole.
    """
    polar_ice_mass = global_ice_mass * cfg.ballistic_hop_effcy  # [kg]
    ice_volume = polar_ice_mass / cfg.ice_density  # [m^3]
    ice_thickness = ice_volume / cfg.coldtrap_area
    return ice_thickness


# Impact gardening module (remove ice by impact overturn)
def remove_ice_overturn(ice_col, ej_col, t, depth, cfg):
    """
    Return ice_col with ice removed by impact gardening to overturn_depth.
    """
    if cfg.impact_gardening_costello:
        ice_col = garden_ice_column(ice_col, ej_col, t, depth)
    else:
        ice_col = erode_ice_cannon(ice_col, ej_col, t, depth)
    return ice_col


def erode_ice_cannon(ice_col, ej_col, t, erosion_depth=0.1, ej_shield=0.4):
    """
    Return eroded ice column using Cannon et al. (2020) method (10 cm / 10 Ma).
    """
    # BUG in Cannon ds01: erosion base not updated for adjacent ejecta layers
    # Erosion base is most recent time when ejecta column was > ejecta_shield
    erosion_base = -1
    erosion_base_idx = np.where(ej_col[: t + 1] > ej_shield)[0]
    if len(erosion_base_idx) > 0:
        erosion_base = erosion_base_idx[-1]

    # Garden from top of ice column until ice_to_erode amount is removed
    # BUG in Cannon ds01: doesn't account for partial shielding by small ej
    layer = t
    while erosion_depth > 0 and layer >= 0:
        if t > erosion_base:
            ice_in_layer = ice_col[layer]
            if ice_in_layer >= erosion_depth:
                ice_col[layer] -= erosion_depth
            else:
                ice_col[layer] = 0
            erosion_depth -= ice_in_layer
            layer -= 1
        else:
            # BUG in Cannon ds01: t should be layer
            # - loop doesn't end if we reach erosion base while eroding
            # - loop only ends here if we started at erosion_base
            break
    return ice_col


def garden_ice_column(ice_column, ejecta_column, t, depth):
    """
    Return ice_column gardened to overturn_depth, preserved by ejecta_column.

    Ejecta deposited on last timestep preserves ice. Loop through ice_col and
    ejecta_col until overturn_depth and remove all ice that is encountered.

    Parameters
    ----------
    ice_column (arr):
    ejecta_column (arr):
    t (int): Current timestep (first index in ice/ej columns to garden)
    depth (num): Total depth [m] to garden.
    ice_first (bool): Erode ice first (bsed) else ejecta first (gardening)
    """
    # Alternate so i//2 is current index to garden (odd: ejecta, even: ice)
    # If ice_first, skip topmost ejecta layer and erode ice first
    i = (2 * t) + 1
    d = 0  # current depth
    while i >= 0 and d < depth:  # and < 2 * len(ice_column):
        if i % 2:
            # Odd i (ejecta): do nothing, add ejecta layer to depth, d
            d += ejecta_column[i // 2]
        else:
            # Even i (ice): remove all ice from layer, add it to depth, d
            d += ice_column[i // 2]
            ice_column[i // 2] = 0
        i -= 1
    # If odd i (ice) on last iter, we likely removed too much ice
    #   Add back any excess depth we travelled to ice_col
    last_ind = i + 1
    if not (last_ind % 2) and d > depth:
        ice_column[last_ind // 2] = d - depth
    return ice_column


# Impact gardening module (Costello et al. 2018, 2020)
def get_overturn_depth(time_arr, t, cfg):
    """
    Return impact overturn depth [m] at timestep t.

    Parameters
    ----------
    time_arr (array): Model time array [yrs].
    t (int): Index of current timestep in the model.
    cfg (Cfg): Config object, must contain:
        ...

    Return
    ------
    overturn_depth (float): Overturn_depth [m] at t.
    """
    global CACHE
    if "overturn_depth_time" not in CACHE:
        CACHE["overturn_depth_time"] = overturn_depth_time(time_arr, cfg)
    return CACHE["overturn_depth_time"][t]


def overturn_depth_costello(
    u,
    v,
    costello_csv,
    t,
    n=1,
    prob_pct="99%",
    c=0.41,
    h=0.04,
):
    """
    Return regolith overturn depth at time t, given lambda and size-frequency
    u, v given (u*D^v). Uses eqn 10, Costello (2020).

    Parameters
    ----------
    n (num): number of events this timestep
    u (num): sfd scaling factor (u*x^v) [m^-(2+v) yrs^-1]
    v (num): sfd slope/exponent (u*x^v). Must be < -2.
    t (num): time elapsed [yrs] (i.e. timestep)
    c (num): proximity scaling parameter for overlapping craters
    h (num): depth fraction of crater overturn

    Return
    ----------
    overturn_depth (num): impact gardening depth [m]
    """
    lam = overturn_lambda(costello_csv, n, prob_pct)
    b = 1 / (v + 2)  # eq 12, Costello 2020
    p1 = (v + 2) / (v * u)
    p2 = 4 * lam / (np.pi * c ** 2)
    a = abs(h * (p1 * p2) ** b)  # eq 11, Costello 2020
    overturn_depth = a * t ** (-b)  # eq 10, Costello 2020
    return overturn_depth


def overturn_u(
    a,
    b,
    regime,
    vf=1800,
    rho_t=1500,
    rho_i=2780,
    kr=0.6,
    k1=0.132,
    k2=0.26,
    mu=0.41,
    y=1e4,
    g=0.62,
    theta_i=45,
):
    """
    Return size-frequecy factor u for overturn (eqn 13, Costello 2020).

    """
    alpha = k2 * (y / (rho_t * vf ** 2)) ** ((2 + mu) / 2)
    beta = (-3 * mu) / (2 + mu)
    delta = 2 * kr
    gamma = (k1 * np.pi * rho_i) / (6 * rho_t)
    eps = (g / (2 * vf ** 2)) * (rho_i / rho_t) ** (1 / 3)

    if regime == "strength":
        t1 = np.sin(np.deg2rad(theta_i)) ** (2 / 3)
        denom = delta * (gamma * alpha ** beta) ** (1 / 3)
        u = t1 * a * (1 / denom) ** b

    elif regime == "gravity":
        t1 = np.sin(np.deg2rad(theta_i)) ** (1 / 3)
        denom = delta * (gamma * eps ** beta) ** (1 / 3)
        exp = (3 * b) / (3 + beta)
        u = t1 * a * (1 / denom) ** exp
    else:
        raise ValueError('Regime must be "strength" or "gravity"')
    return u


def overturn_depth_time(time_arr, cfg):
    """
    Return array of overturn depth [m] as a function of time.

    Required fields in cfg:
    TODO: describe these
    - cfg.n_overturn:
    cfg.overturn_prob_pct
    cfg.overturn_regimes
    cfg.overturn_ab
    cfg.timestep
    cfg.impact_speeds
    cfg.mode
    cfg.dtype
    """
    if not cfg.impact_gardening_costello:
        # Cannon assumes 0.1 m / Ma gardening at all times
        overturn_t = 0.1 * np.ones_like(time_arr)
    else:
        overturn_depths = []
        for r in cfg.overturn_regimes:
            a, b = cfg.overturn_ab[r]
            vf = cfg.impact_speeds[r]
            a_scaled = a * impact_flux(time_arr) / impact_flux(0)
            u = overturn_u(
                a_scaled,
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
                cfg.impact_angle,
            )
            overturn = overturn_depth_costello(
                u,
                b,
                cfg.costello_csv_in,
                cfg.timestep,
                cfg.n_overturn,
                cfg.overturn_prob_pct,
            )
            overturn_depths.append(overturn)
        overturn_t = np.sum(overturn_depths, axis=0)
    return overturn_t


def overturn_lambda(costello_csv, n=1, prob_pct="99%"):
    """
    Return lambda given prob_pct and n events (Table 1, Costello et al. 2018).

    Parameters
    ----------
    n (num): cumulative number of overturn events
    prob_pct ('10%', '50%', or '90%): percent probability threshold

    Return
    ------
    lam (num): avg number of events per area per time
    """
    df = read_lambda_table(costello_csv)

    # Interpolate nearest value in df[prob_pct] from input n
    lam = np.interp(n, df.n, df[prob_pct])
    return lam


def overturn_depth_costello_str(time):
    """Return Costello 2020 eqn 10, strength (1 overturn, 99% prob)."""
    return 3.94e-5 * time ** 0.5


def overturn_depth_costello_grav(time):
    """Return Costello 2020 eqn 10, gravity (1 overturn, 99% prob)"""
    return 7.35e-4 * time ** 0.35


def overturn_depth_morris(time):
    """Return overturn depth from Morris (1978) fits to Apollo samples."""
    return 4.39e-5 * time ** 0.45


def overturn_depth_speyerer(time):
    """Return overturn depth from Speyerer et al. (2016) LROC splotches."""
    return 3.45e-5 * time ** 0.47


# Impact-delivered ice module
def get_impact_ice(time_arr, cfg, rng=None):
    """
    Return ice thickness [m] delivered to pole due to global impacts vs time.

    Return
    ------
    impact_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    impact_ice_t = np.zeros_like(time_arr)
    impact_ice_t += get_micrometeorite_ice(time_arr, cfg)
    impact_ice_t += get_small_impactor_ice(time_arr, cfg)
    impact_ice_t += get_small_simple_crater_ice(time_arr, cfg)
    impact_ice_t += get_large_simple_crater_ice(time_arr, cfg, rng)
    impact_ice_t += get_complex_crater_ice(time_arr, cfg, rng)
    if cfg.impact_ice_basins:
        impact_ice_t += get_basin_ice(time_arr, cfg)
    return impact_ice_t


def get_micrometeorite_ice(time_arr, cfg):
    """
    Return ice thickness [m] delivered to pole due to micrometeorites vs time.

    Return
    ------
    mm_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    mm_ice_mass = ice_micrometeorites(time_arr, cfg)
    mm_ice_t = get_ice_thickness(mm_ice_mass, cfg)
    return mm_ice_t


def get_small_impactor_ice(time_arr, cfg):
    """
    Return ice thickness [m] delivered to pole due to small impactors vs time.

    Return
    ------
    si_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    impactor_diams, impactors = get_small_impactor_pop(time_arr, cfg)
    si_ice_mass = ice_small_impactors(impactor_diams, impactors, cfg)
    si_ice_t = get_ice_thickness(si_ice_mass, cfg)
    return si_ice_t


def get_small_simple_crater_ice(time_arr, cfg):
    """
    Return ice thickness [m] delivered to pole by small simple crater impacts.

    Return
    ------
    ssc_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    crater_diams, n_craters_t, sfd_prob = get_crater_pop(time_arr, "c", cfg)
    n_craters = n_craters_t * sfd_prob
    ssc_ice_mass = ice_small_craters(crater_diams, n_craters, "c", cfg)
    ssc_ice_t = get_ice_thickness(ssc_ice_mass, cfg)
    return ssc_ice_t


def get_large_simple_crater_ice(time_arr, cfg, rng=None):
    """
    Return ice thickness [m] delivered to pole by large simple crater impacts.

    Return
    ------
    lsc_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    lsc_ice_t = get_ice_stochastic(time_arr, "d", cfg, rng)
    return lsc_ice_t


def get_complex_crater_ice(time_arr, cfg, rng=None):
    """
    Return ice thickness [m] delivered to pole by complex crater impacts.

    Return
    ------
    cc_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    cc_ice_t = get_ice_stochastic(time_arr, "e", cfg, rng)
    return cc_ice_t


def get_basin_ice(time_arr, cfg):
    """
    Return ice thickness [m] delivered to pole by basin impacts vs time.

    Return
    ------
    b_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    df_basins = get_crater_list(cfg, basins=True)
    df_basins = df_basins[df_basins.isbasin].reset_index(drop=True)
    b_ice_mass = ice_basins(df_basins, time_arr, cfg)
    b_ice_t = get_ice_thickness(b_ice_mass, cfg)
    return b_ice_t


def get_ice_stochastic(time_arr, regime, cfg, rng=None):
    """
    Return ice thickness [m] delivered to pole at each time due to
    a given stochastic regime.

    Parameters
    ----------
    time_arr (arr): Array of times [yr] to calculate ice thickness at.
    regime (str): Regime of ice production.
    cfg (instance): Configuration object.
    rng (instance): Random number generator.

    Return
    ------
    ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    diams, num_craters_t, sfd_prob = get_crater_pop(time_arr, regime, cfg)

    # Round to integer number of craters at each time
    num_craters_t = probabilistic_round(num_craters_t, rng=rng)

    # Get ice thickness for each time, randomizing crater diam and impact speed
    ice_t = np.zeros_like(time_arr)
    for i, num_craters in enumerate(num_craters_t):
        rand_diams = np.random.choice(diams, num_craters, p=sfd_prob)
        hyd_diams = get_random_hydrated_craters(rand_diams, cfg, rng)
        speeds = get_random_impactor_speeds(len(hyd_diams), cfg, rng)
        ice_mass = ice_large_craters(hyd_diams, speeds, regime, cfg)
        ice_t[i] = get_ice_thickness(ice_mass, cfg)
    return ice_t


@lru_cache(1)
def read_ballistic_hop_csv(bhop_csv):
    """
    Return dict of ballistic hop efficiency of each coldtrap in bhop_csv.
    """
    ballistic_hop_coldtraps = {}
    with open(bhop_csv, "r") as f:
        for line in f.readlines():
            coldtrap, bhop = line.strip().split(",")
            # TODO: our bhop efficiency is per km^2 so we * 1e6 / 100 to make it % like cannon (should change bhop_csv)
            ballistic_hop_coldtraps[coldtrap] = float(bhop) * 1e6 / 100
    return ballistic_hop_coldtraps


def get_ice_coldtrap(ice_polar, coldtrap, cfg):
    """
    Return ice in a particular coldtrap scaling by ballistic hop efficiency.

    See read_ballistic_hop_csv.
    """
    coldtrap_ice = ice_polar
    if cfg.ballistic_hop_moores:
        bhop_effcy_coldtrap = read_ballistic_hop_csv(cfg.bhop_csv_in)[coldtrap]
        # Rescale by coldtrap ballistic hop % instead of polar ballistic hop
        coldtrap_ice *= bhop_effcy_coldtrap / cfg.ballistic_hop_effcy
    return coldtrap_ice


def ice_moores(ice_thickness, bhop_effcy_pole, bhop_effcy_coldtrap):
    """
    Return ice_thickness scaled by ballistic hop efficiency of coldtrap.
    """
    bhop_eff_per_crater = bhop_effcy_pole / bhop_effcy_coldtrap
    return ice_thickness * bhop_eff_per_crater


def ice_micrometeorites(time, cfg):
    """
    Return ice from micrometeorites (Regime A, Cannon 2020).

    Multiply total_mm_mass / yr by timestep and scale by assumed hydration %
    and scale by ancient flux relative to today.

    Unlike larger impactors, we:
    - Do NOT assume ctype composition and fraction of hydrated ctypes
        - maybe bigger fraction of comet grain causese MMs to be more ice rich?
    - Do NOT scale by asteroid retention rate (Ong et al., 2011)
        - All micrometeorite material likely melt/vaporized & retained?
    """
    # Scale by ancient impact flux relative to today, assume some wt% hydration
    mm_mass_t = cfg.mm_mass_rate * impact_flux(time) / impact_flux(0)
    ice_dt = cfg.hydrated_wt_pct * cfg.impactor_mass_retained * cfg.timestep
    # TODO: Why does Cannon not do 36% CC and 2/3 of CC hydrated (reg B, C)
    # TODO: improve micrometeorite flux?
    micrometeorite_ice_t = mm_mass_t * ice_dt
    return micrometeorite_ice_t


def ice_small_impactors(diams, impactors_t, cfg):
    """
    Return ice mass [kg] from small impactors (Regime B, Cannon 2020) given
    impactor diams and number of impactors over time.
    """
    impactor_masses = diam2vol(diams) * cfg.impactor_density
    impactor_mass_t = np.sum(impactors_t * impactor_masses, axis=1)
    impactor_ice_t = impactor_mass2water(
        impactor_mass_t,
        cfg.ctype_frac,
        cfg.ctype_hydrated,
        cfg.hydrated_wt_pct,
        cfg.impactor_mass_retained,
    )
    return impactor_ice_t


def ice_small_craters(
    crater_diams,
    ncraters,
    regime,
    cfg,
):
    """
    Return ice from simple craters, steep branch (Regime C, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams, cfg.impact_speed_mean, regime, cfg)
    impactor_masses = diam2vol(impactor_diams) * cfg.impactor_density  # [kg]
    total_impactor_mass = np.sum(impactor_masses * ncraters, axis=1)
    total_impactor_water = impactor_mass2water(
        total_impactor_mass,
        cfg.ctype_frac,
        cfg.ctype_hydrated,
        cfg.hydrated_wt_pct,
        cfg.impactor_mass_retained,
    )
    return total_impactor_water


def ice_large_craters(crater_diams, impactor_speeds, regime, cfg):
    """
    Return ice from simple/complex craters, shallow branch of sfd
    (Regime D-E, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams, impactor_speeds, regime, cfg)
    impactor_masses = diam2vol(impactor_diams) * cfg.impactor_density  # [kg]

    # Find ice mass assuming hydration wt% and retention based on speed
    ice_retention = ice_retention_factor(impactor_speeds)
    ice_masses = impactor_masses * cfg.hydrated_wt_pct * ice_retention
    return np.sum(ice_masses)


def ice_basins(df_basins, time_arr, cfg):
    """Return ice mass from basin impacts vs time."""
    crater_diams = 2 * df_basins.rad.values
    impactor_speeds = (
        np.ones(len(df_basins), dtype=cfg.dtype) * cfg.impact_speed_mean
    )  # TODO: speeds of basins?
    impactor_masses = np.zeros_like(crater_diams)
    for i, (diam, speed) in enumerate(zip(crater_diams, impactor_speeds)):
        impactor_diam = diam2len(diam, speed, "f", cfg)
        impactor_masses[i] = diam2vol(impactor_diam) * cfg.impactor_density

    # Find ice mass assuming hydration wt% and retention based on speed
    ice_retention = ice_retention_factor(impactor_speeds)
    ice_masses = impactor_masses * cfg.hydrated_wt_pct * ice_retention

    # Insert each basin ice mass to its position in time_arr
    ages = df_basins.age.values
    basin_ice_t = ages2time(time_arr, ages, ice_masses, np.sum, 0)
    return basin_ice_t


def ice_retention_factor(speeds):
    """
    Return ice retained in impact, given impactor speeds (Cannon 2020).

    For speeds < 10 km/s, retain 50% (Svetsov & Shuvalov 2015 via Cannon 2020).
    For speeds >= 10 km/s, use eqn ? (Ong et al. 2010 via Cannon 2020)
    """
    # TODO: find/verify retention(speed) eqn in Ong et al. 2010?
    # BUG? retention distribution is discontinuous
    speeds = speeds * 1e-3  # [m/s] -> [km/s]
    retained = np.ones_like(speeds) * 0.5  # nominal 50%
    retained[speeds >= 10] = 36.26 * np.exp(-0.3464 * speeds[speeds >= 10])
    retained[retained < 0] = 0
    return retained


def impactor_mass2water(
    impactor_mass,
    ctype_frac=0.36,
    ctype_hyd=2 / 3,
    hyd_wt_pct=0.1,
    mass_retained=0.165,
):
    """
    Return water [kg] from impactor mass [kg] using assumptions of Cannon 2020:
        - 36% of impactors are C-type (Jedicke et al., 2018)
        - 2/3 of C-types are hydrated (Rivkin, 2012)
        - Hydrated impactors are 10% water by mass (Cannon et al., 2020)
        - 16% of asteroid mass retained on impact (Ong et al. 2011)
    """
    return ctype_frac * ctype_hyd * hyd_wt_pct * impactor_mass * mass_retained


# Stochasticity and randomness
def get_rng(rng):
    """Return numpy random number generator from given seed in rng."""
    return np.random.default_rng(rng)


def randomize_crater_ages(df, timestep, dtype=None, rng=None):
    """
    Return ages randomized uniformly between agelow, ageupp.
    """
    rng = get_rng(rng)
    # TODO: make sure ages are unique to each timestep?
    ages, agelow, ageupp = df[["age", "age_low", "age_upp"]].values.T
    new_ages = np.zeros(len(df), dtype=dtype)
    for i, (age, low, upp) in enumerate(zip(ages, agelow, ageupp)):
        new_ages[i] = round_to_ts(rng.uniform(age - low, age + upp), timestep)
    df["age"] = new_ages
    df = df.sort_values("age", ascending=False).reset_index(drop=True)
    return df


def get_random_hydrated_craters(diams, cfg, rng=None):
    """
    Return crater diams of hydrated craters from random distribution.
    """
    # Randomly include only craters formed by hydrated, Ctype asteroids
    rng = get_rng(rng)
    rand_arr = rng.random(size=len(diams))
    crater_diams = diams[rand_arr < cfg.ctype_frac * cfg.ctype_hydrated]
    return crater_diams


def get_random_impactor_speeds(n, cfg, rng=None):
    """
    Return n impactor speeds from normal distribution about mean, sd.
    """
    # Randomize impactor speeds with Gaussian around mean, sd
    rng = get_rng(rng)
    impactor_speeds = rng.normal(cfg.impact_speed_mean, cfg.impact_speed_sd, n)
    # Minimum possible impact speed is escape velocity
    impactor_speeds[impactor_speeds < cfg.escape_vel] = cfg.escape_vel
    return impactor_speeds


# Crater/impactor size-frequency helpers
@lru_cache(6)
def neukum(diam, neukum_version="2001"):
    """
    Return number of craters per m^2 per yr at diam [m] (eqn. 2, Neukum 2001).

    Eqn 2 expects diam [km], returns N [km^-2 Ga^-1].

    """
    if str(neukum_version) == "2001":
        a_vals = (
            -3.0876,
            -3.557528,
            0.781027,
            1.021521,
            -0.156012,
            -0.444058,
            0.019977,
            0.086850,
            -0.005874,
            -0.006809,
            8.25e-4,
            5.54e-5,
        )
    elif str(neukum_version) == "1983":
        a_vals = (
            -3.0768,
            -3.6269,
            0.4366,
            0.7935,
            0.0865,
            -0.2649,
            -0.0664,
            0.0379,
            0.0106,
            -0.0022,
            -5.18e-4,
            3.97e-5,
        )
    else:
        raise ValueError('Neukum version invalid (accepts "2001" or "1983").')
    diam = diam * 1e-3  # [m] -> [km]
    j = np.arange(len(a_vals))
    ncraters = 10 ** np.sum(a_vals * np.log10(diam) ** j)  # [km^-2 Ga^-1]
    return ncraters * 1e-6 * 1e-9  # [km^-2 Ga^-1] -> [m^-2 yr^-1]


def n_cumulative(diam, a, b):
    """
    Return N(D) from a cumulative size-frequency distribution aD^b.

    Parameters
    ----------
    diam (num): crater diameter [m]
    a (num): sfd scaling factor [m^-(2+b) yr^-1]
    b (num): sfd slope / exponent

    Return
    ------
    N(D): number of craters [m^-2 yr^-1]
    """
    return a * diam ** b


@lru_cache(1)
def get_impactors_brown(mindiam, maxdiam, timestep, c0=1.568, d0=2.7):
    """
    Return number of impactors per yr in range mindiam, maxdiam (Brown et al.
    2002) and scale by Earth-Moon impact ratio (Mazrouei et al. 2019).
    """
    n_impactors_gt_low = 10 ** (c0 - d0 * np.log10(mindiam))  # [yr^-1]
    n_impactors_gt_high = 10 ** (c0 - d0 * np.log10(maxdiam))  # [yr^-1]
    n_impactors_earth_yr = n_impactors_gt_low - n_impactors_gt_high
    n_impactors_moon = n_impactors_earth_yr * timestep / 22.5
    return n_impactors_moon


def get_crater_pop(time_arr, regime, cfg):
    """
    Return crater population assuming continuous (non-stochastic, regime c).
    """
    mindiam, maxdiam, step = cfg.diam_range[regime]
    slope = cfg.sfd_slopes[regime]
    diams, sfd_prob = get_crater_sfd(mindiam, maxdiam, step, slope, cfg.dtype)
    num_craters_t = num_craters_chronology(mindiam, maxdiam, time_arr, cfg)
    num_craters_t = np.atleast_1d(num_craters_t)[:, np.newaxis]  # make col vec
    return diams, num_craters_t, sfd_prob


def num_craters_chronology(mindiam, maxdiam, time, cfg):
    """
    Return number of craters [m^-2 yr^-1] mindiam and maxdiam at each time.
    """
    # Compute number of craters from neukum pf
    fmax = neukum(mindiam, cfg.neukum_pf_version)
    fmin = neukum(maxdiam, cfg.neukum_pf_version)
    count = (fmax - fmin) * cfg.sa_moon * cfg.timestep

    # Scale count by impact flux relative to present day flux
    num_craters = count * impact_flux(time) / impact_flux(0)
    return num_craters


def get_small_impactor_pop(time_arr, cfg):
    """
    Return population of impactors and number in regime B.

    Use constants and eqn. 3 from Brown et al. (2002) to compute N craters.
    """
    min_d, max_d, step = cfg.diam_range["b"]
    sfd_slope = cfg.sfd_slopes["b"]
    diams, sfd_prob = get_crater_sfd(min_d, max_d, step, sfd_slope, cfg.dtype)
    n_impactors = get_impactors_brown(min_d, max_d, cfg.timestep)

    # Scale n_impactors by historical impact flux, csfd shape: (NT, Ndiams)
    flux_scaling = impact_flux(time_arr[:, None]) / impact_flux(0)
    n_impactors_t = n_impactors * flux_scaling * sfd_prob
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
    sfd = diams ** sfd_slope
    return sfd / np.sum(sfd)


def get_diam_array(dmin, dmax, step, dtype=None):
    """Return array of diameters based on diameters in diam_range."""
    n = int((dmax - dmin) / step)
    return np.linspace(dmin, dmax, n + 1, dtype=dtype)


# @lru_cache(4)
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
    cfg,
):
    """
    Return size of impactors based on diam and sometimes speeds of craters.

    Different crater regimes are scaled via the following scaling laws:
    - regime=='c': (Prieur et al., 2017)
    - regime=='d': (Collins et al., 2005)
    - regime=='e': (Johnson et al., 2016)

    Parameters
    ----------
    diams (arr): Crater diameters [m].
    speeds (arr): Impactor speeds [m/s].
    regime (str): Crater scaling regime ('c', 'd', or 'e').

    Return
    ------
    lengths (arr): Impactor diameters [m].
    """
    t_diams = final2transient(diams, cfg.simple2complex)
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
    elif regime == "e":
        impactor_length = diam2len_johnson(
            t_diams,
            speeds,
            cfg.impactor_density,
            cfg.target_density,
            cfg.grav_moon,
            cfg.impact_angle,
            cfg.simple2complex,
        )
    elif regime == "f":
        impactor_length = diam2len_potter(
            t_diams,
            speeds,
            cfg.impactor_density,
            cfg.target_density,
            cfg.grav_moon,
            cfg.dtype,
        )
    else:
        raise ValueError(f"Invalid regime {regime} in diam2len")
    return impactor_length


def final2transient(diams, ds2c=18e3, gamma=1.25, eta=0.13):
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
    t_diams = diams / gamma
    t_diams[diams > ds2c] = (1 / gamma) * (
        diams[diams > ds2c] * ds2c ** eta
    ) ** (1 / (1 + eta))
    return t_diams


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
    numer = 1.6 * (1.61 * g * i_lengths / v ** 2) ** -0.22
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
        * v ** 0.44
        * g ** -0.22
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
        * v ** 0.5
        * g ** -0.25
        * ds2c ** -0.13
        * sin_theta ** 0.38
    )
    impactor_length = (diam / denom) ** (1 / 0.88)
    return impactor_length


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


# Surface age module
def get_age_grid(df, grdx, grdy, cfg):
    """Return final surface age of each grid point after all craters formed."""
    ny, nx = grdy.shape[0], grdx.shape[1]
    age_grid = cfg.timestart * np.ones((ny, nx), dtype=cfg.dtype)
    for _, crater in df.iterrows():
        age_grid = update_age(age_grid, crater, grdx, grdy, cfg.rad_moon)
    return age_grid


def update_age(age_grid, crater, grdx, grdy, rp=1737.4e3):
    """
    Return new age grid updating the points interior to crater with its age.
    """
    x, y = latlon2xy(crater.lat, crater.lon, rp)
    rad = crater.rad
    crater_mask = (np.abs(grdx - x) < rad) * (np.abs(grdy - y) < rad)
    age_grid[crater_mask] = crater.age
    return age_grid


def get_time_array(cfg):
    """
    Return time_array from tstart [yr] - tend [yr] by tstep [yr] and dtype.
    """
    n = int((cfg.timestart - cfg.timeend) / cfg.timestep)
    return np.linspace(cfg.timestart, cfg.timestep, n, dtype=cfg.dtype)


# Geospatial helpers
def get_grid_arrays(cfg):
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

    Return
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

    Return
    -------
    lat (num or arr): Latitude(s) [deg]
    lon (num or arr): Longitude(s) [deg]
    """
    z = np.sqrt(rp ** 2 - x ** 2 - y ** 2)
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

    Return
    ------
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

    Return
    ------
    x_rounded (int): Either floor(x) or ceil(x), rounded probabalistically
    """
    rng = get_rng(rng)
    x = np.atleast_1d(x)
    random_offset = rng.random(x.shape)
    x_rounded = np.floor(x + random_offset).astype(int)
    return x_rounded


def round_to_ts(values, timestep):
    """Return values rounded to nearest timestep."""
    return np.around(values / timestep) * timestep


def diam2vol(diameter):
    """Return volume of sphere [m^3] given diameter [m]."""
    return (4 / 3) * np.pi * (diameter / 2) ** 3


def m2km(x):
    """Convert meters to kilometers."""
    return x / 1000


def km2m(x):
    """Convert kilometers to meters."""
    return x * 1000


if __name__ == "__main__":
    # Get optional random seed and cfg file from cmd-line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "seed",
        type=int,
        nargs="?",
        help="random seed for this run - superceeds cfg.seed",
    )
    parser.add_argument(
        "--cfg", "-c", nargs="?", type=str, help="path to custom my_config.py"
    )
    args = parser.parse_args()

    # Use custom cfg from file if provided, else return default_config.Cfg
    cfg_dict = default_config.read_custom_cfg(args.cfg)

    # If seed given, it takes precedence over seed set in custom cfg
    if args.seed is not None:
        cfg_dict["seed"] = args.seed
    # TODO: remove (DEBUG)
    cfg_dict['seed'] = 12345
    custom_cfg = default_config.from_dict(cfg_dict)

    # Run model:
    _ = main(custom_cfg)
