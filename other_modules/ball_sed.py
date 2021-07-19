"""
Moon Polar Ice and Ejecta Stratigraphy module
Date: 07/06/21
Authors: CJ Tai Udovicic, K Frizzell, K Luchsinger, A Madera, T Paladino
Acknowledgements: This model is largely updated from Cannon et al. (2020)
"""
import argparse, ast, os
from functools import lru_cache, wraps
import numpy as np
import pandas as pd
import default_config


def main(cfg):
    """Run mixing model with options in cfg (see default_config.py)."""
    # Setup phase
    vprint = print if cfg.verbose else lambda *a, **k: None
    vprint("Initializing run...")
    rng = get_rng(cfg.seed)
    grdy, grdx = get_grid_arrays(cfg)
    time_arr = get_time_array(cfg)
    df = read_crater_list(cfg.crater_csv_in, cfg.crater_cols)  # len: NC
    df = randomize_crater_ages(df, cfg.timestep, cfg.dtype, rng)  # len: NC
    ej_thickness_time, bal_sed_time = get_ejecta_thickness_matrix(df, time_arr, cfg)  # [m]
    volcanic_ice_time = get_volcanic_ice(time_arr, cfg)  # [kg] len: NT
    overturn_depth_time = total_overturn_depth(time_arr, cfg)  # [m] len: NT

    # Init strat columns dict based for all cfg.coldtrap_craters
    strat_cols = init_strat_columns(df, time_arr, ej_thickness_time, cfg)
    
    # Main loop over time
    #vprint("Starting main loop...")
    
    for t, time in enumerate(time_arr):
        # Global ice mass gained [kg] by all processes
        global_ice = np.sum([
            volcanic_ice_time[t],  # Volcanic outgassing ice [kg]
            total_impact_ice(time, cfg, rng=rng)  # Impact delivered ice [kg]
        ])
        
        # Convert global ice mass [kg] to polar ice thickness [m]
        polar_ice_thickness = get_ice_thickness(global_ice, cfg)
        strat_cols = update_ice_cols(
            t,
            strat_cols,
            polar_ice_thickness,
            overturn_depth_time[t],
            bal_sed_time[t,:],
            cfg.mode,
        )
    
    # Format and save outputs
    vprint("Formatting outputs")
    outputs = format_save_outputs(strat_cols, time_arr, df, grdx, grdy, ej_thickness_time, cfg, vprint)
    return outputs

# Import data
def read_crater_list(crater_csv, columns, rp=1737e3):
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

    Columns defined here:
        - 'x': X-distance of crater center from S. pole [m]
        - 'y': Y-distance of crater center from S. pole [m]
        - 'dist2pole': Great circle distance of crater center from S. pole [m]

    Parameters
    ----------
    crater_csv (str): Path to crater list csv
    columns (list of str): List of names of all columns in crater_csv

    Returns
    -------
    df (DataFrame): Crater DataFrame read and updated from crater_csv
    """
    df = pd.read_csv(crater_csv, names=columns, header=0)

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

    # Define new columns
    df["x"], df["y"] = latlon2xy(df.lat, df.lon, rp)
    df["dist2pole"] = gc_dist(0, -90, df.lon, df.lat, rp)

    # Drop basins for now (>250 km diam)
    # TODO: handle basins somehow?
    df = df[df.diam <= 250e3]
    return df


def read_volcanic_csv(volcanic_csv, columns):
    df = pd.read_csv(volcanic_csv, names=columns, header=3)
    df["time"] = df["time"] * 1e9  # [Gyr -> yr]
    return df

def read_teqs(teq_csv):
    df = pd.read_csv(teq_csv, header=None, index_col=False)
    return df

@lru_cache(1)
def read_lambda_table(costello_csv):
    """Read lambda table (Table 1, Costello et al. 2018)."""
    df = pd.read_csv(costello_csv)
    return df


# Format and export model results
def format_csv_outputs(strat_cols, time_arr):
    """
    Return all formatted model outputs and write to outpath, if specified.
    """
    ej_dict = {"time": time_arr}
    ice_dict = {"time": time_arr}
    for cname, (ice_col, ej_col) in strat_cols.items():
        ej_dict[cname] = ej_col
        ice_dict[cname] = ice_col

    # Convert to DataFrames
    ej_cols_df = pd.DataFrame(ej_dict)
    ice_cols_df = pd.DataFrame(ice_dict)
    return ej_cols_df, ice_cols_df

def format_save_outputs(strat_cols, time_arr, df, grdx, grdy, ej_thickness, cfg, vprint=print):
    """
    Save outputs based on cfg.
    """
    df_outputs = format_csv_outputs(strat_cols, time_arr)        
    if cfg.write:
        # Save config file and dataframe outputs
        vprint(f"Saving outputs to {cfg.outpath}")
        save_outputs([cfg], [cfg.config_py_out])
        fnames = (cfg.ejcols_csv_out, cfg.icecols_csv_out)
        save_outputs(df_outputs, fnames)
        print(f"Outputs saved to {cfg.outpath}")
    
    if cfg.write_npy:
        # Note: Only compute these on demand (expensive to do every run)
        # Age grid is age of most recent impact (2D array: NX, NY)
        vprint("Computing age grid...")
        age_grid = get_age_grid(df, grdx, grdy, cfg.timestart, cfg.dtype)
        npy_fnames = (cfg.agegrd_npy_out, cfg.ejmatrix_npy_out)
        npy_outputs = [age_grid, ej_thickness]
        vprint(f"Saving npy outputs to {cfg.outpath}")
        save_outputs(npy_outputs, npy_fnames)
    return df_outputs


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
def get_crater_distances(df, symmetric=True, dtype=None):
    """
    Return 2D array of great circle dist between all craters in df. Distance
    from a crater to itself (or repeat distances if symmetric=False) are nan.

    Mandatory
        - df : Read in crater_list.csv as a DataFrame with defined columns
        - df : Required columns defined 'lat' and 'lon'
        - See 'read_crater_list' function

    Parameters
    ----------
    df (DataFrame): Crater DataFrame, e.g., read by read_crater_list
    TODO: Symmetric :

    Returns
    -------
    out (2D array): great circle distances between all craters in df
    """
    out = np.zeros((len(df), len(df)), dtype=dtype)
    for i in range(len(df)):
        for j in range(i):
            d = gc_dist(
                *df.iloc[i][["lon", "lat"]], *df.iloc[j][["lon", "lat"]]
            )
            out[i, j] = d
    if symmetric:
        out += out.T
    out[out <= 0] = np.nan
    return out


def get_ejecta_thickness(
    distance,
    radius,
    ds2c=18e3,
    order=-3,
    dtype=None,
):
    """
    Return ejecta thickness as a function of distance given crater radius.

    Complex craters McGetchin 1973
    """
    exp_complex = 0.74  # McGetchin 1973, simple craters exp=1
    exp = np.ones(radius.shape, dtype=dtype)
    exp[radius * 2 > ds2c] = exp_complex
    thickness = 0.14 * radius ** exp * (distance / radius) ** order
    thickness[np.isnan(thickness)] = 0
    # TODO: make this only cannon mode?
    thickness[distance < 4 * radius] = 0  # Cannon cuts off at 4 crater radii
    return thickness


def get_ejecta_thickness_matrix(
    df, time_arr, cfg
):
    """
    Return ejecta_matrix of thickness [m] at each time in time_arr given
    triangular matrix of distances between craters in df.

    Return
    ------
    ejecta_thick_time (3D array): Ejecta thicknesses (shape: NY, NX, NT)
    """
    # Symmetric matrix of distance from all craters to each other (NC, NC)
    ej_distances = get_crater_distances(df, cfg.dtype)
    
    # Ejecta thickness deposited in each crater from each crater (NC, NC)
    rad = df.rad.values[:, np.newaxis]  # need to pass column vector of radii
    ej_thick = get_ejecta_thickness(ej_distances, rad, cfg.simple2complex, cfg.ejecta_thickness_order, cfg.dtype)

    # also get equilibrium temperatures and depths for each event
    t_eqs = read_teqs(cfg.teq_csv_in)
    depths = ballistic_sed_depth(rad, t_eqs)

    # Find indices of crater ages in time_arr
    # Note: searchsorted must be ascending, so do -time_arr (-4.3, 0) Ga
    # BUG: python rounding issue - (only shoemaker index off by one)
    rounded_time = np.rint(time_arr / cfg.timestep)
    rounded_ages = np.rint(df.age.values / cfg.timestep)
    time_idx = np.searchsorted(-rounded_time, -rounded_ages)

    # Fill ejecta thickness vs time matrix (rows: time, cols:craters)
    ej_thick_time = np.zeros((len(time_arr), len(time_idx)), dtype=cfg.dtype)
    bal_sed_time = np.zeros((len(time_arr), len(time_idx)), dtype=cfg.dtype)
    for i, t_idx in enumerate(time_idx):
        # Sum here in case more than one crater formed at t_idx
        ej_thick_time[t_idx, :] += ej_thick[:, i]
        bal_sed_time[t_idx, :] += depths[:,i] 
    return ej_thick_time, bal_sed_time


def df2time(src_time, src_values, dst_time, timestep):
    """
    Return df_values inserted at nearest df_times in time_arr.
    
    Parameters
    ----------
    src_time (arr): Input array of times corresponding to each df_value.
    src_values (arr): Values to reshape as time_arr.
    dst_time (arr): Desired array of times to associate values with.
    timestep (num): Timestep of time_arr.

    Return
    ------
    dst_values (arr): Values at each corresponding time in dst_time.
    """
    # Round time and df arrays to nearest integer timestep
    src_time_rounded = np.rint(src_time / timestep)
    dst_time_rounded = np.rint(dst_time / timestep)
    
    # Find indices of df_time in time_arr
    time_idx = np.searchsorted(-src_time_rounded, -dst_time_rounded)

    # Loop through indices and sum if multiple formed at t_idx
    dst_values = np.zeros((len(src_time), len(time_idx)), dtype=cfg.dtype)
    for i, t_idx in enumerate(time_idx):
        dst_values[t_idx, :] += src_values[:, i]
    return dst_values


# Volcanic ice delivery module
def get_volcanic_ice(time_arr, cfg):
    """
    Return ice mass deposited in cold traps by volcanic outgassing over time.

    Values from supplemental spreadsheet S3 (Needham and Kring, 2017)
    transient atmosphere data. Scale by coldtrap_area and pole_pct % of
    material that is delievered to to S. pole.

    @author: tylerpaladino
    """
    if cfg.volc_mode == "NK":
        out = volcanic_ice_nk(
            time_arr,
            cfg.volc_csv_in,
            cfg.volc_cols,
            cfg.volc_species,
            cfg.volc_pole_pct,
            cfg.coldtrap_area,
            cfg.sa_moon,
        )
    elif cfg.volc_mode == "Head":
        out = volcanic_ice_head(
            time_arr,
            cfg.volc_early,
            cfg.volc_late,
            cfg.volc_early_pct,
            cfg.volc_late_pct,
            cfg.volc_total_vol,
            cfg.volc_h2o_ppm,
            cfg.volc_magma_density,
            cfg.ice_density,
            cfg.dtype,
        )
    else:
        raise ValueError(f"Invalid mode {cfg.mode}.")
    return out

def volcanic_ice_nk(
    time_arr,
    volc_csv,
    columns,
    species,
    pole_pct,
    coldtrap_area,
    moon_area,
):
    """
    Return ice [units] deposited in each timestep with Needham & Kring (2017).
    """
    df_volc = read_volcanic_csv(volc_csv, columns)

    # Outer merge df_volc with time_arr to get df with all age timesteps
    time_df = pd.DataFrame(time_arr, columns=["time"])
    df = time_df.merge(df_volc, on="time", how="outer")

    # Fill missing timesteps in df with linear interpolation across age
    df = df.sort_values("time", ascending=False).reset_index(drop=True)
    df_interp = df.set_index("time").interpolate()

    # Extract only relevant timesteps in time_arr and species column
    out = df_interp.loc[time_arr, species].values

    # Weight by fractional area of cold traps and ice transport pct
    area_frac = coldtrap_area / moon_area
    out *= area_frac * pole_pct
    return out

def volcanic_ice_head(
    time_arr,
    early=(4e9, 3e9),
    late=(3e9, 2e9),
    early_pct=0.75,
    late_pct=0.25,
    magma_vol=1e16,
    outgassed_h2o=10,
    magma_rho=3000,
    ice_rho=934,
    dtype=None,
):
    """
    Return ice [units] deposited in each timestep with Head et al. (2020).
    """
    # Global estimated H2O deposition
    tot_H2O_dep = magma_rho * magma_vol * ice_rho * outgassed_h2o / 1e6

    # Define periods of volcanism
    early_idx = (early[0] < time_arr) & (time_arr < early[1])
    late_idx = (late[0] < time_arr) & (time_arr < late[1])

    # H2O deposited per timestep
    H2O_early = tot_H2O_dep * early_pct / len(early_idx)
    H2O_late = tot_H2O_dep * late_pct / len(late_idx)

    out = np.zeros(len(time_arr), dtype=dtype)
    out[early_idx] = H2O_early
    out[late_idx] = H2O_late
    return out

def ballistic_sed_depth(R, teqs):
    depth = np.empty((np.shape(teqs)))
    for i in range(0,24):
        for j in range(0,24):
            if teqs.iloc[i,j] >= 120.:
                depth[i,j] = 1.
            else: 
                depth[i,j] = 0.
    return depth
# Ballistic sedimentation module
def ballistic_sed_ice_column(bal_sed_time):
    """
    Return ballistic sedimentation mixing depths for each crater.
    """
    #TODO?
    bal_cols =1
    return bal_cols


# Strat column functions
def init_strat_columns(df, time_arr, ej_cols, cfg):
    """
    Return dict of ice and ejecta columns for cold trap craters in df.

    Currently init at start of time_arr, but possibly:
    - TODO: init after crater formed? Or destroy some pre-existing ice here?
    - TODO: init only after PSR stable (Siegler 2015)?

    Return
    ------
    strat_columns_dict[cname] = [ice_col, ej_col, bal_col]
    """
    craters = cfg.coldtrap_craters
    strat_columns = {}
    idxs = np.where(df.cname.isin(craters).values)[0]
    ice_col = np.zeros(len(time_arr), cfg.dtype)
    bal_col = np.zeros(len(time_arr), cfg.dtype)
    strat_columns = {
        c: [ice_col.copy(), ej_cols[:, i]] for c, i in zip(craters, idxs)
    }

    return strat_columns


def update_ice_cols(t, strat_cols, new_ice_thickness, overturn_depth, bal_sed_time, mode):
    """
    Update ice_cols new ice added and ice eroded.
    """
    i=0
    # Update all tracked ice columns
    for cname, (ice_col, ej_col) in strat_cols.items():
        # TODO: Ballistic sed first, if crater was formed
        #ice_col = ballistic_sed_ice_column()
        # Ice gained by column
        ice_col[: t] = ballistic_remove_ice(
            ice_col[: t + 1], ej_col[: t + 1], bal_sed_time[i]
        )
        ice_col[t] = new_ice_thickness

        # Ice eroded in column
        ice_col = remove_ice_overturn(ice_col, ej_col, t, overturn_depth, mode)

        # Save ice column back to strat_cols dict
        strat_cols[cname][0] = ice_col
        i+=1
    return strat_cols


def get_ice_thickness(global_ice_mass, cfg):
    """
    Return ice thickness applied to all cold traps given total ice mass
    produced globally, scaled by ice_hop_efficiency, density of ice and 
    total coldtrap_area.
    """
    polar_ice_mass = global_ice_mass * cfg.ice_hop_efficiency  # [kg]
    ice_volume = polar_ice_mass / cfg.ice_density  # [m^3]
    ice_thickness = ice_volume / cfg.coldtrap_area  # [m]
    return ice_thickness


# Impact overturn removal of ice
def remove_ice_overturn(ice_col, ej_col, t, overturn_depth, bal_sed_time, mode="cannon"):
    """Return ice_col with ice removed due to impact overturn via current mode."""
    if mode == "cannon":
        ice_col = erode_ice_cannon(ice_col, ej_col, t, overturn_depth)
    elif mode == "moonpies":
        ice_col[: t + 1] = garden_ice_column(
            ice_col[: t + 1], ej_col[: t + 1], overturn_depth
        )
    else:
        raise ValueError(f'Invalid mode {mode} in remove_ice_overturn')
    return ice_col


def erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4):
    """
    Return eroded ice column using Cannon et al. (2020) method (10 cm / 10 Ma).
    """
    # BUG in Cannon ds01: erosion base not updated for adjacent ejecta layers
    # Erosion base is most recent time when ejecta column was > ejecta_shield
    erosion_base = np.argmax(ej_col[: t + 1] > ej_shield)

    # Garden from top of ice column until ice_to_erode amount is removed
    # BUG in Cannon ds01: doesn't account for partial shielding by small ej
    layer = t
    while overturn_depth > 0 and layer >= 0:
        if t < erosion_base:
            # if layer < erosion_base:
            # End loop if we reach erosion base
            # BUG in Cannon ds01: t should be layer
            # - loop doesn't end if we reach erosion base while eroding
            # - loop only ends here if we started at erosion_base
            break
        ice_in_layer = ice_col[layer]
        if ice_in_layer >= overturn_depth:
            ice_col[layer] -= overturn_depth
        else:
            ice_col[layer] = 0
        overturn_depth -= ice_in_layer
        layer -= 1
    return ice_col

def ballistic_remove_ice(ice_column, ejecta_column, bal_depth):
    """
    Return ice_column gardened to overturn_depth, preserved by ejecta_column.

    Ejecta deposited on last timestep preserves ice. Loop through ice_col and
    ejecta_col until overturn_depth and remove all ice that is encountered.
    """
    i = 0  # current loop iter
    d = 0  # current depth
    while d < bal_depth and i < 2 * len(ice_column):
        if i % 2:
            # Odd i (ejecta): do nothing except add to current depth, d
            d += ejecta_column[-i // 2]
        else:
            # Even i (ice): remove all ice from this layer, add it to depth, d
            d += ice_column[-i // 2]
            ice_column[-i // 2] = 0
        i += 1
    # If odd i (ice) on last iter, we likely removed too much ice
    #   Add back any excess depth we travelled to ice_col
    if i % 2 and d > bal_depth:
        ice_column[-i // 2] = d - bal_depth
    return ice_column

def garden_ice_column(ice_column, ejecta_column, overturn_depth):
    """
    Return ice_column gardened to overturn_depth, preserved by ejecta_column.

    Ejecta deposited on last timestep preserves ice. Loop through ice_col and
    ejecta_col until overturn_depth and remove all ice that is encountered.
    """
    i = 0  # current loop iter
    d = 0  # current depth
    while d < overturn_depth and i < 2 * len(ice_column):
        if i % 2:
            # Odd i (ejecta): do nothing except add to current depth, d
            d += ejecta_column[-i // 2]
        else:
            # Even i (ice): remove all ice from this layer, add it to depth, d
            d += ice_column[-i // 2]
            ice_column[-i // 2] = 0
        i += 1
    # If odd i (ice) on last iter, we likely removed too much ice
    #   Add back any excess depth we travelled to ice_col
    if i % 2 and d > overturn_depth:
        ice_column[-i // 2] = d - overturn_depth
    return ice_column


# Impact gardening module (Costello et al. 2018, 2020)
def overturn_depth(
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


def total_overturn_depth(time_arr, cfg):
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
    if cfg.mode == "cannon":
        return 0.1 * np.ones(len(time_arr), dtype=cfg.dtype)
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
        overturn = overturn_depth(
            u,
            b,
            cfg.costello_csv_in,
            cfg.timestep,
            cfg.n_overturn,
            cfg.overturn_prob_pct,
        )
        overturn_depths.append(overturn)
    overturn_total = np.sum(overturn_depths, axis=0)
    return overturn_total


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
def total_impact_ice(time, cfg, rng=None):
    """Return total impact ice from regimes and sfd_slopes (Cannon 2020)."""
    total_ice = 0  # [kg]
    for r in cfg.impact_regimes:
        if r == "a":
            # Micrometeorites
            total_ice += ice_micrometeorites(
                time,
                cfg.timestep,
                cfg.mm_mass_rate,
                cfg.hydrated_wt_pct,
                cfg.impactor_mass_retained,
            )
        elif r == "b":
            # Small impactors
            impactor_diams, impactors = get_impactor_pop(
                time, r, cfg.timestep, cfg.diam_range, cfg.sfd_slopes, cfg.dtype
            )
            total_ice += ice_small_impactors(impactor_diams, impactors, cfg)
        elif r == "c":
            # Small simple craters (continuous)
            crater_diams, craters = get_crater_pop(
                time,
                r,
                cfg.timestep,
                cfg.diam_range,
                cfg.sfd_slopes,
                cfg.sa_moon,
                cfg.ivanov2000,
                rng=rng,
            )
            total_ice += ice_small_craters(crater_diams, craters, r, cfg)
        else:
            # Large simple & complex craters (stochastic)
            crater_diams = get_crater_pop(
                time,
                r,
                cfg.timestep,
                cfg.diam_range,
                cfg.sfd_slopes,
                cfg.sa_moon,
                cfg.ivanov2000,
                rng=rng,
            )
            crater_diams = get_random_hydrated_craters(
                crater_diams, cfg.ctype_frac, cfg.ctype_hydrated, rng=rng
            )
            impactor_speeds = get_random_impactor_speeds(
                len(crater_diams),
                cfg.impact_speed,
                cfg.impact_sd,
                cfg.escape_vel,
                rng=rng,
            )
            total_ice += ice_large_craters(crater_diams, impactor_speeds, r, cfg)
    return total_ice


def ice_micrometeorites(
    time=0,
    timestep=10e6,
    mm_mass_rate=1e6,
    hyd_wt_pct=0.1,
    mass_retained=0.165,
):
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
    scaling = hyd_wt_pct * impact_flux(time) / impact_flux(0)
    micrometeorite_ice = timestep * scaling * mm_mass_rate * mass_retained
    # TODO: Why don't we account for 36% CC and 2/3 of CC hydrated (like regime B, C)
    # TODO: improve micrometeorite flux?
    return micrometeorite_ice


def ice_small_impactors(diams, num_per_bin, cfg):
    """
    Return ice mass [kg] from small impactors (Regime B, Cannon 2020) given
    impactor diams, num_per_bin, and density.
    """
    impactor_masses = diam2vol(diams) * cfg.impactor_density
    total_impactor_mass = np.sum(impactor_masses * num_per_bin)
    total_impactor_water = impactor_mass2water(
        total_impactor_mass,
        cfg.ctype_frac,
        cfg.ctype_hydrated,
        cfg.hydrated_wt_pct,
        cfg.impactor_mass_retained,
    )
    return total_impactor_water


def ice_small_craters(
    crater_diams,
    ncraters,
    regime,
    cfg,
):
    """
    Return ice from simple craters, steep branch (Regime C, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams, cfg.impact_speed, regime, cfg) 
    impactor_masses = diam2vol(impactor_diams) * cfg.impactor_density  # [kg]
    total_impactor_mass = np.sum(impactor_masses * ncraters)
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
    ice_retention = ice_retention_factor(impactor_speeds, cfg.dtype)
    ice_masses = impactor_masses * cfg.hydrated_wt_pct * ice_retention
    return np.sum(ice_masses)


def ice_retention_factor(speeds, dtype=None):
    """
    Return ice retained in impact, given impactor speeds (Cannon 2020).

    For speeds < 10 km/s, retain 50% (Svetsov & Shuvalov 2015 via Cannon 2020).
    For speeds >= 10 km/s, use eqn ? (Ong et al. 2010 via Cannon 2020)
    """
    # TODO: find/verify retention(speed) eqn in Ong et al. 2010?
    # BUG? retention distribution is discontinuous
    speeds = speeds * 1e-3  # [m/s] -> [km/s]
    retained = np.ones(len(speeds), dtype=dtype) * 0.5  # nominal 50%
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
    df = df.sort_values("age", ascending=False)
    return df


def get_random_hydrated_craters(
    crater_diams,
    ctype_frac=0.36,
    ctype_hyd=2 / 3,
    rng=None,
):
    """
    Return crater diams of hydrated craters from random distribution.
    """
    # Randomly include only craters formed by hydrated, Ctype asteroids
    rng = get_rng(rng)
    rand_arr = rng.random(size=len(crater_diams))
    crater_diams = crater_diams[rand_arr < ctype_frac * ctype_hyd]
    return crater_diams


def get_random_impactor_speeds(
    n,
    mean_speed=20e3,
    sd_speed=6e3,
    esc_vel=2.38e3,
    rng=None,
):
    """
    Return n impactor speeds from normal distribution about mean, sd.
    """
    # Randomize impactor speeds with Gaussian around mean, sd
    rng = get_rng(rng)
    impactor_speeds = rng.normal(mean_speed, sd_speed, n)
    impactor_speeds[impactor_speeds < esc_vel] = esc_vel  # minimum is esc_vel
    return impactor_speeds


# Crater/impactor size-frequency helpers
@lru_cache(6)
def neukum(diam, a_values):
    """
    Return number of craters per m^2 per yr at diam [m] (eqn. 2, Neukum 2001).

    Eqn 2 expects diam [km], returns N [km^-2 Ga^-1].

    """
    diam = diam * 1e-3  # [m] -> [km]
    j = np.arange(len(a_values))
    ncraters = 10 ** np.sum(a_values * np.log10(diam) ** j)  # [km^-2 Ga^-1]
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
    Return number of impactors per yr in range (mindiam, maxdiam) [m]
    (Brown et al. 2002) and scale by Earth-Moon impact ratio (Mazrouei et al. 2019).
    """
    n_impactors_gt_low = 10 ** (c0 - d0 * np.log10(mindiam))  # [yr^-1]
    n_impactors_gt_high = 10 ** (c0 - d0 * np.log10(maxdiam))  # [yr^-1]
    n_impactors_earth_yr = n_impactors_gt_low - n_impactors_gt_high
    n_impactors_moon = n_impactors_earth_yr * timestep / 22.5
    return n_impactors_moon


def get_crater_pop(
    time,
    regime,
    timestep,
    diam_range,
    sfd_slopes,
    sa_moon,
    csfd_coeffs,
    rng=None,
    dtype=None,
):
    """
    Return population of crater diameters and number (regimes C - E).

    Weight small simple craters by size-frequency distribution.

    Randomly resample large simple & complex crater diameters.
    """
    crater_diams, sfd_prob = get_diams_probs(*diam_range[regime], sfd_slopes[regime], dtype)
    n_craters = neukum(crater_diams[0], csfd_coeffs) - neukum(
        crater_diams[-1], csfd_coeffs
    )
    # Scale for timestep, surface area and impact flux
    n_craters *= timestep * sa_moon * impact_flux(time) / impact_flux(0)
    if regime == "c":
        # Steep branch of sfd (simple)
        n_craters *= sfd_prob
        return crater_diams, n_craters

    # Regimes D and E: shallow branch of sfd (simple / complex)
    n_craters = probabilistic_round(n_craters, rng=rng)

    # Resample crater diameters with replacement, weighted by sfd
    rng = get_rng(rng)
    crater_diams = rng.choice(crater_diams, n_craters, p=sfd_prob)
    return crater_diams


def get_impactor_pop(
    time, regime, timestep, diam_range, sfd_slopes, dtype=None
):
    """
    Return population of impactors and number in regime B.

    Use constants and eqn. 3 from Brown et al. (2002) to compute N craters.
    """
    diams, sfd_prob = get_diams_probs(*diam_range[regime], sfd_slopes[regime], dtype)
    n_impactors = get_impactors_brown(diams[0], diams[-1], timestep)

    # Scale for timestep, impact flux and size-frequency dist
    flux_scaling = impact_flux(time) / impact_flux(0)
    n_impactors *= flux_scaling * sfd_prob
    return diams, n_impactors


@lru_cache(4)
def get_diams_probs(dmin, dmax, step, sfd_slope, dtype=None):
    """
    Return diam_array and sfd_prob. This func makes it easier to cache both.
    """
    diam_array = get_diam_array(dmin, dmax, step, dtype)
    sfd_prob = get_sfd_prob(diam_array, sfd_slope)
    return diam_array, sfd_prob


def get_sfd_prob(diams, sfd_slope):
    """Return size-frequency distribution probability given diams, sfd slope."""
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
    t_diams = final2transient(diams, cfg.grav_moon, cfg.simple2complex)
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
            diams,
            speeds,
            cfg.impactor_density,
            cfg.target_density,
            cfg.grav_moon,
            cfg.impact_angle,
            cfg.simple2complex,
        )
    else:
        raise ValueError(f"Invalid regime {regime} in diam2len")
    return impactor_length


def final2transient(diams, g=1.62, ds2c=18e3, gamma=1.25, eta=0.13):
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

    # diams < simple2complex == diam/gamma, else use complex scaling
    t_diams = np.copy(diams) / gamma
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
    theta (num): impact angle (degrees)

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
    Return impactor length from final crater diam using Johnson et al. (2016)
    method. TODO: Only valid for diam > ds2c.

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


# Surface age module
def get_age_grid(
    df,
    grdx,
    grdy,
    timestart,
    dtype=None,
):
    """Return final surface age of each grid point after all craters formed."""
    ny, nx = grdy.shape[0], grdx.shape[1]
    age_grid = timestart * np.ones((ny, nx), dtype=dtype)
    for _, crater in df.iterrows():
        age_grid = update_age(age_grid, crater, grdx, grdy)
    return age_grid


def update_age(age_grid, crater, grdx, grdy):
    """
    Return new age grid updating the points interior to crater with its age.
    """
    x, y, rad = crater.x, crater.y, crater.rad
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
    ysize, ystep =  cfg.grdysize, cfg.grdstep
    xsize, xstep = cfg.grdxsize, cfg.grdstep
    grdy, grdx = np.meshgrid(
        np.arange(ysize, -ysize, -ystep, dtype=cfg.dtype),
        np.arange(-xsize, xsize, xstep, dtype=cfg.dtype),
        sparse=True,
        indexing="ij",
    )
    return grdy, grdx


def latlon2xy(lat, lon, rp=1737e3):
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


def xy2latlon(x, y, rp=1737e3):
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


def gc_dist(lon1, lat1, lon2, lat2, rp=1737e3):
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
    gc_dist = rp * c
    return gc_dist


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
    x_rounded = int(np.floor(x + rng.random()))
    return x_rounded


def round_to_ts(values, timestep):
    """Return values rounded to nearest timestep."""
    return np.around(values / timestep) * timestep


def diam2vol(diameter):
    """Return volume of sphere [m^3] given diameter [m]."""
    return (4 / 3) * np.pi * (diameter / 2) ** 3


if __name__ == "__main__":
    # Get optional random seed and cfg options from cmd-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', 
                        type=int, 
                        nargs='?', 
                        # default=0, 
                        help='random seed for this run - superceeds cfg.seed')
    parser.add_argument('--cfg', '-c', 
                        nargs='?', 
                        type=str, 
                        help='path to custom my_config.py')
    args = parser.parse_args()
    cfg_dict = {}
    if args.cfg:  # Read config options from args.cfg path
        cfg_dict = default_config.read_custom_cfg(args.cfg)
    if args.seed:  # If seed given, it takes precedence
        cfg_dict['seed'] = args.seed    
    elif 'seed' not in cfg_dict: # If no seed given and no cfg seed, randomize
        cfg_dict['seed'] = np.random.randint(1, 99999)
    
    # Configure run (populates blank fields with defaults in default_config.py)
    cfg = default_config.Cfg(**cfg_dict)
    _ = main(cfg)
