"""
Main mixing module updated from Cannon et al. (2020)
Date: 07/05/21
Authors: CJ Tai Udovicic, K Frizzell, K Luchsinger, A Madera, T Paladino

All model results are saved to OUTPATH.

# From command line (requires python, numpy, pandas)
     <seed> is any integer (default 0) and makes run deterministic
     change seed between runs to get different randomness

python mixing.py <seed>

# From Jupyter, make sure to os.chdir to location of this file:
    use import mixing to run any function in this file e.g. mixing.main()

import os
os.chdir('/home/cjtu/projects/essi21/essi21')
import mixing as mm
ej_cols, ice_cols, run_meta = mixing.main()

# Run with gnuparallel (6 s/run normal, 0.35 s/run 48 cores)
    parallel -P-1 uses all cores except 1

conda activate essi
seq 10000 | parallel -P-1 python mixing.py

# Code Profiling (pip install snakeviz)

python -m cProfile -o mixing.prof mixing.py
snakeviz mixing.prof
"""
import os
import sys
from functools import lru_cache
import numpy as np
import pandas as pd
from essi21 import config

# TODO: remove all CFG
# Functions
def main(write=True, cfg=CFG, rng=None, verbose=CFG._verbose):
    """
    Before loop:
      1) read_crater_list(): Reads in crater list from CRATER_CSV above.
      2) Get ejecta thickness matrix
      3) Get volcanic ice matrix
      4) Get ballistic sed matrix
      5) Get sublimation rate
    Main loop for mixing model.
      1) Every timestep:
        a. get impact ice mass
        b. get volcanic ice mass
      2) If a crater is formed this timestep:
        a. deposit ejecta
        b. update age grid
    """
    # Before loop (setup)
    if verbose:
        print("Initializing run...")
    df = read_crater_list()  # DataFrame, len: NC
    # df = randomize_crater_ages(df)  # DataFrame, len: NC
    ej_thickness_time = get_ejecta_thickness_matrix(df)  # [m] shape: NY,NX,NT
    volcanic_ice_time = get_volcanic_ice(mode=cfg.volc_mode)  # [kg] len: NT
    overturn_depth_time = total_overturn_depth(mode=cfg.mode)  # [m] len: NT
    # ballistic_sed_matrix = get_ballistic_sed(df, mode=cmode)
    # sublimation_depth_time = get_sublimation_rate(mode=mmode)  # [m] len: NT

    # Init ice columns dictionary based on desired COLD_TRAP_CRATERS
    strat_cols = init_strat_columns(df, ej_thickness_time)

    # Main time loop
    if verbose:
        print("Starting main loop...")
    for t, time in enumerate(cfg._time_arr):
        # Global ice mass gained [kg] by all processes
        global_ice = volcanic_ice_time[t] + total_impact_ice(time, rng=rng)

        # South polar ice gain [kg]
        polar_ice = global_ice * cfg.ice_hop_efficiency

        # Convert mass [kg] to thickness [m]
        polar_ice_thickness = get_ice_thickness(polar_ice)
        strat_cols = update_ice_cols(
            t, strat_cols, polar_ice_thickness, overturn_depth_time[t]
        )

    # Format and save outputs
    if verbose:
        print("Formatting outputs")
    df_outputs = format_outputs(strat_cols)
    if write:
        if verbose:
            print(f"Saving outputs to {cfg.outdir}")
        fnames = (cfg.ejcols_csv_out, cfg.icecols_csv_out, cfg.runmeta_csv_out)
        save_outputs(df_outputs, fnames)
        if cfg.save_npy:
            # Compute these on demand (expensive to do every run)
            age_grid = get_age_grid(
                df
            )  # shape: (NY, NX) age of youngest impact
            npy_fnames = (cfg.agegrd_npy_out, cfg.ejmatrix_npy_out)
            npy_outputs = [age_grid, ej_thickness_time]
            save_outputs(npy_outputs, npy_fnames)
    return df_outputs


def get_age_grid(
    df,
    grdx=CFG._grdx,
    grdy=CFG._grdy,
    timestart=CFG.timestart,
    dtype=CFG.dtype,
):
    """Return final surface age of each grid point after all craters formed."""
    ny, nx = grdy.shape[0], grdx.shape[1]
    age_grid = timestart * np.ones((ny, nx), dtype=dtype)
    for _, crater in df.iterrows():
        age_grid = update_age(age_grid, crater, grdx, grdy)
    return age_grid


def get_ejecta_thickness_matrix(
    df, time_arr=CFG._time_arr, ts=CFG.timestep, dtype=CFG.dtype
):
    """
    Return ejecta_matrix of thickness [m] at each time in time_arr given
    triangular matrix of distances between craters in df.
    """
    # Symmetric matrix of distance from all craters to each other (NC, NC)
    ej_distances = get_crater_distances(df)

    # Ejecta thickness deposited in each crater from each crater (NC, NC)
    rad = df.rad.values[:, np.newaxis]  # need to pass column vector of radii
    ej_thick = get_ejecta_thickness(ej_distances, rad)

    # Find indices of crater ages in time_arr
    # Note: searchsorted must be ascending, so do -time_arr (-4.3, 0) Ga
    # BUG: python rounding issue - (only shoemaker index off by one)
    rounded_time = np.rint(time_arr / ts)
    rounded_ages = np.rint(df.age.values / ts)
    time_idx = np.searchsorted(-rounded_time, -rounded_ages)

    # Fill ejecta thickness vs time matrix (rows: time, cols:craters)
    ej_thick_time = np.zeros((len(time_arr), len(time_idx)), dtype=dtype)
    for i, t_idx in enumerate(time_idx):
        # Sum here in case more than one crater formed at t_idx
        ej_thick_time[t_idx, :] += ej_thick[:, i]
    return ej_thick_time


def grid_interp(x, y, grdvalues, grdx=CFG._grdx, grdy=CFG._grdy):
    """Return ejecta thickness at (x, y) in ejecta_thickness 2D grid."""
    ix, iy = get_grd_ind(x, y, grdx, grdy)
    gx, gy = grdx.flatten(), grdy.flatten()

    dy = (y - gy[iy]) / (gy[iy + 1] - gy[iy])
    interp_y = (1 - dy) * grdvalues[iy] + dy * grdvalues[iy + 1]

    dx = (x - gx[ix]) / (gx[ix + 1] - gx[ix])
    interp = (1 - dx) * interp_y[ix] + dx * interp_y[ix + 1]

    return interp


def format_outputs(strat_cols, time_arr=CFG._time_arr, cfg=CFG):
    """
    Return all formatted model outputs and write to outpath, if specified.
    """
    ej_dict = {"time": time_arr}
    ice_dict = {"time": time_arr}
    for cname, (ice_col, ej_col) in strat_cols.items():
        ej_dict[cname] = ej_col
        ice_dict[cname] = ice_col

    # Save cfg excluding "_" fields
    run_meta = cfg.to_dict_no_underscore()

    # Convert to DataFrames
    ej_cols_df = pd.DataFrame(ej_dict)
    ice_cols_df = pd.DataFrame(ice_dict)
    run_meta_df = pd.DataFrame.from_dict(
        run_meta, orient="index"
    ).reset_index()
    return ej_cols_df, ice_cols_df, run_meta_df


def save_outputs(outputs, fnames, verbose=CFG._verbose):
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
        if verbose:
            print(f"Saved {fout}")
    print(f"All outputs saved to {outpath}")


def round_to_ts(values, timestep):
    """Return values rounded to nearest timestep."""
    return np.around(values / timestep) * timestep


def randomize_crater_ages(
    df, timestep=CFG.timestep, dtype=CFG.dtype, rng=None
):
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


def update_age(age_grid, crater, grdx=CFG._grdx, grdy=CFG._grdy):
    """
    Return new age grid updating the points interior to crater with its age.
    """
    x, y, rad = crater.x, crater.y, crater.rad
    crater_mask = (np.abs(grdx - x) < rad) * (np.abs(grdy - y) < rad)
    age_grid[crater_mask] = crater.age
    return age_grid


def update_ice_cols(
    t,
    strat_cols,
    new_ice_thickness,
    overturn_depth=0,
    # sublimation_thickness,
    # ballistic_sed_matrix,
    mode=CFG.mode,
):
    """
    Update ice_cols new ice added and ice eroded.
    """
    # Update all tracked ice columns
    for cname, (ice_col, ej_col) in strat_cols.items():
        # TODO: Ballistic sed first, if crater was formed
        # ice_col = ballistic_sed_ice_column()

        # Ice gained by column
        ice_col[t] = new_ice_thickness

        # Ice eroded in column
        ice_col = remove_ice_overturn(ice_col, ej_col, t, overturn_depth, mode)

        # Save ice column back to strat_cols dict
        strat_cols[cname][0] = ice_col
    return strat_cols


def read_crater_list(crater_csv=CFG.crater_csv_in, columns=CFG.crater_cols):
    """
    Return dataframe of craters from path to crater_csv with columns names.

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
    columns (list of str): Names of columns in csv in order

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
    df["x"], df["y"] = latlon2xy(df.lat, df.lon)
    df["dist2pole"] = gc_dist(0, -90, df.lon, df.lat)

    # Drop basins for now (>250 km diam)
    # TODO: handle basins somehow?
    df = df[df.diam <= 250e3]
    return df


def read_volcanic_csv(volcanic_csv=CFG.volc_csv_in, col=CFG.volc_cols):
    df = pd.read_csv(volcanic_csv, names=col, header=3)
    df["age"] = df["age"] * 1e9  # [Gyr -> yr]
    return df


# Pre-compute grid functions
def get_crater_distances(df, symmetric=True, dtype=CFG.dtype):
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
    exp_complex=0.74,
    ds2c=CFG.simple2complex,
    order=CFG.ejecta_thickness_order,
    dtype=CFG.dtype,
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


def get_volcanic_ice(time_arr=CFG._time_arr, mode="Needham"):
    """
    Return ice mass deposited in cold traps by volcanic outgassing over time.

    Values from supplemental spreadsheet S3 (Needham and Kring, 2017)
    transient atmosphere data. Scale by coldtrap_area and pole_pct % of
    material that is delievered to to S. pole.

    @author: tylerpaladino
    """
    if mode == "Needham":
        out = volcanic_ice_needham(time_arr)
    elif mode == "Head":
        out = volcanic_ice_head(time_arr)
    else:
        raise ValueError(f"Invalid mode {mode}.")
    return out


def volcanic_ice_needham(
    time_arr,
    f=CFG.volc_csv_in,
    cols=CFG.volc_cols,
    species=CFG.volc_species,
    pole_pct=CFG.volc_pole_pct,
    coldtrap_area=CFG.coldtrap_area,
    moon_area=CFG.sa_moon,
):
    """
    Return ice [units] deposited in each timestep with Needham & Kring (2017).
    """
    df_volc = read_volcanic_csv(f, cols)

    # Outer merge df_volc with time_arr to get df with all age timesteps
    time_df = pd.DataFrame(time_arr, columns=["age"])
    df = time_df.merge(df_volc, on="age", how="outer")

    # Fill missing timesteps in df with linear interpolation across age
    df = df.sort_values("age", ascending=False).reset_index(drop=True)
    df_interp = df.set_index("age").interpolate()

    # Extract only relevant timesteps in time_arr and species column
    out = df_interp.loc[time_arr, species].values

    # Weight by fractional area of cold traps and ice transport pct
    area_frac = coldtrap_area / moon_area
    out *= area_frac * pole_pct
    return out


def volcanic_ice_head(
    time_arr,
    early=CFG.volc_early,
    late=CFG.volc_late,
    early_pct=CFG.volc_early_pct,
    late_pct=CFG.volc_late_pct,
    magma_vol=CFG.volc_total_vol,
    outgassed_h2o=CFG.volc_h2o_ppm,
    magma_rho=CFG.volc_magma_density,
    ice_rho=CFG.ice_density,
    dtype=CFG.dtype,
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


def get_ice_thickness(
    mass, density=CFG.ice_density, cold_trap_area=CFG.coldtrap_area
):
    """
    Return ice thickness applied to all cold traps given total ice mass
    produced, density of ice and total cold_trap_area.
    """
    ice_volume = mass / density  # [m^3]
    ice_thickness = ice_volume / cold_trap_area  # [m]
    return ice_thickness


def get_ballistic_sed(df):
    """
    Return ballistic sedimentation mixing depths for each crater.
    """
    # TODO: add Kristen code
    return np.zeros((CFG._ny, CFG._nx, len(df)))


def ballistic_planar(theta, d, g=CFG.grav_moon):
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


def ballistic_spherical(theta, d, g=CFG.grav_moon, rp=CFG.rad_moon):
    """
    Return ballistic speed (v) given ballistic range (d) and gravity of planet (g).
    Assumes perfectly spherical planet (Vickery, 1986).

    Parameters
    ----------
    d (num or array): ballistic range [m]
    g (num): gravitational force of the target body [m s^-2]
    theta (num): angle of impaact [radians]
    rp (num): radius of the target body [m]

    Returns
    -------
    v (num or array): ballistic speed [m s^-1]

    """
    tan_phi = np.tan(d / (2 * rp))
    return np.sqrt(
        (g * rp * tan_phi)
        / ((np.sin(theta) * np.cos(theta)) + (np.cos(theta) ** 2 * tan_phi))
    )


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


def thick2mass(thick, density=CFG.target_density):
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


def ice_melted(
    ke,
    t_surf=CFG.coldtrap_max_temp,
    cp=CFG.regolith_cp,
    ice_rho=CFG.ice_density,
    ice_frac=CFG.ice_frac,
    heat_frac=CFG.heat_frac,
    heat_ret=CFG.heat_retained,
    lat_heat=CFG.ice_latent_heat,
    t_melt=CFG.ice_melt_temp,
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
    ice_depth (num or array): depth of ice melted due to ejecta [m]
    """
    heat = ice_frac * heat_ret * heat_frac * ke  # [J m^-2]
    delta_t = t_melt - t_surf  # heat to go from t_surf to melting point
    ice_mass = heat / (lat_heat + cp * delta_t)  # [kg]
    # ice_depth = ice_mass / (ice_frac * ice_rho)  # [m]
    return ice_mass


def get_sublimation_rate(timestep=CFG.timestep, temp=CFG.coldtrap_max_temp):
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


# Ice column functions
def get_grd_ind(x, y, grdx=CFG._grdx, grdy=CFG._grdy):
    """Return index of val in monotonic grid_axis."""
    xidx = np.searchsorted(grdx[0, :], x) - 1
    yidx = grdy.shape[0] - (np.searchsorted(-grdy[:, 0], y) - 1)
    return (xidx, yidx)


def init_strat_columns(
    df,
    ej_cols,
    craters=CFG.cold_trap_craters,
    time_arr=CFG._time_arr,
    dtype=CFG.dtype,
):
    """
    Return dict of ice and ejecta columns for cold trap craters in df.

    Currently init at start of time_arr, but possibly:
    - TODO: init after crater formed? Or destroy some pre-existing ice here?
    - TODO: init only after PSR stable (Siegler 2015)?

    Return
    ------
    strat_columns_dict[cname] = [ice_col, ej_col]
    """
    strat_columns = {}
    idxs = np.where(df.cname.isin(craters).values)[0]
    ice_col = np.zeros(len(time_arr), dtype)
    strat_columns = {
        c: [ice_col.copy(), ej_cols[:, i]] for c, i in zip(craters, idxs)
    }
    return strat_columns


def ballistic_sed_ice_column(c, ice_column, ballistic_sed_matrix):
    """Return ice column with ballistic sed grid applied"""
    ballistic_sed_grid = ballistic_sed_matrix[:, :, c]
    # TODO: add code from Kristen
    return ice_column


# Impact overturn removal of ice
def remove_ice_overturn(ice_col, ej_col, t, overturn_depth, mode="cannon"):
    """Return ice_col with ice removed due to impact overturn via current mode."""
    if mode == "cannon":
        ice_col = erode_ice_cannon(ice_col, ej_col, t, overturn_depth)
    else:
        ice_col[: t + 1] = garden_ice_column(
            ice_col[: t + 1], ej_col[: t + 1], overturn_depth
        )
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


def garden_ice_column(ice_column, ejecta_column, overturn_depth):
    """
    Return ice_column gardened to overturn_depth, preserved by ejecta_column.

    Ejecta deposited on last timestep preserves ice. Loop through ice_col and
    ejecta_col until overturn_depth and remove all ice that is encountered.

    TODO: garden new ice first, new ejecta next (swap evens and odds)
    """
    i = 0  # current loop iter
    d = 0  # current depth
    while d < overturn_depth and i < 2 * len(ice_column):
        if i % 2:
            # Odd i (ice): remove all ice from this layer, add it to depth
            d += ice_column[-i // 2]
            ice_column[-i // 2] = 0
        else:
            # Even i (ejecta): do nothing, add to current depth
            d += ejecta_column[-i // 2]
        i += 1
    # If odd i (ice) on last iter, we likely removed too much ice
    #   Add back any excess depth we travelled to ice_col
    if (i - 1) % 2 and d > overturn_depth:
        ice_column[-i // 2] = d - overturn_depth
    return ice_column


def overturn_depth(
    t,
    u,
    v,
    n=CFG.n_overturn,
    prob_pct=CFG.overturn_prob_pct,
    c=CFG.crater_proximity,
    h=CFG.depth_overturn,
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
    lam = overturn_lambda(n, prob_pct)
    b = 1 / (v + 2)  # eq 12, Costello 2020
    p1 = (v + 2) / (v * u)
    p2 = 4 * lam / (np.pi * c ** 2)
    a = abs(h * (p1 * p2) ** b)  # eq 11, Costello 2020
    overturn_depth = a * t ** (-b)  # eq 10, Costello 2020
    return overturn_depth


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


def overturn_u(
    a,
    b,
    regime,
    rho_t=CFG.target_density,
    rho_i=CFG.impactor_density_avg,
    kr=CFG.target_kr,
    k1=CFG.target_k1,
    k2=CFG.target_k2,
    mu=CFG.target_mu,
    y=CFG.target_yield_str,
    vf=CFG.impact_speed,
    g=CFG.grav_moon,
    theta_i=CFG.impact_angle,
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


def total_overturn_depth(
    time_arr=CFG._time_arr,
    n_overturns=CFG.n_overturn,
    prob_pct=CFG.overturn_prob_pct,
    regimes=CFG.overturn_regimes,
    sfd_ab=CFG.overturn_ab,
    ts=CFG.timestep,
    vfs=CFG.impact_speeds,
    mode="cannon",
    dtype=CFG.dtype,
):
    """Return array of overturn depth [m] as a function of time."""
    if mode == "cannon":
        return 0.1 * np.ones(len(time_arr), dtype=dtype)
    overturn_depths = []
    for r in regimes:
        a, b = sfd_ab[r]
        vf = vfs[r]
        a_scaled = a * impact_flux(time_arr) / impact_flux(0)
        u = overturn_u(a_scaled, b, "strength", vf=vf)
        overturn = overturn_depth(ts, u, b, n_overturns, prob_pct)
        overturn_depths.append(overturn)
    overturn_total = np.sum(overturn_depths, axis=0)
    return overturn_total


@lru_cache(1)
def read_lambda_table(costello_csv=CFG.costello_csv_in):
    """Read lambda table (Table 1, Costello et al. 2018)."""
    df = pd.read_csv(costello_csv)
    return df


def overturn_lambda(n, prob_pct=CFG.overturn_prob_pct):
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
    df = read_lambda_table()

    # Interpolate nearest value in df[prob_pct] from input n
    lam = np.interp(n, df.n, df[prob_pct])
    return lam


def sublimate_ice_column(ice_column, sublimation_rate):
    """
    Return ice column with thickness of ice lost from top according to
    sublimation rate
    """
    # TODO
    return ice_column


def total_impact_ice(age, regimes=CFG.impact_regimes, rng=None):
    """Return total impact ice from regimes and sfd_slopes (Cannon 2020)."""
    total_ice = 0  # [kg]
    for r in regimes:
        if r == "a":
            # Micrometeorites
            total_ice += ice_micrometeorites(age)
        elif r == "b":
            # Small impactors
            impactor_diams, impactors = get_impactor_pop(age, r)
            total_ice += ice_small_impactors(impactor_diams, impactors)
        elif r == "c":
            # Small simple craters (continuous)
            crater_diams, craters = get_crater_pop(age, r, rng=rng)
            total_ice += ice_small_craters(crater_diams, craters, r)
        else:
            # Large simple & complex craters (stochastic)
            crater_diams = get_crater_pop(age, r, rng=rng)
            crater_diams = get_random_hydrated_craters(crater_diams, rng=rng)
            impactor_speeds = get_random_impactor_speeds(
                len(crater_diams), rng=rng
            )
            total_ice += ice_large_craters(crater_diams, impactor_speeds, r)
    return total_ice


def ice_micrometeorites(
    age=0,
    timestep=CFG.timestep,
    mm_mass_rate=CFG.mm_mass_rate,
    hyd_wt_pct=CFG.hydrated_wt_pct,
    mass_retained=CFG.impactor_mass_retained,
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
    scaling = hyd_wt_pct * impact_flux(age) / impact_flux(0)
    micrometeorite_ice = timestep * scaling * mm_mass_rate * mass_retained
    # TODO: Why don't we account for 36% CC and 2/3 of CC hydrated (like regime B, C)
    # TODO: improve micrometeorite flux?
    return micrometeorite_ice


def ice_small_impactors(diams, num_per_bin, density=CFG.impactor_density):
    """
    Return ice mass [kg] from small impactors (Regime B, Cannon 2020) given
    impactor diams, num_per_bin, and density.
    """
    impactor_masses = diam2vol(diams) * density  # [kg]
    total_impactor_mass = np.sum(impactor_masses * num_per_bin)
    total_impactor_water = impactor_mass2water(total_impactor_mass)

    return total_impactor_water


def ice_small_craters(
    crater_diams,
    ncraters,
    regime,
    v=CFG.impact_speed,
    impactor_density=CFG.impactor_density,
):
    """
    Return ice from simple craters, steep branch (Regime C, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams, v, regime)  # [m]
    impactor_masses = diam2vol(impactor_diams) * impactor_density  # [kg]
    total_impactor_mass = np.sum(impactor_masses * ncraters)
    total_impactor_water = impactor_mass2water(total_impactor_mass)
    return total_impactor_water


def get_random_hydrated_craters(
    crater_diams,
    ctype_frac=CFG.ctype_frac,
    ctype_hyd=CFG.ctype_hydrated,
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
    mean_speed=CFG.impact_speed,
    sd_speed=CFG.impact_sd,
    esc_vel=CFG.escape_vel,
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


def ice_large_craters(
    crater_diams,
    impactor_speeds,
    regime,
    impactor_density=CFG.impactor_density,
    hyd_wt_pct=CFG.hydrated_wt_pct,
):
    """
    Return ice from simple/complex craters, shallow branch of sfd
    (Regime D-E, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams, impactor_speeds, regime)
    impactor_masses = diam2vol(impactor_diams) * impactor_density  # [kg]

    # Find ice mass assuming hydration wt% and retention based on speed
    ice_retention = ice_retention_factor(impactor_speeds)
    ice_masses = impactor_masses * hyd_wt_pct * ice_retention
    return np.sum(ice_masses)


def ice_retention_factor(speeds, dtype=CFG.dtype):
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
    ctype_frac=CFG.ctype_frac,
    ctype_hyd=CFG.ctype_hydrated,
    hyd_wt_pct=CFG.hydrated_wt_pct,
    mass_retained=CFG.impactor_mass_retained,
):
    """
    Return water [kg] from impactor mass [kg] using assumptions of Cannon 2020:
        - 36% of impactors are C-type (Jedicke et al., 2018)
        - 2/3 of C-types are hydrated (Rivkin, 2012)
        - Hydrated impactors are 10% water by mass (Cannon et al., 2020)
        - 16% of asteroid mass retained on impact (Ong et al. 2011)
    """
    return ctype_frac * ctype_hyd * hyd_wt_pct * impactor_mass * mass_retained


def get_impactor_pop(
    age, regime, sfd_slopes=CFG.sfd_slopes, timestep=CFG.timestep
):
    """
    Return population of impactors and number in regime B.

    Use constants and eqn. 3 from Brown et al. (2002) to compute N craters.
    """
    impactor_diams = get_diam_array(regime)
    n_impactors = get_impactors_brown(
        impactor_diams[0], impactor_diams[-1], timestep
    )

    # Scale for timestep, impact flux and size-frequency dist
    flux_scaling = impact_flux(age) / impact_flux(0)
    sfd_prob = get_sfd_prob(regime)
    n_impactors *= flux_scaling * sfd_prob
    return impactor_diams, n_impactors


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


@lru_cache(4)
def get_sfd_prob(regime, sfd_slopes=CFG.sfd_slopes):
    """Return size-frequency distribution probability given diams, sfd slope."""
    diams = get_diam_array(regime)
    sfd = diams ** sfd_slopes[regime]
    return sfd / np.sum(sfd)


@lru_cache(1)
def get_impactors_brown(
    mindiam, maxdiam, timestep=CFG.timestep, c0=1.568, d0=2.7
):
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
    age, regime, ts=CFG.timestep, sa_moon=CFG.sa_moon, rng=None
):
    """
    Return population of crater diameters and number (regimes C - E).

    Weight small simple craters by size-frequency distribution.

    Randomly resample large simple & complex crater diameters.
    """
    crater_diams = get_diam_array(regime)
    sfd_prob = get_sfd_prob(regime)
    n_craters = neukum(crater_diams[0]) - neukum(crater_diams[-1])
    # Scale for timestep, surface area and impact flux
    n_craters *= ts * sa_moon * impact_flux(age) / impact_flux(0)
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


# @lru_cache(4)
def impact_flux(time):
    """Return impact flux at time [yrs] (Derivative of eqn. 1, Ivanov 2008)."""
    time = time * 1e-9  # [yrs -> Ga]
    flux = 6.93 * 5.44e-14 * (np.exp(6.93 * time)) + 8.38e-4  # [n/Ga]
    return flux * 1e-9  # [Ga^-1 -> yrs^-1]


@lru_cache(6)
def neukum(diam, a_values=CFG.ivanov2000):
    """
    Return number of craters per m^2 per yr at diam [m] (eqn. 2, Neukum 2001).

    Eqn 2 expects diam [km], returns N [km^-2 Ga^-1].

    """
    diam = diam * 1e-3  # [m] -> [km]
    j = np.arange(len(a_values))
    ncraters = 10 ** np.sum(a_values * np.log10(diam) ** j)  # [km^-2 Ga^-1]
    return ncraters * 1e-6 * 1e-9  # [km^-2 Ga^-1] -> [m^-2 yr^-1]


@lru_cache(4)
def get_diam_array(regime, diam_range=CFG.diam_range, dtype=CFG.dtype):
    """Return array of diameters based on diameters in diam_range."""
    dmin, dmax, step = diam_range[regime]
    n = int((dmax - dmin) / step)
    return np.linspace(dmin, dmax, n + 1, dtype=dtype)


# Crater scaling laws
def diam2len(diams, speeds=None, regime="c"):
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
    t_diams = final2transient(diams)
    if regime == "c":
        impactor_length = diam2len_prieur(tuple(t_diams), speeds)
    elif regime == "d":
        impactor_length = diam2len_collins(t_diams, speeds)
    elif regime == "e":
        impactor_length = diam2len_johnson(diams)
    else:
        raise ValueError(f"Invalid regime {regime} in diam2len")
    return impactor_length


def final2transient(
    diams, g=CFG.grav_moon, ds2c=CFG.simple2complex, gamma=1.25, eta=0.13
):
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


def simple2complex_diam(
    gravity,
    density=CFG.bulk_density,
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
    c2pr_moon=CFG.complex2peakring,
    g_moon=CFG.grav_moon,
    rho_moon=CFG.bulk_density,
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


@lru_cache(1)
def diam2len_prieur(
    t_diam,
    v=CFG.impact_speed,
    rho_i=CFG.impactor_density,
    rho_t=CFG.target_density,
    g=CFG.grav_moon,
    dtype=CFG.dtype,
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
    v=CFG.impact_speed,
    rho_i=CFG.impactor_density,
    rho_t=CFG.target_density,
    g=CFG.grav_moon,
    theta=CFG.impact_angle,
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
    rho_i=CFG.impactor_density,
    rho_t=CFG.bulk_density,
    g=CFG.grav_moon,
    v=CFG.impact_speed,
    theta=CFG.impact_angle,
    ds2c=CFG.simple2complex,
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


# Helper functions
def latlon2xy(lat, lon, rp=CFG.rad_moon):
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


def xy2latlon(x, y, rp=CFG.rad_moon):
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


def gc_dist(lon1, lat1, lon2, lat2, rp=CFG.rad_moon):
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


def diam2vol(diameter):
    """Return volume of sphere [m^3] given diameter [m]."""
    return (4 / 3) * np.pi * (diameter / 2) ** 3


def get_rng(rng):
    """Return np.random random number generator from given seed in rng."""
    return np.random.default_rng(rng)


if __name__ == "__main__":
    # Set rng random seed for whole model run
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])  # 1st cmd-line arg is seed
        cfg = config.Cfg(seed=seed)
    rng = get_rng(cfg.seed)
    _ = main(cfg=cfg, rng=rng)
