"""
Main mixing module adapted from Cannon et al. (2020)
Date: 06/22/21
Authors: CJ Tai Udovicic, K Frizzell, K Luchsinger, A Madera, T Paladino

Set params and run this file.
All model results are saved to OUTPATH.

# From Jupyter, in first cell type:
import os
os.chdir('/home/cjtu/projects/essi21/code')
import mixing
ej_cols, ice_cols, ice_meta, run_meta, age_grid, ej_matrix = mixing.main()

# Code Profiling (trace efficiency)
python -m cProfile -o mixing.prof mixing.py
snakeviz mixing.prof
"""
import os
import numpy as np
import pandas as pd

# Metadata
RUN_DATETIME = pd.Timestamp.now().strftime("%Y/%m/%d-%H:%M:%S")
RUN = "ensemble1"
RANDOM_SEED = 1  # Set seed to make reproducible random results

# Paths
if "JPY_PARENT_PID" in os.environ:
    FPATH = os.getcwd() + os.sep
else:
    FPATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
DATAPATH = os.path.abspath(FPATH + "../data/") + os.sep
OUTPATH = DATAPATH + RUN + os.sep

# Files to import (# of COLS must equal # columns in CSV)
CRATER_CSV = DATAPATH + "crater_list.csv"
CRATER_COLS = (
    "cname",
    "lat",
    "lon",
    "diam",
    "age",
    "age_low",
    "age_upp",
    "psr_area",
    "age_ref",
    "priority",
    "notes",
)

# Model grid
GRDXSIZE = 400e3  # [m]
GRDYSIZE = 400e3  # [m]
GRDSTEP = 1e3  # [m / pixel]

TIMESTEP = 10e6  # [yr]
TIMESTART = 4.25e9  # [yr]

# Parameters
MODEL_MODE = "cannon"  # ['cannon', 'updated']
ICE_DENSITY = 934  # [kg / m^3], (Cannon 2020)
COLDTRAP_MAX_TEMP = 120  # [K]
COLDTRAP_AREA = 1.3e4 * 1e6  # [m^2], (Williams 2019, via Text S1, Cannon 2020)
ICE_HOP_EFFICIENCY = 0.054  # 5.4% gets to the S. Pole (Text S1, Cannon 2020)
IMPACTOR_DENSITY = 1300  # [kg m^-3], Cannon 2020
# IMPACTOR_DENSITY = 3000  # [kg m^-3] ordinary chondrite (Melosh scaling)
# IMPACT_SPEED = 17e3  # [m/s] average impact speed (Melosh scaling)
IMPACT_SPEED = 20e3  # [m/s] mean impact speed (Cannon 2020)
IMPACT_SD = 6e3  # [m/s] standard deviation impact speed (Cannon 2020)
ESCAPE_VEL = 2.38e3  # [m/s] lunar escape velocity
IMPACT_ANGLE = 45  # [deg]  average impact velocity
TARGET_DENSITY = 1500  # [kg m^-3] (Cannon 2020)
BULK_DENSITY = 2700  # [kg m^-3] simple to complex (Melosh)
# EJECTA_THICKNESS_EXPONENT = # [-3.5, -3, -2.5] min, avg, max Kring 1995
ICE_EROSION_RATE = 0.1 * (TIMESTEP / 10e6)  # [m], 10 cm / 10 Ma (Cannon 2020)

# Impact ice delivery
MM_MASS_RATE = 1e6  # [kg/yr], lunar micrometeorite flux (Grun et al. 2011)
CTYPE_FRAC = 0.36  # 36% of impactors are C-type (Jedicke et al., 2018)
CTYPE_HYDRATED = 2 / 3  # 2/3 of C-types are hydrated (Rivkin, 2012)
HYDRATED_WT_PCT = 0.1  # impactors are 10 wt% water (Cannon 2020)
IMPACTOR_MASS_RETAINED = 0.165  # Asteroid mass retention (Ong et al., 2011)

# Impact ice module
IMPACT_REGIMES = ("A", "B", "C", "D", "E")
DIAM_RANGE = {
    # Regime: (rad_min, rad_max, step)
    "A": (0, 0.01, None),  # Micrometeorites (<1 mm)
    "B": (0.01, 3, 1e-4),  # Small impactors (1 mm - 3 m)
    "C": (100, 1.5e3, 1),  # Simple craters, steep sfd (100 m - 1.5 km)
    "D": (1.5e3, 15e3, 1e2),  # Simple craters, shallow sfd (1.5 km - 15 km)
    # "E": (15e3, 300e3, 1e3),  # Complex craters, shallow sfd (15 km - 300 km)
    "E": (15e3, 500e3, 1e3),  # Complex craters, shallow sfd (15 km - 300 km)
}

SFD_SLOPES = {
    "B": -3.70,  # Small impactors
    "C": -3.82,  # Simple craters "steep" branch
    "D": -1.80,  # Simple craters "shallow" branch
    "E": -1.80,  # Complex craters "shallow" branch
}


VOLC_MODE = "Head"  # ['Head', 'Needham]

# Head et al. (2020)
VOLC_EARLY = (4e9, 3e9)  # [yrs]
VOLC_LATE = (3e9, 2e9)  # [yrs]
VOLC_EARLY_PCT = 0.75  # 75%
VOLC_LATE_PCT = 0.25  # 25%
VOLC_TOTAL_VOL = 1e7 * 1e9  # [m^3] basalt
VOLC_H2O_PPM = 10  # [ppm]
VOLC_MAGMA_DENSITY = 3000  # [kg/m^3]

# Needham & Kring (2017)
VOLC_POLE_PCT = 0.1  # 10%
VOLC_SPECIES = "min_H2O"  # volcanic species, must be in VOLC_COLS
VOLC_CSV = DATAPATH + "needham_kring_2017.csv"
VOLC_COLS = (
    "age",
    "tot_vol",
    "sphere_mass",
    "min_CO",
    "max_CO",
    "min_H2O",
    "max_H2O",
    "min_H",
    "max_H",
    "min_S",
    "max_S",
    "min_sum",
    "max_sum",
    "min_psurf",
    "max_psurf",
    "min_atm_loss",
    "max_atm_loss",
)

# Constants
RAD_MOON = 1737e3  # [m], lunar radius
GRAV_MOON = 1.62  # [m s^-2], gravitational acceleration
SA_MOON = 4 * np.pi * RAD_MOON ** 2  # [m^2]
SIMPLE2COMPLEX = 18e3  # [m], lunar s2c transition diameter (Melosh 1989)
COMPLEX2PEAKRING = 1.4e5  # [m], lunar c2pr transition diameter (Melosh 1989)

# Lunar production function a_values (Neukum 2001)
NEUKUM1983 = (
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
IVANOV2000 = (
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

# Names of files to export
SAVE_NPY = False  # npy save is slow. Generate age / ejecta grids as needed
FEJDF = OUTPATH + f"ej_columns_{RUN}.csv"
FICEDF = OUTPATH + f"ice_columns_{RUN}.csv"
FICEMETA = OUTPATH + f"ice_metadata_{RUN}.csv"
FRUNMETA = OUTPATH + f"run_metadata_{RUN}.csv"
FAGEGRD = OUTPATH + f"age_grid_{RUN}.npy"
FEJMATRIX = OUTPATH + f"ejecta_matrix_{RUN}.npy"

# Set options
COLD_TRAP_CRATERS = (
    "Haworth",
    "Shoemaker",
    "Faustini",
    "Shackleton",
    "Amundsen",
    "Sverdrup",
    "Cabeus B",
    "de Gerlache",
    "Idel'son L",
    "Wiechert J",
)

# Make random number generator
def set_seed(seed=RANDOM_SEED):
    """Return numpy random number generator at given seed."""
    return np.random.default_rng(seed=seed)


_RNG = set_seed()

# Make arrays
_GRD_Y, _GRD_X = np.ogrid[
    GRDYSIZE:-GRDYSIZE:-GRDSTEP, -GRDXSIZE:GRDXSIZE:GRDSTEP
]
_TIME_ARR = np.linspace(TIMESTART, TIMESTEP, int(TIMESTART / TIMESTEP))

# Length of arrays
_NY, _NX = _GRD_Y.shape[0], _GRD_X.shape[1]
NT = len(_TIME_ARR)

# Functions
def main(write=True):
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
    df = read_crater_list()
    # df = randomize_crater_ages(df)
    ej_thickness_matrix = get_ejecta_thickness_matrix(df)  # shape: (NY,NX,NT)
    volcanic_ice_matrix = get_volcanic_ice(mode=VOLC_MODE)  # shape: (NT)

    if MODEL_MODE == "cannon":
        ballistic_sed_matrix = sublimation_thickness = None
    else:
        ballistic_sed_matrix = get_ballistic_sed(df)  # shape: (NY,NX,len(df))
        sublimation_thickness = get_sublimation_rate()  # [m]

    # Init ice columns dictionary based on desired COLD_TRAP_CRATERS
    ice_cols = init_ice_columns(df)

    # Main time loop
    c = None
    for t, time in enumerate(_TIME_ARR):
        # Compute ice mass [kg] gained by all processes
        new_ice_mass = 0
        new_ice_mass += volcanic_ice_matrix[t]
        new_ice_mass += total_impact_ice(time)
        new_ice_mass = new_ice_mass * ICE_HOP_EFFICIENCY
        # Convert mass [kg] to thickness [m] assuming ice evenly distributed
        new_ice_thickness = get_ice_thickness(new_ice_mass)
        ice_cols = update_ice_cols(
            t,
            c,
            ice_cols,
            new_ice_thickness,
            sublimation_thickness,
            ej_thickness_matrix,
            ballistic_sed_matrix,
        )

    age_grid = get_age_grid(df)  # shape: (NY, NX) age of youngest impact
    df_outputs = format_outputs(ej_thickness_matrix, ice_cols)
    outputs = [*df_outputs, age_grid, ej_thickness_matrix]
    if write:
        fnames = (FEJDF, FICEDF, FICEMETA, FRUNMETA, FAGEGRD, FEJMATRIX)
        # Numpy outputs take a long time to write - do we need them?
        if SAVE_NPY:
            save_outputs(outputs, fnames)
        else:
            save_outputs(outputs[:-2], fnames[:-2])
    return outputs


def get_age_grid(df, grd_x=_GRD_X, grd_y=_GRD_Y, timestart=TIMESTART):
    """Return final surface age of each grid point after all craters formed."""
    age_grid = np.ones((grd_y.shape[0], grd_x.shape[1])) * timestart
    for _, crater in df.iterrows():
        age_grid = update_age(age_grid, crater, grd_x, grd_y)
    return age_grid


def get_ejecta_thickness_matrix(df, time_arr=_TIME_ARR):
    """
    Return ejecta_matrix of thickness [m] at each time in time_arr.
    """
    # Compute ejecta thicknesses of each crater
    ejecta_dist = get_ejecta_distances(df)
    ejecta_thickness = get_ejecta_thickness(ejecta_dist, df.rad.values)

    # Find indices of crater ages in time_arr
    # Note: searchsorted must be ascending, so do -time_arr (-4.3, 0) Ga
    time_idx = np.searchsorted(-time_arr, -df.age.values)

    # Fill ejecta_thickness_time with values from ejecta_thickness
    nx, ny = ejecta_thickness.shape[:2]
    nt = len(time_arr)
    ejecta_thickness_time = np.zeros((ny, nx, nt))

    for c_idx, t_idx in enumerate(time_idx):
        # Loop is needed to sum ejecta formed at same t_idx
        ejecta_thickness_time[:, :, t_idx] += ejecta_thickness[:, :, c_idx]

    return ejecta_thickness_time


def format_outputs(ej_matrix, ice_cols):
    """
    Return all formatted model outputs and write to outpath, if specified.
    """
    ej_dict = {"time": _TIME_ARR}
    ice_dict = {"time": _TIME_ARR}
    ice_meta = []
    for cname, (row, col, area, ice_column) in ice_cols.items():
        ej_dict[cname] = ej_matrix[row, col]
        ice_dict[cname] = ice_column
        ice_meta.append([cname, row, col, area])
    # Save all uppercase globals except ones starting with "_"
    gvars = list(globals().items())
    run_meta = {k: v for k, v in gvars if k.isupper() and k[0] != "_"}

    # Convert to DataFrames
    ej_df = pd.DataFrame(ej_dict)
    ice_cols_df = pd.DataFrame(ice_dict)
    ice_meta_df = pd.DataFrame(
        ice_meta, columns=["name", "row", "col", "psr_area"]
    )
    run_meta_df = pd.DataFrame.from_dict(
        run_meta, orient="index"
    ).reset_index()
    return ej_df, ice_cols_df, ice_meta_df, run_meta_df


def save_outputs(outputs, fnames, outpath=OUTPATH):
    """
    Save outputs to files in fnames in directory outpath.
    """
    if not os.path.exists(outpath):
        print(f"Creating new directory: {outpath}.")
        os.makedirs(outpath)
    for out, fname in zip(outputs, fnames):
        fout = os.path.join(outpath, fname)
        if isinstance(out, pd.DataFrame):
            out.to_csv(fout, index=False)
        elif isinstance(out, np.ndarray):
            np.save(fout, out)
        print(f"Saved {fname}")
    print(f"All outputs saved to {outpath}")


def randomize_crater_ages(df, mode=MODEL_MODE):
    """
    Return ages randomized uniformly between agelow, ageupp.
    """
    # TODO: make sure ages are unique to each timestep?
    ages, agelow, ageupp = df[["age", "age_low", "age_upp"]].values.T
    new_ages = np.zeros(len(df))
    for i, (age, low, upp) in enumerate(zip(ages, agelow, ageupp)):
        new_ages[i] = _RNG.uniform(age - low, age + upp)
    if mode == "cannon":
        new_ages = np.round(new_ages, -7)
    df["age"] = new_ages
    df = df.sort_values("age", ascending=False)
    return df


def update_age(age_grid, crater, grd_x=_GRD_X, grd_y=_GRD_Y):
    """
    Return new age grid updating the points interior to crater with its age.
    """
    x, y, rad = crater.x, crater.y, crater.rad
    crater_mask = (np.abs(grd_x - x) < rad) * (np.abs(grd_y - y) < rad)
    age_grid[crater_mask] = crater.age
    return age_grid


def update_ice_cols(
    t,
    c,
    ice_cols,
    new_ice_thickness,
    sublimation_thickness,
    ejecta_matrix,
    ballistic_sed_matrix,
    mode=MODEL_MODE,
):
    """Return ice_cols updated with new ice added and ice eroded dep on mode"""
    # Update all tracked ice columns
    for cname, (row, col, area, ice_col) in ice_cols.items():
        ejecta_column = ejecta_matrix[row, col]

        # Ice gained by column
        ice_col[t] = new_ice_thickness

        # Ice eroded in column
        if mode == "cannon":
            # pass
            ice_col = erode_ice_cannon(ice_col, ejecta_column, t)
        else:
            if c is not None and ejecta_column[t] > 0:
                ice_col = ballistic_sed_ice_column(
                    c, ice_col, ballistic_sed_matrix
                )
            ice_col = garden_ice_column(ice_col, ejecta_column, t)
            ice_col = sublimate_ice_column(ice_col, sublimation_thickness)

        # TODO: Other icy things?
        # thermal pumping?

        # Save ice column
        ice_cols[cname][3] = ice_col
    return ice_cols


def read_crater_list(crater_csv=CRATER_CSV, columns=CRATER_COLS):
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


def read_volcanic_csv(volcanic_csv=VOLC_CSV, col=VOLC_COLS):
    df = pd.read_csv(volcanic_csv, names=col, header=3)
    df["age"] = df["age"] * 1e9  # [Gyr -> yr]
    return df


# Pre-compute grid functions
def get_ejecta_distances(df, grd_x=_GRD_X, grd_y=_GRD_Y):
    """
    Return 3D array shape (len(grd_x), len(grd_y), len(df)) of ejecta distances
    from each crater in df.

    Distances computed with simple dist. Distances within crater rad are NaN.
    """
    ej_dist_all = np.zeros([_NY, _NX, len(df)])
    for i, crater in df.iterrows():
        ej_dist = dist(crater.x, crater.y, grd_x, grd_y)
        ej_dist[ej_dist < crater.rad] = np.nan
        ej_dist_all[:, :, i] = ej_dist
    return ej_dist_all


def get_ejecta_thickness(distance, radius, simple2complex=SIMPLE2COMPLEX):
    """
    Return ejecta thickness as a function of distance given crater radius.

    Complex craters eqn. 1, Kring 1995
    """
    # TODO: account for simple craters
    thickness = 0.14 * radius ** 0.74 * (distance / radius) ** (-3.0)
    thickness = np.nan_to_num(thickness)  # fill nan with 0
    return thickness


def get_volcanic_ice(time_arr=_TIME_ARR, mode="Needham"):
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
    f=VOLC_CSV,
    cols=VOLC_COLS,
    species=VOLC_SPECIES,
    pole_pct=VOLC_POLE_PCT,
    coldtrap_area=COLDTRAP_AREA,
    moon_area=SA_MOON,
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
    early=VOLC_EARLY,
    late=VOLC_LATE,
    early_pct=VOLC_EARLY_PCT,
    late_pct=VOLC_LATE_PCT,
    magma_vol=VOLC_TOTAL_VOL,
    outgassed_h2o=VOLC_H2O_PPM,
    magma_rho=VOLC_MAGMA_DENSITY,
    ice_rho=ICE_DENSITY,
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

    out = np.zeros(len(time_arr))
    out[early_idx] = H2O_early
    out[late_idx] = H2O_late
    return out


def get_ice_thickness(mass, density=ICE_DENSITY, cold_trap_area=COLDTRAP_AREA):
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
    return np.zeros((_NY, _NX, len(df)))


def get_sublimation_rate(timestep=TIMESTEP, temp=COLDTRAP_MAX_TEMP):
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
def init_ice_columns(df, craters=COLD_TRAP_CRATERS, time_arr=_TIME_ARR):
    """
    Return dict of ice columns for cold trap craters in df.

    Currently init at start of time_arr, but possibly:
    - TODO: init after crater formed? Or destroy some pre-existing ice here?
    - TODO: init only after PSR stable (Siegler 2015)?

    Return
    ------
    ice_cols_dict[cname] = [row, col, area, ice_col]
    """
    ice_columns = {}
    for cname in craters:
        if not df.cname.str.contains(cname).any():
            print(f"Cold trap crater {cname} not found. Skipping...")
            continue
        crater = df[df.cname == cname].iloc[0]
        row = int(round(crater.x / GRDSTEP))
        col = int(round(crater.y / GRDSTEP))
        area = crater.psr_area
        ice_col = np.zeros(len(time_arr))
        # ice_col[crater.age < time_arr] = np.nan  # 0 ice before crater formed
        ice_columns[cname] = [row, col, area, ice_col]
    return ice_columns


def ballistic_sed_ice_column(c, ice_column, ballistic_sed_matrix):
    """Return ice column with ballistic sed grid applied"""
    ballistic_sed_grid = ballistic_sed_matrix[:, :, c]
    # TODO: add code from Kristen
    return ice_column


def erode_ice_cannon(
    ice_column, ejecta_column, t, ice_to_erode=0.1, ejecta_shield=0.4
):
    """
    Return eroded ice column using impact gardening
    """
    # BUG in Cannon ds01: erosion base not updated for adjacent ejecta layers
    # Erosion base is most recent time when ejecta column was > ejecta_shield
    erosion_base = np.argmax(ejecta_column[: t + 1] > ejecta_shield)

    # Garden from top of ice column until ice_to_erode amount is removed
    # BUG in Cannon ds01: doesn't account for partial shielding by small ej
    layer = t
    while ice_to_erode > 0 and layer >= 0:
        # if t < erosion_base:
        if layer < erosion_base:
            # End loop if we reach erosion base
            # BUG in Cannon ds01: t should be layer
            # - loop doesn't end if we reach erosion base while eroding
            # - loop only ends here if we started at erosion_base
            break
        ice_in_layer = ice_column[layer]
        if ice_in_layer >= ice_to_erode:
            ice_column[layer] -= ice_to_erode
        else:
            ice_column[layer] = 0
        ice_to_erode -= ice_in_layer
        layer -= 1
    return ice_column


def garden_ice_column(
    ice_column, ejecta_column, time, dt=TIMESTEP, area=COLDTRAP_AREA
):
    """
    Return ice column (area m^2) gardened to some depth in dt amount of time
    given the gardening rates in Costello (2018, 2020) at time yrs in the past.

    Ice is only gardened if the gardening depth exceed the thickness of the
    ejecta column on top of the ice.
    """
    # TODO: Katelyn
    return ice_column


def sublimate_ice_column(ice_column, sublimation_rate):
    """
    Return ice column with thickness of ice lost from top according to
    sublimation rate
    """
    # TODO
    return ice_column


def total_impact_ice(age, regimes=IMPACT_REGIMES):
    """Return total impact ice from regimes and sfd_slopes (Cannon 2020)."""
    total_ice = 0  # [kg]
    for r in regimes:
        if r == "A":
            # Micrometeorites
            total_ice += ice_micrometeorites(age)
        elif r == "B":
            # Small impactors
            impactor_diams, impactors = get_impactor_pop(age, r)
            total_ice += ice_small_impactors(impactor_diams, impactors)
        elif r == "C":
            # Small simple craters (continuous)
            crater_diams, craters = get_crater_pop(age, r)
            total_ice += ice_small_craters(crater_diams, craters, r)
        else:
            # Large simple & complex craters (stochastic)
            crater_diams = get_crater_pop(age, r)
            crater_diams = get_random_hydrated_craters(crater_diams)
            impactor_speeds = get_random_impactor_speeds(len(crater_diams))
            total_ice += ice_large_craters(crater_diams, impactor_speeds, r)
    return total_ice


def ice_micrometeorites(
    age=0,
    timestep=TIMESTEP,
    mm_mass_rate=MM_MASS_RATE,
    hyd_wt_pct=HYDRATED_WT_PCT,
    mass_retained=IMPACTOR_MASS_RETAINED,
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


def ice_small_impactors(diams, num_per_bin, density=IMPACTOR_DENSITY):
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
    v=IMPACT_SPEED,
    impactor_density=IMPACTOR_DENSITY,
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
    crater_diams, ctype_frac=CTYPE_FRAC, ctype_hyd=CTYPE_HYDRATED
):
    """
    Return crater diams of hydrated craters from random distribution.
    """
    # Randomly include only craters formed by hydrated, Ctype asteroids
    rand_arr = _RNG.random(size=len(crater_diams))
    crater_diams = crater_diams[rand_arr < ctype_frac * ctype_hyd]
    return crater_diams


def get_random_impactor_speeds(
    n, mean_speed=IMPACT_SPEED, sd_speed=IMPACT_SD, esc_vel=ESCAPE_VEL
):
    """
    Return n impactor speeds from normal distribution about mean, sd.
    """
    # Randomize impactor speeds with Gaussian around mean, sd
    impactor_speeds = _RNG.normal(mean_speed, sd_speed, n)
    impactor_speeds[impactor_speeds < esc_vel] = esc_vel  # minimum is esc_vel
    return impactor_speeds


def ice_large_craters(
    crater_diams,
    impactor_speeds,
    regime,
    impactor_density=IMPACTOR_DENSITY,
    hyd_wt_pct=HYDRATED_WT_PCT,
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


def ice_retention_factor(speeds):
    """
    Return ice retained in impact, given impactor speeds (Cannon 2020).

    For speeds < 10 km/s, retain 50% (Svetsov & Shuvalov 2015 via Cannon 2020).
    For speeds >= 10 km/s, use eqn ? (Ong et al. 2010 via Cannon 2020)
    """
    # TODO: find/verify retention(speed) eqn in Ong et al. 2010?
    # retention distribution is discontinuous
    speeds = speeds * 1e-3  # [m/s] -> [km/s]
    retained = np.ones(len(speeds)) * 0.5  # nominal 50%
    retained[speeds >= 10] = 36.26 * np.exp(-0.3464 * speeds[speeds >= 10])
    retained[retained < 0] = 0
    return retained


def impactor_mass2water(
    impactor_mass,
    ctype_frac=CTYPE_FRAC,
    ctype_hyd=CTYPE_HYDRATED,
    hyd_wt_pct=HYDRATED_WT_PCT,
    mass_retained=IMPACTOR_MASS_RETAINED,
):
    """
    Return water [kg] from impactor mass [kg] using assumptions of Cannon 2020:
        - 36% of impactors are C-type (Jedicke et al., 2018)
        - 2/3 of C-types are hydrated (Rivkin, 2012)
        - Hydrated impactors are 10% water by mass (Cannon et al., 2020)
        - 16% of asteroid mass retained on impact (Ong et al. 2011)
    """
    return ctype_frac * ctype_hyd * hyd_wt_pct * impactor_mass * mass_retained


def get_impactor_pop(age, regime, sfd_slopes=SFD_SLOPES, timestep=TIMESTEP):
    """
    Return population of impactors and number in regime B.

    Use constants and eqn. 3 from Brown et al. (2002) to compute N craters.
    """
    impactor_diams = get_diam_array(regime)
    n_impactors = get_impactors_brown(
        impactor_diams[0], impactor_diams[-1], timestep
    )

    # Scale for timestep and impact flux
    flux_scaling = impact_flux(age) / impact_flux(0)
    n_impactors *= flux_scaling

    # Scale by size-frequency distribution
    sfd = impactor_diams ** sfd_slopes[regime]
    sfd_prob = sfd / np.sum(sfd)
    impactors = sfd_prob * n_impactors
    return impactor_diams, impactors


def get_impactors_brown(mindiam, maxdiam, timestep=TIMESTEP, c0=1.568, d0=2.7):
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
    age, regime, sfd_slopes=SFD_SLOPES, ts=TIMESTEP, sa_moon=SA_MOON
):
    """
    Return population of crater diameters and number (regimes C - E).

    Weight small simple craters by size-frequency distribution.

    Randomly resample large simple & complex crater diameters.
    """
    crater_diams = get_diam_array(regime)
    n_craters = neukum(crater_diams[0]) - neukum(crater_diams[-1])
    # Scale for timestep, surface area and impact flux
    n_craters *= ts * sa_moon * impact_flux(age) / impact_flux(0)
    sfd = crater_diams ** sfd_slopes[regime]
    sfd_prob = sfd / np.sum(sfd)
    if regime == "C":
        # Steep branch of sfd (simple)
        craters = sfd_prob * n_craters
        return crater_diams, craters

    # Regimes D and E: shallow branch of sfd (simple / complex)
    n_craters = probabalistic_round(n_craters)

    # Resample crater diameters with replacement, weighted by sfd
    crater_diams = _RNG.choice(crater_diams, n_craters, p=sfd_prob)
    return crater_diams


def impact_flux(time):
    """Return impact flux at time [yrs] (Derivative of eqn. 1, Ivanov 2008)."""
    time = time * 1e-9  # [yrs -> Ga]
    flux = 6.93 * 5.44e-14 * (np.exp(6.93 * time)) + 8.38e-4  # [n/Ga]
    return flux * 1e-9  # [Ga^-1 -> yrs^-1]


def neukum(diam, a_values=IVANOV2000):
    """
    Return number of craters per m^2 per yr at diam [m] (eqn. 2, Neukum 2001).

    Eqn 2 expects diam [km], returns N [km^-2 Ga^-1].

    """
    diam = diam * 1e-3  # [m] -> [km]
    j = np.arange(len(a_values))
    ncraters = 10 ** np.sum(a_values * np.log10(diam) ** j)  # [km^-2 Ga^-1]
    return ncraters * 1e-6 * 1e-9  # [km^-2 Ga^-1] -> [m^-2 yr^-1]


def get_diam_array(regime, diam_range=DIAM_RANGE):
    """Return array of diameters based on diameters in diam_range."""
    dmin, dmax, step = diam_range[regime]
    n = int((dmax - dmin) / step)
    return np.linspace(dmin, dmax, n + 1)


# Crater scaling laws
def diam2len(diams, speeds=None, regime="C"):
    """
    Return size of impactors based on diam and sometimes speeds of craters.

    Different crater regimes are scaled via the following scaling laws:
    - regime=='C': (Prieur et al., 2017)
    - regime=='D': (Collins et al., 2005)
    - regime=='E': (Johnson et al., 2016)

    Parameters
    ----------
    diams (arr): Crater diameters [m].
    speeds (arr): Impactor speeds [m/s].
    regime (str): Crater scaling regime ('C', 'D', or 'E').

    Return
    ------
    lengths (arr): Impactor diameters [m].
    """
    t_diams = final2transient(diams)
    if regime == "C":
        impactor_length = diam2len_prieur(t_diams, speeds)
    elif regime == "D":
        impactor_length = diam2len_collins(t_diams, speeds)
    elif regime == "E":
        impactor_length = diam2len_johnson(diams)
    else:
        raise ValueError(f"Invalid regime {regime} in diam2len")
    return impactor_length


def final2transient(diams, g=GRAV_MOON):
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
    # Parameters from Melosh (1989)
    gamma = 1.25
    eta = 0.13

    # Scale simple to complex diameter
    ds2c = simple2complex_diam(g)  # [m]
    s_idx = diams <= ds2c
    c_idx = diams > ds2c

    # Convert final diams to transient diams
    t_diams = np.zeros(len(diams))
    t_diams[s_idx] = diams[s_idx] / gamma
    t_diams[c_idx] = (1 / gamma) * (diams[c_idx] * ds2c ** eta) ** (
        1 / (1 + eta)
    )
    return t_diams


def simple2complex_diam(
    gravity,
    density=BULK_DENSITY,
    s2c_moon=18e3,
    g_moon=1.62,
    rho_moon=2700,
):
    """
    Return simple to complex transition diameter given gravity of body [m s^-2]
    and density of target [kg m^-3] (Melosh 1989).
    """
    return g_moon * rho_moon * s2c_moon / (gravity * density)


def complex2peakring_diam(
    gravity,
    density,
    c2pr_moon=COMPLEX2PEAKRING,
    g_moon=GRAV_MOON,
    rho_moon=BULK_DENSITY,
):
    """
    Return complex to peak ring basin transition diameter given gravity of
    body [m s^-2] and density of target [kg m^-3] (Melosh 1989).
    """
    return g_moon * rho_moon * c2pr_moon / (gravity * density)


def diam2len_prieur(
    t_diam,
    v=IMPACT_SPEED,
    rho_i=IMPACTOR_DENSITY,
    rho_t=TARGET_DENSITY,
    g=GRAV_MOON,
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
    i_lengths = np.linspace(np.min(t_diam) / 100, np.max(t_diam), 10000)
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
    v=IMPACT_SPEED,
    rho_i=IMPACTOR_DENSITY,
    rho_t=TARGET_DENSITY,
    g=GRAV_MOON,
    theta=IMPACT_ANGLE,
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
    rho_i=IMPACTOR_DENSITY,
    rho_t=BULK_DENSITY,
    g=GRAV_MOON,
    v=IMPACT_SPEED,
    theta=IMPACT_ANGLE,
    ds2c=SIMPLE2COMPLEX,
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
def latlon2xy(lat, lon, rp=RAD_MOON):
    """Return (x, y) [rp units] S. Pole stereo coords from (lon, lat) [deg]."""
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = rp * np.cos(lat) * np.sin(lon)
    y = rp * np.cos(lat) * np.cos(lon)
    return x, y


def xy2latlon(x, y, rp=RAD_MOON):
    """Return (lat, lon) [deg] from S. Pole stereo coords (x, y) [rp units]."""
    z = np.sqrt(rp ** 2 - x ** 2 - y ** 2)
    lat = -np.arcsin(z / rp)
    lon = np.arctan2(x, y)
    return np.rad2deg(lat), np.rad2deg(lon)


def gc_dist(lon1, lat1, lon2, lat2, rp=RAD_MOON):
    """Return great circ dist (lon1, lat1) - (lon2, lat2) [deg] in rp units."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return rp * 2 * np.arcsin(np.sqrt(a))


def dist(x1, y1, x2, y2):
    """Return simple distance between coordinates (x1, y1) and (x2, y2)."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def probabalistic_round(x):
    """
    Randomly round positive float x up or down based on its distance to x + 1.

    E.g. 6.1 will round down ~90% of the time and round up ~10% of the time
    such that over many trials, the expected value is 6.1.
    """
    return int(x + _RNG.random())


def diam2vol(diameter):
    """Return volume of sphere given diameter."""
    return (4 / 3) * np.pi * (diameter / 2) ** 3


if __name__ == "__main__":
    _ = main()
