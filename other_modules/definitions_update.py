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


def get_crater_distances(df, symmetric=True):
    """
    Return 2D array of great circle dist between all craters in df.

    Mandatory
        - df : Read in crater_list.csv as a DataFrame with defined columns
        - df : Required columns defined 'lat' and 'lon'
        - See 'read_crater_list' function

    Parameters
    ----------
    df (DataFrame): Crater DataFrame read and updated from crater_csv
    TODO: Symmetric : 

    Returns
    -------
    out (array): 2D array of great circle distances between all craters defined
                 in df(DataFrame)

    All zero crater distances are reported as nan
    """
    out = np.zeros((len(df), len(df)), dtype=_DTYPE)
    for i in range(len(df)):
        for j in range(i):
            d = gc_dist(*df.iloc[i][['lon', 'lat']], *df.iloc[j][['lon', 'lat']])
            out[i, j] = d
    if symmetric:
        out += out.T
    out[out <= 0] = np.nan
    return out

def latlon2xy(lat, lon, rp=RAD_MOON):
    """
    Return (x, y) [rp units] S. Pole stereo coords from (lat, lon) [deg].
    
    Supplement
    ----------
        - rp units: [num], derived from given radius of the moon in meters (see below)

    Parameters
    ----------
    lat [deg]: Stereo latitude coordinates 
    lon [deg]: Stereo longitude coordinates
    rp [num]: Radius of the Moon, 1737e3 in meters

    Return
    -------
    x [rp units]: South Pole stereo coordinates, converted from lat [deg]
    y [rp units]: South Pole stereo coodinates, converted from lon [deg] 
    """

    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = rp * np.cos(lat) * np.sin(lon)
    y = rp * np.cos(lat) * np.cos(lon)
    return x, y

def xy2latlon(x, y, rp=RAD_MOON):
    """
    Return (lat, lon) [deg] from S. Pole stereo coords (x, y) [rp units].
    
    Supplement
    ----------
    rp units: [num], derived from given radius of the moon in meters (see below)

    Parameters
    ----------
    x [rp units]:  (x,0) coordinate, to be plotted on a grid
    y [rp units]:  (0,y) coordinate, to be plotted on a grid
    rp [num]: Radius of the Moon, 1737e3 in meters

    Return
    -------
    np.rad2deg(lat) [deg] : South Pole stereo coodinate in degrees
    np.rad2deg(lon) [deg] :  South Pole stereo coordinate in degrees
    """

    z = np.sqrt(rp ** 2 - x ** 2 - y ** 2)
    lat = -np.arcsin(z / rp)
    lon = np.arctan2(x, y)
    return np.rad2deg(lat), np.rad2deg(lon)

def gc_dist(lon1, lat1, lon2, lat2, rp=RAD_MOON):
    """
    Return great circ dist (lon1, lat1) - (lon2, lat2) [deg] in rp units.
    
    Parameters
    ----------
    lon1 [deg]: Longitude [coord] of crater of interest, 1
    lat1 [deg]: Latitude [coord] of crater of interet, 1
    lon2 [deg]: Longitude [coord] of crater of interest, 2
    lat2 [deg]: Latitude [coord] of crater of interest, 2
    rp [num]: Radius of the Moon, 1737e3 in meters

    Return
    ------
    rp * c [num] : Distance of craters of interest (1 and 2), in meters [m]
    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    sin2_dlon = np.sin((lon2 - lon1)/2) ** 2
    sin2_dlat = np.sin((lat2 - lat1)/2) ** 2
    a = sin2_dlat + np.cos(lat1) * np.cos(lat2) * sin2_dlon
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return rp * c

#def dist(x1, y1, x2, y2):
    """
    Return simple distance between coordinates (x1, y1) and (x2, y2).
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def probabalistic_round(x):
    """
    Randomly round positive float x up or down based on its distance to x + 1.

    E.g. 6.1 will round down ~90% of the time and round up ~10% of the time
    such that over many trials, the expected value is 6.1.

    Parameters
    ----------
    x [float] : TODO

    Return
    ------
    TODO:

    """
    return int(x + _RNG.random())

def diam2vol(diameter):
    """
    Return volume of sphere given diameter.
    
    Parameters
    ----------
    Diameter [num]: 2D dimension of a circle

    Return
    ------
    Volume [num]: 3D dimension of a sphere, calculated from given circle diameter
    """
    return (4 / 3) * np.pi * (diameter / 2) ** 3

def complex2peakring_diam(
    gravity,
    density,
    c2pr_moon=COMPLEX2PEAKRING,
    g_moon=GRAV_MOON,
    rho_moon=BULK_DENSITY):
    """
    Return complex to peak ring basin transition diameter given gravity of
    body [m s^-2] and density of target [kg m^-3] (Melosh 1989).

    Parameters
    ----------
    gravity [num] : Gravity of a body [m s^-2] TODO: What body?
    density [num] : Density of the target material [kg m^-3]
    c2pr_moon [num] : Lunar complex crater to peak ring transition diamter, 1.4e5 [m], see source
    g_moon [num] : Gravity of the Moon, 1.62 [m s^-2]
    rho_moon [num] : Bulk density of the Moon, 2700 [kg m^-3]
    
    Return
    ------
    complex2peakring_diam [num] : Transition diameter [m] for a complex crater to a peak ring basin
    """
    return g_moon * rho_moon * c2pr_moon / (gravity * density)

def simple2complex_diam(
    gravity,
    density=BULK_DENSITY,
    s2c_moon=SIMPLE2COMPLEX,
    g_moon=GRAV_MOON,
    rho_moon=BULK_DENSITY,
):
    """
    Return simple to complex transition diameter given gravity of body [m s^-2]
    and density of target [kg m^-3] (Melosh 1989).

    Parameters
    ----------
    gravity [num] : Gravity of a body [m s^-2] TODO: What body?
    density [num] : Density of the target material [kg m^-3]
    s2c_moon [num] : Lunar simple to complex transition diamter, 18e3 [m], see source
    g_moon [num] : Gravity of the Moon, 1.62 [m s^-2]
    rho_moon [num] : Bulk density of the Moon, 2700 [kg m^-3]

    Return
    ------
    simple2complex_diam [num] : Transition diamter [m] for a simple crater to complex crater
    """
    return g_moon * rho_moon * s2c_moon / (gravity * density)

def get_age_grid(df, grd_x=_GRD_X, grd_y=_GRD_Y, timestart=TIMESTART):
    """
    Return final surface age of each grid point after all craters formed.
    
    Parameters
    ----------
    df (DataFrame): Crater DataFrame read and updated from crater_csv
    grid_x :
    grid_y :
    timestart :

    Return
    ------

    
    """
    ny, nx = grd_y.shape[0], grd_x.shape[1]
    age_grid = timestart * np.ones((ny, nx), dtype=_DTYPE) 
    for _, crater in df.iterrows():
        age_grid = update_age(age_grid, crater, grd_x, grd_y)
    return age_grid

 def get_ejecta_thickness_matrix(df, time_arr=_TIME_ARR, ts=TIMESTEP):
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
    rounded_time = np.around(time_arr, -6)
    rounded_age = np.around(df.age.values, -6)
    time_idx = np.searchsorted(-time_arr, -df.age.values)

    # Fill ejecta thickness vs time matrix (rows: time, cols:craters)
    ej_thick_time = np.zeros((len(time_arr), len(time_idx)), dtype=_DTYPE)
    for i, t_idx in enumerate(time_idx):
        # Sum here in case more than one crater formed at t_idx
        ej_thick_time[t_idx, :] += ej_thick[:, i]    
    return ej_thick_time

def grid_interp(x, y, grdvalues, grd_x=_GRD_X, grd_y=_GRD_Y):
    """Return ejecta thickness at (x, y) in ejecta_thickness 2D grid."""
    ix, iy = get_grd_ind(x, y, grd_x, grd_y)
    gx, gy = grd_x.flatten(), grd_y.flatten()

    dy = (y - gy[iy]) / (gy[iy + 1] - gy[iy])
    interp_y = (1 - dy) * grdvalues[iy] + dy * grdvalues[iy + 1]

    dx = (x - gx[ix]) / (gx[ix + 1] - gx[ix])
    interp = (1 - dx) * interp_y[ix] + dx * interp_y[ix + 1]

    return interp


def format_outputs(strat_cols, time_arr=_TIME_ARR):
    """
    Return all formatted model outputs and write to outpath, if specified.
    """
    ej_dict = {"time": time_arr}
    ice_dict = {"time": time_arr}
    for cname, (ice_col, ej_col) in strat_cols.items():
        ej_dict[cname] = ej_col
        ice_dict[cname] = ice_col

    # Save all uppercase globals except ones starting with "_"
    gvars = list(globals().items())
    run_meta = {k: v for k, v in gvars if k.isupper() and k[0] != "_"}

    # Convert to DataFrames
    ej_cols_df = pd.DataFrame(ej_dict)
    ice_cols_df = pd.DataFrame(ice_dict)
    run_meta_df = pd.DataFrame.from_dict(
        run_meta, orient="index"
    ).reset_index()
    return ej_cols_df, ice_cols_df, run_meta_df

def save_outputs(outputs, fnames, outpath=OUTPATH, verbose=_VERBOSE):
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
        if verbose:
            print(f"Saved {fname}")
    print(f"All outputs saved to {outpath}")

def round_to_ts(values, timestep):
    """Return values rounded to nearest timestep."""
    return np.around(values / timestep) * timestep

def randomize_crater_ages(df, timestep=TIMESTEP):
    """
    Return ages randomized uniformly between agelow, ageupp.
    """
    # TODO: make sure ages are unique to each timestep?
    ages, agelow, ageupp = df[["age", "age_low", "age_upp"]].values.T
    new_ages = np.zeros(len(df), dtype=_DTYPE)
    for i, (age, low, upp) in enumerate(zip(ages, agelow, ageupp)):
        new_ages[i] = round_to_ts(_RNG.uniform(age - low, age + upp), timestep)
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
    strat_cols,
    new_ice_thickness,
    # sublimation_thickness,
    # ballistic_sed_matrix,
    mode=MODEL_MODE,
):
    """
    Update ice_cols new ice added and ice eroded.
    """
    # Update all tracked ice columns
    for cname, (ice_col, ej_col) in strat_cols.items():
        # Ice gained by column
        ice_col[t] = new_ice_thickness

        # Ice eroded in column
        if mode == "cannon":
            ice_col = erode_ice_cannon(ice_col, ej_col, t)
        # else:
        #     if c is not None and ej_col[t] > 0:
        #         ice_col = ballistic_sed_ice_column(
        #             c, ice_col, ballistic_sed_matrix
        #         )
        #     ice_col = garden_ice_column(ice_col, ej_col, t)
        #     ice_col = sublimate_ice_column(ice_col, sublimation_thickness)

        # TODO: Other icy things?
        # thermal pumping?

        # Save ice column back to strat_cols dict
        strat_cols[cname][0] = ice_col













def get_diam_array(regime, diam_range=DIAM_RANGE):
    """Return array of diameters based on diameters in diam_range."""
    dmin, dmax, step = diam_range[regime]
    n = int((dmax - dmin) / step)
    return np.linspace(dmin, dmax, n + 1, dtype=_DTYPE)


def neukum(diam, a_values=IVANOV2000):
    """
    Return number of craters per m^2 per yr at diam [m] (eqn. 2, Neukum 2001).

    Eqn 2 expects diam [km], returns N [km^-2 Ga^-1].

    """
    diam = diam * 1e-3  # [m] -> [km]
    j = np.arange(len(a_values))
    ncraters = 10 ** np.sum(a_values * np.log10(diam) ** j)  # [km^-2 Ga^-1]
    return ncraters * 1e-6 * 1e-9  # [km^-2 Ga^-1] -> [m^-2 yr^-1]

def impact_flux(time):
    """Return impact flux at time [yrs] (Derivative of eqn. 1, Ivanov 2008)."""
    time = time * 1e-9  # [yrs -> Ga]
    flux = 6.93 * 5.44e-14 * (np.exp(6.93 * time)) + 8.38e-4  # [n/Ga]
    return flux * 1e-9  # [Ga^-1 -> yrs^-1]

def get_crater_pop(age, regime, ts=TIMESTEP, sa_moon=SA_MOON):
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
    if regime == "C":
        # Steep branch of sfd (simple)
        n_craters *= sfd_prob
        return crater_diams, n_craters

    # Regimes D and E: shallow branch of sfd (simple / complex)
    n_craters = probabalistic_round(n_craters)

    # Resample crater diameters with replacement, weighted by sfd
    crater_diams = _RNG.choice(crater_diams, n_craters, p=sfd_prob)
    return crater_diams

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

def get_sfd_prob(regime, sfd_slopes=SFD_SLOPES):
    """Return size-frequency distribution probability given diams, sfd slope."""
    diams = get_diam_array(regime)
    sfd = diams ** sfd_slopes[regime]
    return sfd / np.sum(sfd)

def get_impactor_pop(age, regime, sfd_slopes=SFD_SLOPES, timestep=TIMESTEP):
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

def ice_retention_factor(speeds):
    """
    Return ice retained in impact, given impactor speeds (Cannon 2020).

    For speeds < 10 km/s, retain 50% (Svetsov & Shuvalov 2015 via Cannon 2020).
    For speeds >= 10 km/s, use eqn ? (Ong et al. 2010 via Cannon 2020)
    """
    # TODO: find/verify retention(speed) eqn in Ong et al. 2010?
    # BUG? retention distribution is discontinuous
    speeds = speeds * 1e-3  # [m/s] -> [km/s]
    retained = np.ones(len(speeds), dtype=_DTYPE) * 0.5  # nominal 50%
    retained[speeds >= 10] = 36.26 * np.exp(-0.3464 * speeds[speeds >= 10])
    retained[retained < 0] = 0
    return retained

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

