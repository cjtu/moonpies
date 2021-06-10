"""
Main mixing module adapted from Cannon et al. (2020)
Date: 06/10/21
Authors: CJ Tai Udovicic, K Frizzell, K Luchsinger, A Madera, T Paladino

Set params and run this file. 
All model results are saved to OUTPATH.


# From Jupyter, in first cell type:
import os
os.chdir('/home/user/path/to/code/') 
import mixing
ice_cols_df, age_grid, ejecta_matrix = mixing.main()
"""
import os
import numpy as np
import pandas as pd

# Metadata
RUN_DATETIME = pd.Timestamp.now().strftime('%Y/%m/%d-%H:%M:%S')
RUN = 'week2'
RANDOM_SEED = 0  # Set seed to make reproducible random results

# Paths
if 'JPY_PARENT_PID' in os.environ:
    FPATH = os.getcwd() + os.sep
else:
    FPATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
DATAPATH = os.path.abspath(FPATH + '../data/') + os.sep
OUTPATH = DATAPATH + RUN + os.sep
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

# Files to import (# of COLS must equal # columns in CSV)
CRATER_CSV = DATAPATH + 'crater_list.csv'
VOLCANIC_CSV = DATAPATH + 'needham_kring_2017.csv'
CRATER_COLS = ('cname', 'lat', 'lon', 'diam', 'age', 'age_low', 'age_upp', 
               'psr_area', 'age_ref', 'priority', 'notes')
VOLCANIC_COLS = ('age', 'tot_vol', 'sphere_mass', 'min_CO', 'max_CO', 
                'min_H2O', 'max_H2O', 'min_H', 'max_H', 'min_S', 'max_S', 
                'min_sum', 'max_sum', 'min_psurf', 'max_psurf', 'min_atm_loss', 
                'max_atm_loss')

# Parameters
ICE_DENSITY = 934  # [kg / m^3] - make sure this is a good number
COLDTRAP_AREA = 13e3  # [m^2]
COLDTRAP_MAX_TEMP = 120  # [K]
ICE_HOP_EFFICIENCY = 0.054  # Cannon 2020
IMPACTOR_DENSITY = 1300  # [kg / m^3], Cannon 2020
# IMPACTOR_DENSITY = 3000  # [kg / m^3] ordinary chondrite density
IMPACT_SPEED = 20000  # [m/s] average impact speed
IMPACT_ANGLE = 45  # [deg]  average impact velocity
TARGET_DENSITY = 1500  # [kg / m^3]
# EJECTA_THICKNESS_EXPONENT = # [-3.5, -3, -2.5] min, avg, max Kring 1995
VOLCANIC_SPEC = 'min_H20'  # volcanic species, must be in VOLCANIC_COLS

# Constants
R_MOON = 1737e3  # [m], radius
G_MOON = 1.62  # [m s^-2], gravitational acceleration
SIMPLE2COMPLEX = 15e3  # [m]

# Files to export
AGE_GRID_NPY = OUTPATH + f'age_grid_{RUN}.csv'
EJECTA_MATRIX_NPY = OUTPATH + f'ejecta_matrix_{RUN}.csv'
ICE_COLS_CSV = OUTPATH + f'ice_columns_{RUN}.csv'
ICE_META_CSV = OUTPATH + f'ice_metadata_{RUN}.csv'
RUN_META_CSV = OUTPATH + f'run_metadata_{RUN}.csv'

# Set options
COLD_TRAP_CRATERS = ['Haworth', 'Shoemaker', 'Faustini', 'Shackleton',
                     'Amundsen', 'Sverdrup', 'Cabeus B', 'de Gerlache', 
                     "Idel'son L", 'Wiechert J']

GRDXSIZE = 400e3  # [m]
GRDYSIZE = 400e3  # [m]
GRDSTEP = 1e3  # [m / pixel]

TIMESTEP = 10e6  # [yr]
TIMESTART = 4.3e9  # [yr]

# Make random number generator
_RNG = np.random.default_rng(seed=RANDOM_SEED)

# Make arrays
_GRD_Y, _GRD_X = np.ogrid[GRDYSIZE:-GRDYSIZE:-GRDSTEP, 
                          -GRDXSIZE:GRDXSIZE:GRDSTEP]
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
    df = randomize_crater_ages(df)
    ej_dist = get_ejecta_distances(df)
    ej_thickness = get_ejecta_thickness(ej_dist, df.rad.values)  # shape: (NY,NX,len(df))
    volcanic_ice_matrix = get_volcanic_ice()  # shape: 1D len(time_arr)
    ballistic_sed_matrix = get_ballistic_sed(df)  # shape: (NY,NX,len(df))
    sublimation_thickness = get_sublimation_rate()  # [m]

    # Init arrays
    ice_cols = init_ice_columns(df)  # 1D ice thickness column
    age_grid = np.ones((_NY, _NX)) * TIMESTART  # age of youngest crater on grid [yr]
    ejecta_matrix = np.zeros((_NY, _NX, NT))  # ejecta thickness, grid over time [m]

    # Main time loop
    for t, time in enumerate(_TIME_ARR):
        crater = c = None
        # If crater formed at this timestep, update ejecta_matrix and age_grid
        make_crater = df[df.age.between(time - TIMESTEP/2, time + TIMESTEP/2)]
        if len(make_crater) == 1:
            c = int(make_crater.index[0])
            crater = df.iloc[c]
            # Ejecta map - add ejecta thickness to this timestep
            # Age map - update age of surface interior to this crater
            ejecta_matrix[:, :, t] += ej_thickness[:, :, c]
            age_grid = update_age(age_grid, crater.x, crater.y, crater.rad, crater.age)
        elif len(make_crater) > 1:
            # TODO: throw error here - don't allow simultaneous craters
            pass
        
        # Compute ice gained by all cold traps
        #  - add mass of ice from volcanism [kg]
        #  - add mass of ice from impacts [kg]
        #  - convert to thickness [m] (assumes ice is evenly distributed)
        new_ice_mass = 0
        new_ice_mass += volcanic_ice_matrix[:, :, t] * ICE_HOP_EFFICIENCY
        new_ice_mass += total_impact_ice(time) * ICE_HOP_EFFICIENCY
        new_ice_thickness = get_ice_thickness(new_ice_mass)

        # Update all tracked ice columns
        for cname in ice_cols:
            row, col, area, ice_column = ice_cols[cname]
            if np.isnan(ice_column[t]):
                # If crater doesn't exist yet, skip.
                continue
            
            # Ice gained by column
            ice_column[t] = new_ice_thickness

            # Ice lost in column
            ejecta_column = ejecta_matrix[row, col]
            if c is not None:
                ice_column = ballistic_sed_ice_column(ice_column, ballistic_sed_matrix[:, :, c])
            ice_column = garden_ice_column(ice_column, ejecta_column, time)
            ice_column = sublimate_ice_column(ice_column, sublimation_thickness)

            # Other icy things
            # thermal pumping?

    # Format outputs
    ice_cols_df = pd.DataFrame({k:v[3] for k, v in ice_cols.items()})
    ice_cols_df.insert(0, 'time', _TIME_ARR)
    ice_meta = [[k, *v[:3]] for k, v in ice_cols.items()]
    ice_meta_df = pd.DataFrame(ice_meta, columns=['name', 'row', 'col', 'psr_area'])
    run_meta_df = pd.DataFrame.from_dict({
        k:globals()[k] for k in list(globals()) if k.isupper()}, 
        orient='index').reset_index()
    # Save outputs
    if write:
        ice_cols_df.to_csv(ICE_COLS_CSV, index=False)
        ice_meta_df.to_csv(ICE_META_CSV, index=False)
        run_meta_df.to_csv(RUN_META_CSV, index=False)
        np.save(AGE_GRID_NPY, age_grid)
        np.save(EJECTA_MATRIX_NPY, ejecta_matrix)
        print(f'Saved results to {OUTPATH}')
    return ice_cols_df, age_grid, ejecta_matrix


def randomize_crater_ages(df, timestep=TIMESTEP):
    """
    Return df with age column unique and randomized within age_low, age_upp
    at timestep precision.
    """
    # TODO: randomize crater ages, make sure all ages are unique
    return df


def update_age(age_grid, x, y, radius, age, grd_x=_GRD_X, grd_y=_GRD_Y):
    """
    Return new age grid updating the points interior to crater with its age.
    """
    crater_mask = ((np.abs(grd_x - x) < radius) * 
                   (np.abs(grd_y - y) < radius))
    
    age_grid[crater_mask] = age
    return age_grid


def read_crater_list(crater_csv=CRATER_CSV, columns=CRATER_COLS):
    """
    Return dataframe of craters from path to crater_csv with columns names.

    Mandatory columns and naming convention:
        - 'lat': Latiude [deg]
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
    df['diam'] = df['diam'] * 1000  # [km -> m]
    df['rad'] = df['diam'] / 2
    df['age'] = df['age'] * 1e9  # [Gyr -> yr]
    df['age_low'] = df['age_low'] * 1e9  # [Gyr -> yr]
    df['age_upp'] = df['age_upp'] * 1e9  # [Gyr -> yr]

    # Define optional columns
    if 'psr_area' in df.columns:
        df['psr_area'] = df['psr_area'] * 1e6  # [km^2 -> m^2]
    else:
        # Estimate psr area as 90% of crater area
        print('ugh')
        df['psr_area'] = 0.9 * np.pi * df.rad ** 2

    # Define new columns
    df['x'], df['y'] = latlon2xy(df.lon, df.lat)
    df['dist2pole'] = gc_dist(0, -90, df.lon, df.lat)
    return df


def read_volcanic_csv(volcanic_csv=VOLCANIC_CSV, columns=VOLCANIC_COLS):
    df = pd.read_csv(volcanic_csv, names=columns, header=3)
    return df


# Pre-compute grid functions
def get_ejecta_distances(df, grd_x=_GRD_X, grd_y=_GRD_Y):
    """
    Return 3D array shape (len(grd_x), len(grd_y), len(df)) of ejecta distances 
    from each crater in df.

    Distances computed with simple dist. Distances within crater radius are NaN.
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
    thickness = 0.14 * radius**0.74 * (distance / radius)**(-3.0)
    thickness[np.isnan(thickness)] = 0
    return thickness


def get_volcanic_ice(fvolcanic=VOLCANIC_CSV, dt=TIMESTEP, timestart=TIMESTART):
    """
    Return a matrix of the mass of volcanic ice produced at each timestep.
    """
    # TODO: add Tyler code
    return np.zeros((_NX, _NY, NT))

def get_volcanic_ice(fvolcanic=VOLCANIC_CSV, volcanic_spec=VOLCANIC_SPEC, 
                     pole_pct=POLE_PCT, coldtrap_area=COLDTRAP_AREA, 
                     time_arr=_TIME_ARR):
    """
    Return ice mass deposited in cold traps by volcanic outgassing over time.
    
    Values from supplemental spreadsheet S3 (Needham and Kring, 2017) 
    transient atmosphere data. Scale by coldtrap_area and pole_pct % of 
    material that is delievered to to S. pole.

    fvolcanic - string dictating path of N*K data file
    dt - timestep in main model
    timestart - initial start time given in main model
    scheme - Either 'NK' or 'Head'. 'Head' will output an array of zeros
    pole_perc - the percentage of material that makes it to the pole (given as a decimal)
    col - The atmospheric species that is being deposited. Choices include 'min_CO', 'max_CO', 'min_H2O', 'max_H2O',
            'min_H', 'max_H', 'min_S', 'max_S'

    @author: tylerpaladino
    """
    TIME_ARR = np.linspace(timestart, dt, int(timestart / dt))
    timestart = 
    out = np.zeros(len(TIME_ARR))
    if scheme.upper() == 'NK':

        tstart_ga = timestart/1e9

        r = 1737  # km
        theta = 6 * (np.pi/180)  # radians
        AOI = (2 * np.pi * r**2) * (1-np.cos(theta))  # surface area of south pole region
        # km^2 Poli︠a︡nin, A. D., & Manzhirov, A. V. (2007). Handbook of mathematics for engineers and scientists. Boca Raton, FL: Chapman & Hall/CRC.

        # surface area of entire moon
        surf_area_moon = 4 * np.pi * r**2  # km^2

        AOI_frac = AOI/surf_area_moon

        # Read in transient atmosphere data
        # fvolcanic = '/Users/tylerpaladino/Documents/ISU/LPI_NASA/Needham_tab.xlsx'
        cols = ('age', 'tot_vol', 'sphere_mass', 'min_CO', 'max_CO',
                'min_H2O', 'max_H2O', 'min_H', 'max_H', 'min_S', 'max_S', 'min_sum',
                'max_sum', 'min_psurf', 'max_psurf', 'min_atm_loss', 'max_atm_loss')

        df = pd.read_excel(fvolcanic, names=cols, skiprows=3)

        # Sort in descending order
        sdf = df.sort_values('age', ascending=False).reset_index().drop(columns=['index'])

        # Either remove old ages outside of range specified in input args or add on
        # ages to beginning of df to match input args
        if tstart_ga <= max(df.age):
            new_df = sdf.loc[list(sdf.age).index(tstart_ga):]
            new_df.reset_index(drop=True)
        else:
            year_diff = tstart_ga - max(df.age)
            num_rows = int(year_diff/0.1)  # 0.1 is interval between years in atmosphere data

            ages_needed = np.linspace(max(TIME_ARR)/1e9,
                                      max(df.age),
                                      num_rows,
                                      endpoint=False)

            new_array = np.zeros((len(ages_needed), df.shape[1]))
            # Place new ages into zeros array in correct position (0th)
            new_array[:, 0] = ages_needed

            new_df_row = pd.DataFrame(new_array, columns=cols)
            # Concatenate
            new_df = pd.concat([new_df_row, sdf]).reset_index(drop=True)

        # Add on min of TIME_ARR to end of df
        temp = pd.DataFrame([np.zeros(new_df.shape[1])], columns=cols)

        new_df = new_df.append(temp, ignore_index=True)

        # Insert age in years to df
        new_df.insert(1, 'age_yrs', new_df.age*1e9, True)  # Add in column of ages in years

        # Figure out how many data entries we need to create in new array to match TIME_ARR
        div_num = int(round(new_df.age_yrs[0]-new_df.age_yrs[1])/dt)
        # Loop through new_df and split up values based on div num into equal parts into out var.
        for i, row in new_df.iterrows():
            out[i*div_num:(i+1) * div_num] = row[col]/div_num

        return out * AOI_frac * pole_perc

    elif scheme.upper() == 'HEAD':
        return out


def get_ice_thickness(ice_mass, density=ICE_DENSITY, cold_trap_area=COLDTRAP_AREA):
    """
    Return ice thickness applied to all cold traps given total ice_mass 
    produced, density of ice and total cold_trap_area.
    """
    ice_volume = ice_mass / density  # [m^3]
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
    vu0 = 2e12 # [s^-1] (vibrational frequency of water)
    Ea = 0.456 # [eV] (activation energy)
    kb = 8.6e-5 # [ev K^-1] (Boltzmann constant, in eV units)
    tau = (1 / vu0) * np.exp(Ea / (kb * temp)) # [s], surface residence time 
    # TODO: covert tau to our units, get num H2O out/s, multiply by timestep
    # convert num H2O to thickness of ice
    return 0

# Ice column functions
def init_ice_columns(df, craters=COLD_TRAP_CRATERS, time_arr=_TIME_ARR):
    """
    Return dict of ice columns for cold trap craters in df.

    Currently: init after crater formed
    TODO: Maybe: init only after this PSR is stable, given instability, Sigler 2015
    
    dict[cname] = [row, col, area, ice_col]
    """
    ice_columns = {}
    for cname in craters:
        if not df.cname.str.contains(cname).any():
            print(f'Cold trap crater {cname} not found. Skipping...')
            continue
        crater = df[df.cname == cname].iloc[0]
        row = int(round(crater.x / GRDSTEP))
        col = int(round(crater.y / GRDSTEP))
        area = crater.psr_area
        ice_col = np.zeros(len(time_arr))
        ice_col[crater.age < _TIME_ARR] = np.nan  # no ice before crater formed
        ice_columns[cname] = [row, col, area, ice_col]
    return ice_columns


def ballistic_sed_ice_column(ice_column, ballistic_sed_grid):
    """Return ice column with ballistic sed grid applied"""
    # TODO: add code from Kristen
    return ice_column


def garden_ice_column(ice_column, ejecta_column, time, dt=TIMESTEP, area=COLDTRAP_AREA):
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


# Impact ice module
diam_range = {
    # Regime: (rad_min, rad_max, step)
    'A': (0, 0.01, None),    # Micrometeorites (<1 mm)
    'B': (0.01, 3, 1e-4),    # Small impactors (1 mm - 3 m)
    'C': (100, 1.5e3, 1),    # Simple craters, steep branch (100 m - 1.5 km)
    'D': (1.5e3, 15e3, 1e2), # Simple craters, shallow branch (1.5 km - 15 km)
    'E': (15e3, 300e3, 1e3)  # Complex craters, shallow branch (15 km - 300 km)
}

sfd_slope = {
    'B': -3.70,  # Small impactors
    'C': -3.82,  # Simple craters "steep" branch
    'D': -1.80,  # Simple craters "shallow" branch
    'E': -1.80   # Complex craters "shallow" branch
}

def total_impact_ice(age, regimes=('A', 'B', 'C', 'D', 'E')):
    """Return total impact ice from regimes A - E (Cannon 2020)."""
    total_ice = 0  # [kg]
    for regime in regimes:
        if regime == 'A':
            # Micrometeorites
            total_ice += ice_micrometeorites(age)
        elif regime == 'B':
            # Small impactors
            impactor_diams, impactors = get_impactor_pop(age, regime)
            total_ice += ice_small_impactors(age, impactor_diams, impactors)
        elif regime == 'C':
            # Small simple craters (continuous)
            crater_diams, craters = get_crater_pop(age, regime)
            total_ice += ice_small_craters(crater_diams, craters, regime)
        else:
            # Large simple & complex craters (stochastic)
            crater_diams = get_crater_pop(age, regime)
            total_ice += ice_large_craters(crater_diams, regime)
    return total_ice

def ice_micrometeorites(age, timestep=TIMESTEP):
    """
    Return ice from micrometeorites (Regime A, Cannon 2020).

    Grun et al. (2011) give 10**6 kg per year of < 1mm asteroid & comet grains.
    """
    # Multiply by years per timestep and assume 10% hydration
    scaling = 1e6 * timestep * 0.1
    # TODO: improve micrometeorite flux?
    # TODO: why not impactor_mass2water() here?
    # TODO: why not Avg retention from Ong et al., 2011
    return scaling * impact_flux(age) / impact_flux(0)


def ice_small_impactors(diams, num_per_bin, density=IMPACTOR_DENSITY):
    """
    Return ice mass [kg] from small impactors (Regime B, Cannon 2020) given
    impactor diams, num_per_bin, and density.
    """
    impactor_masses = diam2vol(diams) * density  # [kg]
    total_impactor_mass = sum(impactor_masses * num_per_bin)
    total_impactor_water = impactor_mass2water(total_impactor_mass)

    # Avg retention from Ong et al., 2011 (16.5% all asteroid mass retained)
    return total_impactor_water * 0.165


def ice_small_craters(crater_diams, craters, regime, impactor_density=IMPACTOR_DENSITY):
    """
    Return ice from simple craters, steep branch (Regime C, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams, 20, regime)  # [m]
    impactor_masses = diam2vol(impactor_diams) * impactor_density  # [kg]
    total_impactor_mass = sum(impactor_masses * craters)
    total_impactor_water = impactor_mass2water(total_impactor_mass)

    # Avg retention from Ong et al., 2011 (16.5% all asteroid mass retained)
    return total_impactor_water * 0.165


def ice_large_craters(crater_diams, regime, impactor_density=IMPACTOR_DENSITY):
    """
    Return ice from simple/complex craters, shallow branch (Regime D-E, Cannon 2020).
    """
    # Randomly include only craters formed by hydrated, Ctype asteroids
    # Uses same assumptions of impactor_mass2water()
    # TODO: make these assumptions global parameters
    rand_arr = _RNG.random(size=len(crater_diams))
    crater_diams = crater_diams[rand_arr < 0.36 * (2/3)]

    # Randomize impactor speeds with Gaussian around 20
    impactor_speeds = _RNG.normal(20, 6, len(crater_diams))
    impactor_speeds[impactor_speeds < 2.38] = 2.38 # minimum is Vesc
    impactor_diams = diam2len(crater_diams*1000, impactor_speeds, regime)
    impactor_masses =  diam2vol(impactor_diams) * impactor_density  # [kg]

    # Retain half unless impactor speed > 10 
    # TODO: where does water retention with speed eqn come from?
    water_retained = np.ones(len(impactor_speeds)) * 0.5
    water_retained[impactor_speeds >= 10] = 36.26*np.exp(-0.3464*impactor_speeds)
    water_retained[water_retained < 0] = 0

    # Assuming 10% hydration
    water_masses = impactor_masses * water_retained * 0.1

    # This is a direct copy of the water array in Cannon 2020 - maybe there 
    # was a function for mass of water released vs ice mass?
    ice_masses = water_masses

    # TODO: Why not avg retention from Ong et al. here?
    return sum(ice_masses)

def impactor_mass2water(impactor_mass):
    """
    Return water [kg] from impactor mass [kg] using assumptions of Cannon 2020:
        - 36% of impactors are C-type (Jedicke et al., 2018)
        - 2/3 of C-types are hydrated (Rivkin, 2012)
        - Hydrated impactors are 10% water by mass (Cannon et al., 2020)
    """
    return 0.36 * (2/3) * 0.1 * impactor_mass


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
    crater_diams = _RNG.choice(crater_diams, n_craters, p=sfd/sum(sfd))   
    return crater_diams


def impact_flux(time):
    """Return impact flux at time [yrs] (Derivative of eqn. 1, Ivanov 2008)."""
    time = time * 1e-9  # [yrs -> Ga] 
    return 6.93 * 5.44e-14 * (np.exp(6.93 * time)) + 8.38e-4


def neukum(diam, fit='1983'):
    """Return number of craters at diam (eqn. 2, Neukum 2001)."""
    a = {
        '1983': (-3.0768, -3.6269, 0.4366, 0.7935, 0.0865, -0.2649, -0.0664, 
                 0.0379, 0.0106, -0.0022, -5.18e-4, 3.97e-5),
        '2000': ()  # TODO: copy other chronology function
    }
    j = np.arange(len(a[fit]))
    return 10 ** np.sum(a[fit] * np.log10(diam)**j)


def get_diam_array(regime):
    """Return array of diameters based on diameters in diam_range."""
    dmin, dmax, step = diam_range[regime]
    n = int((dmax - dmin) / step)
    return np.linspace(dmin, dmax, n + 1)


def diam2len(diams, speeds, regime):
    """
    Return size of impactor based on diam of crater.
    
    Regime C (Prieur et al., 2017)

    Regime D (Collins et al., 2005)

    Regime E (Johnson et al., 2016)
    """
    # TODO: Kristen
    if regime == 'C':
        impactor_length = 0
    elif regime == 'D':
        impactor_length = 0
    elif regime == 'E':
        impactor_length = diam2len_johnson(diams)
    else:
        raise ValueError(f'Invalid regime {regime} in diam2len')
    return impactor_length


def diam2len_johnson(diam, rho_i=IMPACTOR_DENSITY, rho_t=TARGET_DENSITY, 
                     g=G_MOON, v=IMPACT_SPEED, theta=IMPACT_ANGLE):
    """
    Return impactor length from input diam using Johnson et al. (2016) method.
    """
    Dstar=(1.62 * 2700 * 1.8e4) / (g * rho_t)
    denom = (1.52 * (rho_i / rho_t)**0.38 * v**0.5 * g**-0.25 * 
             Dstar**-0.13 * np.sin(theta)**0.38)
    impactor_length = (diam / denom)**(1 / 0.88)
    return impactor_length


# Helper functions
def latlon2xy(lon, lat, rp=R_MOON):
    """
    Return (x, y) distance from pole of coords (lon, lat) [degrees]. 

    Returns in units of rp.
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    y = rp * np.cos(lat) * np.cos(lon)
    x = rp * np.cos(lat) * np.sin(lon)
    return x, y


def gc_dist(lon1, lat1, lon2, lat2, rp=R_MOON):
    """
    Calculate the great circle distance between two points using the Haversine formula.

    All args must be of equal length, lon and lat in decimal degrees.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return rp * 2 * np.arcsin(np.sqrt(a))


def dist(x1, y1, x2, y2):
    """Return simple distance between coordinates (x1, y1) and (x2, y2)."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def probabalistic_round(x):
    """
    Randomly round positive float x up or down based on its distance to x + 1.

    E.g. 6.1 will round down ~90% of the time and round up ~10% of the time
    such that over many trials, the expected value is 6.1.
    """
    return int(x + _RNG.random())


def diam2vol(diameter):
    """Return volume of sphere given diameter."""
    return (4/3) * np.pi *(diameter / 2)**3

if __name__ == '__main__':
    _ = main()