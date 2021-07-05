"""Set up configuration for the model run."""
from os import path, sep, environ, getcwd
from dataclasses import dataclass, fields, field, asdict
import numpy as np
import pandas as pd


@dataclass
class Cfg:
    """Class to configure a mixing model run."""
    _verbose = False
    mode: str = 'essi'  # 'essi' or 'cannon'
    seed: int = 0  # Set to None to get random results
    run: str = 'essi'  # Name of the current run
    run_date: str = pd.Timestamp.now().strftime("%y%m%d")
    run_time: str = pd.Timestamp.now().strftime("%H:%M:%S")

    # Paths set in post_init if not specified here
    modelpath: str = ''  # path to mixing.py
    datapath: str = ''  # path to import data
    outpath: str = ''  # path to save outputs
    
    # Files to import from datapath (attr name must end with "_in")
    crater_csv_in: str = 'crater_list.csv'
    volc_csv_in: str = 'needham_kring_2017.csv'
    costello_csv_in: str = 'costello_2018_t1.csv'

    # Files to export to outpath (attr name must end with "_out")
    save_npy: bool = False  # toggle saving age grid / ej matrix
    ejcols_csv_out: str = f'ej_columns_{run}.csv'
    icecols_csv_out: str = f'ice_columns_{run}.csv'
    runmeta_csv_out: str = f'run_metadata_{run}.csv'
    agegrd_npy_out: str = f'age_grid_{run}.npy'
    ejmatrix_npy_out: str = f'ejecta_matrix_{run}.npy'

    # Grid and time size and resolution
    dtype = np.float32  # np.float64 (32 should be good for most purposes)
    grdxsize: int = 400e3  # [m]
    grdysize: int = 400e3  # [m]
    grdstep: int = 1e3  # [m / pixel]
    timestart: int = 4.25e9  # [yr]
    timeend: int = 0  # [yr]
    timestep: int = 10e6  # [yr]

    # Cannon model params
    coldtrap_max_temp: float = 120  # [k]
    coldtrap_area: float = 1.3e4 * 1e6  # [m^2], (williams 2019, via text s1, cannon 2020)
    ice_hop_efficiency: float = 0.054  # 5.4% gets to the s. pole (text s1, cannon 2020)
    cold_trap_craters: tuple = (
        'Haworth', 'Shoemaker', 'Faustini', 'Shackleton', 'Amundsen',
        'Sverdrup', 'Cabeus B', 'de Gerlache', "Idel'son L", 'Wiechert J')

    # Lunar constants
    rad_moon: float = 1737e3  # [m], lunar radius
    grav_moon: float = 1.62  # [m s^-2], gravitational acceleration
    sa_moon: float = 4 * np.pi * rad_moon ** 2  # [m^2]
    simple2complex: float = 18e3  # [m], lunar s2c transition diameter (melosh 1989)
    complex2peakring: float = 1.4e5  # [m], lunar c2pr transition diameter (melosh 1989)

    # Impact cratering constants
    impactor_density: float = 1300  # [kg m^-3], cannon 2020
    impactor_density_avg: float = 2780  # [kg m^-3] costello 2018
    # impactor_density = 3000  # [kg m^-3] ordinary chondrite (melosh scaling)
    # impact_speed = 17e3  # [m/s] average impact speed (melosh scaling)
    impact_speed: float = 20e3  # [m/s] mean impact speed (cannon 2020)
    impact_sd: float = 6e3  # [m/s] standard deviation impact speed (cannon 2020)
    escape_vel: float = 2.38e3  # [m/s] lunar escape velocity
    impact_angle: float = 45  # [deg]  average impact velocity
    target_density: float = 1500  # [kg m^-3] (cannon 2020)
    bulk_density: float = 2700  # [kg m^-3] simple to complex (melosh)
    ice_erosion_rate: float = 0.1 * (timestep / 10e6)  # [m], 10 cm / 10 ma (cannon 2020)

    # Ice constants
    ice_density: float = 934  # [kg m^-3], (cannon 2020)
    ice_melt_temp: float = 273  # [k]
    ice_latent_heat: float = 334e3  # [j/kg] latent heat of h2o ice

    # Ejecta shielding module
    crater_cols: tuple = ('cname', 'lat', 'lon', 'diam', 'age', 'age_low',
                          'age_upp', 'psr_area', 'age_ref', 'prio', 'notes')
    ejecta_thickness_order: float = -3  # min: -3.5, avg: -3, max: -2.5 (kring 1995)

    # Ballistic sedimentation module
    ice_frac: float = 0.056  # fraction ice vs regolith (5.6% colaprete 2010)
    heat_frac: float = 0.5  # fraction of ballistic ke used in heating vs mixing
    heat_retained: float = 0.1  # fraction of heat retained (10-30%; stopar 2018)
    regolith_cp: float = 4.3e3  # heat capacity [j kg^-1 k^-1] (0.7-4.2 kj/kg/k for h2o)

    # Impact gardening module (Costello 2020)
    overturn_prob_pct: str = '99%'  # poisson probability ['10%', '50%', '99%'] (table 1, costello 2018)
    n_overturn: int = 100  # number of overturns needed for ice loss
    crater_proximity: float = 0.41  # crater proximity scaling parameter
    depth_overturn: float = 0.04  # fractional depth overturned
    target_kr: float = 0.6  # costello 2018, for lunar regolith
    target_k1: float = 0.132  # costello 2018, for lunar regolith
    target_k2: float = 0.26  # costello 2018, for lunar regolith
    target_mu: float = 0.41  # costello 2018, for lunar regolith
    target_yield_str: float = 0.01*1e6  # [pa] costello 2018, for lunar regolith
    overturn_regimes: tuple = ('primary', 'secondary', 'micrometeorite')
    overturn_ab: dict = field(default_factory = lambda: ({  # overturn sfd params aD^b (table 2, costello et al. 2018)
        'primary': (6.3e-11, -2.7), 
        'secondary': (7.25e-9, -4), # 1e5 secondaries, -4 slope from mcewen 2005
        'micrometeorite': (1.53e-12, -2.64)
    }))
    impact_speeds: dict = field(default_factory = lambda: ({
        'primary': 1800,  # [km/s]
        'secondary': 507,  # [km/s]
        'micrometeorite': 1800  # [km/s]
    }))

    # Impact ice module   
    mm_mass_rate: float = 1e6  # [kg/yr], lunar micrometeorite flux (grun et al. 2011)
    ctype_frac: float = 0.36  # 36% of impactors are c-type (jedicke et al., 2018)
    ctype_hydrated: float = 2 / 3  # 2/3 of c-types are hydrated (rivkin, 2012)
    hydrated_wt_pct: float = 0.1  # impactors wt% H2O (cannon 2020)
    impactor_mass_retained: float = 0.165  # asteroid mass retained in impact (ong et al., 2011)
    impact_regimes: tuple = ('a', 'b', 'c', 'd', 'e')
    diam_range: dict = field(default_factory = lambda: ({
        # regime: (rad_min, rad_max, step)
        'a': (0, 0.01, None),  # micrometeorites (<1 mm)
        'b': (0.01, 3, 1e-4),  # small impactors (1 mm - 3 m)
        'c': (100, 1.5e3, 1),  # simple craters, steep sfd (100 m - 1.5 km)
        'd': (1.5e3, 15e3, 1e2),  # simple craters, shallow sfd (1.5 km - 15 km)
        # 'e': (15e3, 300e3, 1e3),  # complex craters, shallow sfd (15 km - 300 km)
        'e': (15e3, 500e3, 1e3),  # complex craters, shallow sfd (15 km - 300 km)
    }))
    sfd_slopes: dict = field(default_factory = lambda: ({
        'b': -3.70,  # small impactors
        'c': -3.82,  # simple craters 'steep' branch
        'd': -1.80,  # simple craters 'shallow' branch
        'e': -1.80,  # complex craters 'shallow' branch
    }))

    # Volcanic ice module
    volc_mode: str = 'Head'  # ['Head', 'NK']

    # volc_mode == 'Head': Head et al. (2020)
    volc_early: tuple = (4e9, 3e9)  # [yrs]
    volc_late: tuple = (3e9, 2e9)  # [yrs]
    volc_early_pct: float = 0.75  # 75%
    volc_late_pct: float = 0.25  # 25%
    volc_total_vol: float = 1e7 * 1e9  # [m^3] basalt
    volc_h2o_ppm: float = 10  # [ppm]
    volc_magma_density: float = 3000  # [kg/m^3]

    # volc_mode == 'NK': Needham & Kring (2017)
    volc_pole_pct: float = 0.1  # 10%
    volc_species: str = 'min_h2o'  # volcanic species, must be in volc_cols
    volc_cols: tuple = (
        'age', 'tot_vol', 'sphere_mass', 'min_co', 'max_co', 'min_h2o', 
        'max_h2o', 'min_h', 'max_h','min_s', 'max_s', 'min_sum', 'max_sum',
        'min_psurf', 'max_psurf', 'min_atm_loss', 'max_atm_loss'
    )

    # lunar production function a_values (neukum 2001)
    neukum1983: tuple = (-3.0768, -3.6269, 0.4366, 0.7935, 0.0865, -0.2649, 
                         -0.0664, 0.0379, 0.0106, -0.0022, -5.18e-4, 3.97e-5)
    ivanov2000: tuple = (-3.0876, -3.557528, 0.781027, 1.021521, -0.156012, 
                         -0.444058, 0.019977, 0.086850, -0.005874, -0.006809, 
                         8.25e-4, 5.54e-5)

    # Make arrays
    _grdy, _grdx = np.meshgrid(
        np.arange(grdysize, -grdysize, -grdstep, dtype=dtype), 
        np.arange(-grdxsize, grdxsize, grdstep, dtype=dtype), 
        sparse=True, indexing='ij'
    )
    _time_arr = np.linspace(timestart, timestep, int(timestart / timestep), dtype=dtype)

    # length of arrays
    _ny, _nx = _grdy.shape[0], _grdx.shape[1]
    nt = len(_time_arr)


    def to_dict_no_underscore(self):
        """Return dict of dataclass, remove underscore fields."""
        return {k: v for k, v in asdict(self).items() if k[0] != '_'}


    def enforce_dataclass_type(self, field):
        """
        Force set all dataclass types from their type hint, raise error if invalid.
        
        Since scientific notation is float in Python, this forces int specified in 
        scientific notation to be int in the code.
        """
        value = getattr(self, field.name)
        try:
            setattr(self, field.name, field.type(value))
        except ValueError:
            msg = f'Type mismatch for {field.name} in config.py: ' 
            msg += f'Expected: {field.type}. Got: {type(value)}.'
            raise ValueError(msg)


    def make_paths_absolute(self, field, datapath, outpath):
        """
        Make all file paths absolute. 
        
        Prepend datapath to all fields ending with "_in".
        Prepend outpath to all fields ending with "_out".
        """
        value = getattr(self, field.name)
        if '_in' in field.name:
            setattr(self, field.name, path.join(datapath, value))
        elif '_out' in field.name:
            setattr(self, field.name, path.join(outpath, value))


    def __post_init__(self):
        """Force set all cfg types, raise error if invalid type."""
        if self.modelpath == '':
            setattr(self, 'modelpath', get_model_path())
        if self.datapath == '':
            modelpath = getattr(self, 'modelpath')
            datapath = path.abspath(path.join(modelpath, '..', 'data'))
            setattr(self, 'datapath', datapath)
        if self.outpath == '':
            datapath = getattr(self, 'datapath')
            run = getattr(self, 'run')
            run_date = getattr(self, 'run_date')
            seed = getattr(self, 'seed')
            outpath = path.join(datapath, run_date, f'{run}_{seed}')
            setattr(self, 'outpath', outpath)
        for field in fields(self):
            self.make_paths_absolute(field, self.datapath, self.outpath)
            self.enforce_dataclass_type(field)


# Config helper functions
def get_model_path():
    """
    Return path to directory containing mixing.py assuming following structure:

    /project
        /data
        /essi21
            - config.py (this file)
            - mixing.py
        /figs
        /test

    """
    # Try to import absolute path from installed essi21 module
    try:  # Python > 3.7
        import importlib.resources as pkg_resources
    except ImportError:  # Python < 3.7
        import importlib_resources as pkg_resources
    try:
        with pkg_resources.path('essi21', 'mixing.py') as fpath:
            modelpath = fpath.parent.as_posix()
    except ModuleNotFoundError:
        # If essi21 module not installed, get path from current file
        if "JPY_PARENT_PID" in environ:  # Jupyter (assume user used chdir)
            modelpath = getcwd() + sep
        else:  # Non-notebook - assume user is running the .py script directly
            modelpath = path.abspath(path.dirname(__file__)) + sep
    return modelpath
