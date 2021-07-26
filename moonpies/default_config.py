"""DO NOT EDIT. Default config for moonpies (see README for config guide)."""
import ast, pprint
from os import path, sep, environ, getcwd
from dataclasses import dataclass, fields, field, asdict
import numpy as np
import pandas as pd


@dataclass
class Cfg:
    """Class to configure a mixing model run."""
    seed: int = 0  # Note: Should not be set here - set when model is run only
    run_name: str = 'mpies'  # Name of the current run
    verbose: bool = False  # Print info as model is running
    write: bool = True  # Write model outputs to a file (if False, just return)
    write_npy: bool = False  # Write large arrays to files - slow! (age_grid, ej_thickness)
    plot: bool = False  # Save strat column plots - slow!
    mode: str = 'cannon'  # 'moonpies' or 'cannon'
    run_date: str = pd.Timestamp.now().strftime("%y%m%d")
    run_time: str = pd.Timestamp.now().strftime("%H:%M:%S")

    # Paths set in post_init if not specified here
    modelpath: str = ''  # path to mixing.py
    datapath: str = ''  # path to import data
    outpath: str = ''  # path to save outputs
    figspath: str = ''  # path to save figures
    
    # Files to import from datapath (attr name must end with "_in")
    crater_csv_in: str = 'crater_list.csv'
    basin_csv_in: str = 'basin_list.csv'
    nk_csv_in: str = 'needham_kring_2017_s3.csv'
    costello_csv_in: str = 'costello_etal_2018_t1.csv'
    bahcall_csv_in: str = 'bahcall_etal_2001_t2.csv'

    # Files to export to outpath (attr name must end with "_out")
    ejcols_csv_out: str = f'ej_columns_{run_name}.csv'
    icecols_csv_out: str = f'ice_columns_{run_name}.csv'
    config_py_out: str = f'config_{run_name}.py'
    agegrd_npy_out: str = f'age_grid_{run_name}.npy'
    ejmatrix_npy_out: str = f'ejecta_matrix_{run_name}.npy'

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
    coldtrap_area: float = 1.3e4 * 1e6  # [m^2], (Williams 2019, via text s1, Cannon 2020)
    ice_hop_efficiency: float = 0.054  # 5.4% gets to the s. pole (text s1, Cannon 2020)
    coldtrap_craters: tuple = (
        'Haworth', 'Shoemaker', 'Faustini', 'Shackleton', 'Slater', 'Amundsen', 
        'Cabeus', 'Sverdrup', 'de Gerlache', "Idel'son L", 'Wiechert J')

    # Lunar constants
    rad_moon: float = 1737.4e3  # [m], lunar radius
    grav_moon: float = 1.62  # [m s^-2], gravitational acceleration
    sa_moon: float = 4 * np.pi * rad_moon ** 2  # [m^2]
    simple2complex: float = 18e3  # [m], lunar s2c transition diameter (Melosh 1989)
    complex2peakring: float = 1.4e5  # [m], lunar c2pr transition diameter (Melosh 1989)

    # Impact cratering constants
    impactor_density: float = 1300  # [kg m^-3], Cannon 2020
    impactor_density_avg: float = 2780  # [kg m^-3] Costello 2018
    # impactor_density = 3000  # [kg m^-3] ordinary chondrite (Melosh scaling)
    # impact_speed = 17e3  # [m/s] average impact speed (Melosh scaling)
    impact_speed: float = 20e3  # [m/s] mean impact speed (Cannon 2020)
    impact_sd: float = 6e3  # [m/s] standard deviation impact speed (Cannon 2020)
    escape_vel: float = 2.38e3  # [m/s] lunar escape velocity
    impact_angle: float = 45  # [deg]  average impact velocity
    target_density: float = 1500  # [kg m^-3] (Cannon 2020)
    bulk_density: float = 2700  # [kg m^-3] simple to complex (Melosh)
    ice_erosion_rate: float = 0.1 * (timestep / 10e6)  # [m], 10 cm / 10 ma (Cannon 2020)

    # Ice constants
    ice_density: float = 934  # [kg m^-3], (Cannon 2020)
    ice_melt_temp: float = 273  # [k]
    ice_latent_heat: float = 334e3  # [j/kg] latent heat of h2o ice

    # Ejecta shielding module
    crater_cols: tuple = ('cname', 'lat', 'lon', 'diam', 'age', 'age_low',
                          'age_upp', 'psr_area', 'age_ref', 'prio', 'notes')
    ejecta_thickness_order: float = -3  # min: -3.5, avg: -3, max: -2.5 (kring 1995)

    # Basin ice module
    basin_cols: tuple = ('cname', 'lat', 'lon', 'diam', 'inner_ring_diam', 
                         'bouger_diam', 'age', 'age_low', 'age_upp', 'ref')
    basin_impact_speed = 20e3  # [km/s]

    # Ballistic sedimentation module
    ice_frac: float = 0.056  # fraction ice vs regolith (5.6% colaprete 2010)
    heat_frac: float = 0.5  # fraction of ballistic ke used in heating vs mixing
    heat_retained: float = 0.1  # fraction of heat retained (10-30%; stopar 2018)
    regolith_cp: float = 4.3e3  # heat capacity [j kg^-1 k^-1] (0.7-4.2 kj/kg/k for h2o)

    # Impact gardening module (Costello 2020)
    overturn_prob_pct: str = '99%'  # poisson probability ['10%', '50%', '99%'] (table 1, Costello 2018)
    n_overturn: int = 100  # number of overturns needed for ice loss
    crater_proximity: float = 0.41  # crater proximity scaling parameter
    depth_overturn: float = 0.04  # fractional depth overturned
    target_kr: float = 0.6  # Costello 2018, for lunar regolith
    target_k1: float = 0.132  # Costello 2018, for lunar regolith
    target_k2: float = 0.26  # Costello 2018, for lunar regolith
    target_mu: float = 0.41  # Costello 2018, for lunar regolith
    target_yield_str: float = 0.01*1e6  # [pa] Costello 2018, for lunar regolith
    overturn_regimes: tuple = ('primary', 'secondary', 'micrometeorite')
    overturn_ab: dict = field(default_factory = lambda: ({  # overturn sfd params aD^b (table 2, Costello et al. 2018)
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
    ctype_hydrated: float = 2/3  # 2/3 of c-types are hydrated (rivkin, 2012)
    hydrated_wt_pct: float = 0.1  # impactors wt% H2O (Cannon 2020)
    impactor_mass_retained: float = 0.165  # asteroid mass retained in impact (ong et al., 2011)
    # impact_regimes: tuple = ('a', 'b', 'c', 'd', 'e')  # TODO: how to handle modes + regimes?
    impact_regimes: tuple = ('a', 'b', 'c', 'd', 'e', 'f')
    diam_range: dict = field(default_factory = lambda: ({
        # regime: (rad_min, rad_max, step)
        'a': (0, 0.01, None),  # micrometeorites (<1 mm)
        'b': (0.01, 3, 1e-4),  # small impactors (10 mm - 3 m)
        'c': (100, 1.5e3, 1),  # simple craters, steep sfd (100 m - 1.5 km)
        'd': (1.5e3, 15e3, 1e2),  # simple craters, shallow sfd (1.5 km - 15 km)
        'e': (15e3, 300e3, 1e3),  # complex craters, shallow sfd (15 km - 300 km)
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
    volc_total_vol: float = 1e7 * 1e9 * 1e-9 # [m^3 yr^-1] basalt
    volc_h2o_ppm: float = 10  # [ppm]
    volc_magma_density: float = 3000  # [kg/m^3]

    # volc_mode == 'NK': Needham & Kring (2017)
    volc_pole_pct: float = 0.1  # 10%
    nk_species: str = 'min_h2o'  # volcanic species, must be in volc_cols
    nk_cols: tuple = (
        'time', 'tot_vol', 'sphere_mass', 'min_co', 'max_co', 'min_h2o', 
        'max_h2o', 'min_h', 'max_h', 'min_s', 'max_s', 'min_sum', 'max_sum',
        'min_psurf', 'max_psurf', 'min_atm_loss', 'max_atm_loss'
    )
    volc_cols: tuple = (
        'time', 'tot_vol', 'sphere_mass', 'min_co', 'max_co', 'min_h2o', 
        'max_h2o', 'min_h', 'max_h', 'min_s', 'max_s', 'min_sum', 'max_sum',
        'min_psurf', 'max_psurf', 'min_atm_loss', 'max_atm_loss'
    )

    # Solar Wind module
    solar_wind_mode: str = 'Benna'  # ['Benna', 'Lucey-Hurley']
    faint_young_sun: bool = True  # use faint young sun (Bahcall et al., 2001)

    # lunar production function a_values (neukum 2001)
    neukum1983: tuple = (-3.0768, -3.6269, 0.4366, 0.7935, 0.0865, -0.2649, 
                         -0.0664, 0.0379, 0.0106, -0.0022, -5.18e-4, 3.97e-5)
    ivanov2000: tuple = (-3.0876, -3.557528, 0.781027, 1.021521, -0.156012, 
                         -0.444058, 0.019977, 0.086850, -0.005874, -0.006809, 
                         8.25e-4, 5.54e-5)


    def to_dict(self):
        """Return dict representation of dataclass."""
        return asdict(self)


    def to_string(self, fdict={}):
        """Return representation of self as dict formatted python string."""
        if not fdict:
            fdict = self.to_dict()
        try:
            s = pprint.pformat(fdict, compact=True, sort_dicts=False)
        except TypeError:
            # Cancel sorting on Python < 3.8
            pprint._sorted = lambda x:x  # Python 3.6
            pprint.sorted = lambda x, key=None: x  # Python 3.7
            s = pprint.pformat(fdict, compact=True)
        return s


    def to_py(self, outpath, fstring=''):
        """Write dataclass to dict at outpath."""
        if not fstring:
            fstring = self.to_string()
        with open(outpath, 'w') as f:
            f.write(fstring)


    def to_default(self, outpath):
        """Write defaults to my_config.py"""
        ddict = self.to_dict()
        del ddict['seed']
        del ddict['run_time']
        del ddict['run_date']
        for k, v in ddict.copy().items():
            if 'path' in k:
                ddict[k] = ''
            elif '_in' in k or '_out' in k:
                ddict[k] = path.basename(v)
        self.to_py(outpath, self.to_string(ddict))


    def __post_init__(self):
        """Force set all cfg types, raise error if invalid type."""
        setattr(self, 'seed', _get_random_seed(self))
        setattr(self, 'modelpath', _get_modelpath(self))
        setattr(self, 'datapath', _get_datapath(self))
        setattr(self, 'outpath', _get_outpath(self))
        setattr(self, 'figspath', _get_figspath(self))
        for field in fields(self):
            _make_paths_absolute(self, field, self.datapath, self.outpath)
            _enforce_dataclass_type(self, field)


# Config helper functions
def _get_random_seed(cfg):
    """Return random_seed in (1, 99999) if not set in cfg."""
    seed = cfg.seed
    if not seed:
        seed = np.random.randint(1, 99999)
    return seed


def _get_modelpath(cfg):
    """
    Return path to directory containing mixing.py assuming following structure:

    /project
        /data
        /moonpies
            - config.py (this file)
            - mixing.py
        /figs
        /test

    """
    if cfg.modelpath != '':
        return cfg.modelpath
    # Try to import absolute path from installed moonpies module
    try:  # Python > 3.7
        import importlib.resources as pkg_resources
    except ImportError:  # Python < 3.7
        import importlib_resources as pkg_resources
    try:
        with pkg_resources.path('moonpies', 'moonpies.py') as fpath:
            modelpath = fpath.parent.as_posix()
    except (TypeError, ModuleNotFoundError):
        # If module not installed, get path from current file
        if "JPY_PARENT_PID" in environ:  # Jupyter (assume user used chdir)
            modelpath = getcwd()
        else:  # Non-notebook - assume user is running the .py script directly
            modelpath = path.abspath(path.dirname(__file__))
    return modelpath + sep


def _get_datapath(cfg):
    """Return default datapath if not specified in cfg."""
    datapath = cfg.datapath
    if datapath == '':
        modelpath = getattr(cfg, 'modelpath')
        datapath = path.abspath(path.join(modelpath, '..', 'data')) + sep
    return datapath


def _get_figspath(cfg):
    """Return default datapath if not specified in cfg."""
    figspath = cfg.figspath
    if figspath == '':
        modelpath = getattr(cfg, 'modelpath')
        figspath = path.abspath(path.join(modelpath, '..', 'figs')) + sep
    return figspath


def _get_outpath(cfg):
    """Return default outpath if not specified in cfg."""
    outpath = cfg.outpath
    run_seed = f'{cfg.seed:05d}'
    if outpath != '' and run_seed not in outpath:
        # Prevent overwrites by appending run_seed/ dir to end of outpath
        if path.basename(path.normpath(outpath)).isnumeric():
            # Trim previous seed path if it exists
            outpath = path.dirname(path.normpath(outpath))
        outpath = path.join(outpath, run_seed)
    elif outpath == '':
        # Default outpath is datapath/yymmdd_runname/seed/
        datapath = cfg.datapath
        run_dir = f'{cfg.run_date}_{cfg.run_name}'
        outpath = path.join(datapath, run_dir, run_seed)
    return outpath + sep


def _enforce_dataclass_type(cfg, field):
    """
    Force set all dataclass types from their type hint, raise error if invalid.
    
    Since scientific notation is float in Python, this forces int specified in 
    scientific notation to be int in the code.
    """
    value = getattr(cfg, field.name)
    try:
        setattr(cfg, field.name, field.type(value))
    except ValueError:
        msg = f'Type mismatch for {field.name} in config.py: ' 
        msg += f'Expected: {field.type}. Got: {type(value)}.'
        raise ValueError(msg)


def _make_paths_absolute(cfg, field, datapath, outpath):
    """
    Make all file paths absolute. 
    
    Prepend datapath to all fields ending with "_in".
    Prepend outpath to all fields ending with "_out".
    """
    value = getattr(cfg, field.name)
    newpath = ''
    if '_in' in field.name:
        newpath = path.join(datapath, value)
    elif '_out' in field.name:
        # Recompute path to _out files in case outpath changed (e.g. appending seed dir)
        newpath = path.join(outpath, path.basename(value))
    elif 'path' in field.name:
        newpath = value

    if newpath:
        setattr(cfg, field.name, path.abspath(path.expanduser(newpath)))


def read_custom_cfg(cfg_path):
    """
    Return dictionary from custom config.py file.
    """
    if cfg_path is None:
        return {}
    with open(path.abspath(cfg_path), 'r') as f:
        cfg_dict = ast.literal_eval(f.read())
    return cfg_dict


def from_dict(cdict):
    """Return Cfg setting all provided values in cdict."""
    return Cfg(**cdict)


if __name__ == '__main__':
    Cfg().to_default('my_config.py')
    print('Wrote my_config.py')
