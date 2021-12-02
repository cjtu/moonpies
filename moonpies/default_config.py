"""DO NOT EDIT. Default config for moonpies (see README for config guide)."""
import ast, pprint
from os import path, sep, environ, getcwd
from dataclasses import dataclass, fields, field, asdict
import numpy as np
import pandas as pd
try:
    from ._version import __version__
except ImportError:
    from _version import __version__

MODE_DEFAULTS = {
    'cannon': {
        'solar_wind_ice': False,
        'ballistic_hop_moores': False,  # constant hop_effcy 
        'ejecta_basins': False,
        'impact_ice_basins': False,
        'impact_ice_comets': False,
        'use_volc_dep_effcy': False,  # Use ballistic_hop efficiency rather than volc_dep_efficiency
        'ballistic_sed': False,
        'impact_gardening_costello': False,
        'impact_speed_mean': 20e3, # [m/s] 
    },
    'moonpies': {
        'solar_wind_ice': True,
        'ballistic_hop_moores': True,  # hop_effcy per crater (Moores et al 2016)
        'ejecta_basins': True,
        'impact_ice_basins': True,
        'impact_ice_comets': True,
        'use_volc_dep_effcy': True, # Use volc_dep_efficiency instead of ballistic_hop_efficiency
        'ballistic_sed': True, # TODO: improvements in progress
        'impact_gardening_costello': True,
        'impact_speed_mean': 17e3,  # [m/s] 
    }
}

COLDTRAP_DEFAULTS = {
    's': {
        'ballistic_hop_effcy': 0.054,  # Cannon et al. (2020)
        'volc_dep_effcy': 0.258,  # 25.8% Spole (Wilcoski et al., 2021)
        'coldtrap_names': (
            'Haworth', 'Shoemaker', 'Faustini', 'Shackleton', 'Slater', 
            'Amundsen', 'Cabeus', 'Sverdrup', 'de Gerlache', "Idel'son L", 
            'Wiechert J', 'Cabeus B'
        ),
    },
    'n': {
        'ballistic_hop_effcy': 0.027,  # Cannon et al. (2020)
        'volc_dep_effcy': 0.152,  # 15.2% Npole (Wilcoski et al., 2021)
        'coldtrap_names': ('Fibiger', 'Hermite', 'Hermite A', 'Hevesy', 
            'Lovelace', 'Nansen A', 'Nansen F', 'Rozhdestvenskiy U', 
            'Rozhdestvenskiy W', 'Sylvester')
    },
    'coldtrap_max_temp':{
        'H2O': 110,  # [K]
        'CO2': 60  # [K]
    },
    'coldtrap_area': {
        's': {
            'H2O': 1.3e4 * 1e6,  # [m^2], (Williams et al., 2019) (1.7e4 * 1e6 Shorghofer 2020)
            'CO2': 104 * 1e6, # [m^2], (<60 K max summer T, Williams et al., 2019)
        },
        'n': {
            'H2O': 5.3e3 * 1e6,  # [m^2], (Williams et al., 2019)
            'CO2': 0  # [m^2], TODO: compute
        },
    }
}

@dataclass
class Cfg:
    """Class to configure a mixing model run."""
    seed: int = 0  # Note: Should not be set here - set when model is run only
    run_name: str = 'mpies'  # Name of the current run
    _version: str = __version__  # Current version of MoonPIES
    run_date: str = pd.Timestamp.now().strftime("%y%m%d")
    run_time: str = pd.Timestamp.now().strftime("%H:%M:%S")

    # Model output behavior
    verbose: bool = False  # Print info as model is running
    write: bool = True  # Write model outputs to a file (if False, just return)
    write_npy: bool = False  # Write large arrays to files - slow! (age_grid, ej_thickness)
    plot: bool = False  # Save strat column plots - slow!
    
    # Setup Cannon vs MoonPIES config modes
    mode: str = 'moonpies'  # ['moonpies', 'cannon']

    # set in __post_init__ by _set_mode_defaults(self, self.mode, MODE_DEFAULTS)
    solar_wind_ice: bool = None
    ballistic_hop_moores: bool = None  # hop_effcy per crater (Moores et al 2016)
    ejecta_basins: bool = None
    impact_ice_basins: bool = None
    impact_ice_comets: bool = None
    use_volc_dep_effcy: bool = None  # Use volc_dep_efficiency rather than ballistic hop for volcanics
    ballistic_sed: bool = None
    impact_gardening_costello: bool = None

    # Lunar pole and coldtrap defaults
    pole: str = 's'  # ['s', 'n'] TODO: only s is currently supported
    ice_species = 'H2O'  # ['H2O', 'CO2']
    
    # set in __post_init__ by _set_coldtrap_defaults(self, self.pole, self.ice_species, COLDTRAP_DEFAULTS)
    ballistic_hop_effcy: float = None
    volc_dep_effcy: float = None
    coldtrap_area: float = None
    coldtrap_names: tuple = None
    coldtrap_max_temp: float = None

    # Paths set in post_init if not specified in custom config
    modelpath: str = ''  # path to mixing.py
    datapath: str = ''  # path to import data
    outpath: str = ''  # path to save outputs
    figspath: str = ''  # path to save figures
    
    # Files to import from datapath (attr name must end with "_in")
    crater_csv_in: str = 'crater_list.csv'
    basin_csv_in: str = 'basin_list.csv'
    nk_csv_in: str = 'needham_kring_2017_s3.csv'
    costello_csv_in: str = 'costello_etal_2018_t1_expanded.csv'
    bahcall_csv_in: str = 'bahcall_etal_2001_t2.csv'
    bhop_csv_in: str = 'ballistic_hop_coldtraps.csv'
    bsed_frac_mean_in: str = 'ballistic_sed_frac_melted_mean.csv'
    bsed_frac_std_in: str = 'ballistic_sed_frac_melted_std.csv'

    # Files to export to outpath (attr name must end with "_out")
    ej_t_csv_out: str = f'ej_columns_{run_name}.csv'
    ice_t_csv_out: str = f'ice_columns_{run_name}.csv'
    config_py_out: str = f'run_config_{run_name}.py'
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

    # Lunar constants
    rad_moon: float = 1737.4e3  # [m], lunar radius
    grav_moon: float = 1.62  # [m s^-2], gravitational acceleration
    sa_moon: float = 4 * np.pi * rad_moon ** 2  # [m^2]
    simple2complex: float = 18e3  # [m], lunar s2c transition diameter (Melosh 1989)
    complex2peakring: float = 1.4e5  # [m], lunar c2pr transition diameter (Melosh 1989)

    # Impact cratering constants
    impactor_density: float = 1300  # [kg m^-3], (Carry 2012 via Cannon 2020)
    # impactor_density = 3000  # [kg m^-3] ordinary chondrite (Melosh scaling)
    # impact_speed = 17e3  # [m/s] average impact speed (Melosh scaling)
    impact_speed_mean: float = 20e3  # [m/s] mean impact speed (Cannon 2020)
    impact_speed_sd: float = 6e3  # [m/s] standard deviation impact speed (Cannon 2020)
    escape_vel: float = 2.38e3  # [m/s] lunar escape velocity
    impact_angle: float = 45  # [deg]  average impact velocity
    target_density: float = 1500  # [kg m^-3] (Cannon 2020)
    bulk_density: float = 2700  # [kg m^-3] simple to complex (Melosh)
    ice_erosion_rate: float = 0.1 # [m], 10 cm / 10 ma (Cannon 2020)
    ej_threshold: float = 4  # [crater radii] Radius of influence of a crater (-1: no threshold)
    thickness_threshold: float = 1e-3  # [m] minimum thickness to form a layer
    neukum_pf_new: bool = True  # [True=>2001, False=>1983] Neukum production function version (Neukum et al. 2001)
    neukum_pf_a_2001: tuple = (-3.0876, -3.557528, 0.781027, 1.021521, -0.156012, -0.444058, 0.019977, 0.086850, -0.005874, -0.006809, 8.25e-4, 5.54e-5)
    neukum_pf_a_1983: tuple = (-3.0768, -3.6269, 0.4366, 0.7935, 0.0865, -0.2649, -0.0664, 0.0379, 0.0106, -0.0022, -5.18e-4, 3.97e-5)

    # Ice constants
    ice_density: float = 934  # [kg m^-3], (Cannon 2020)
    ice_melt_temp: float = 273  # [k]
    ice_latent_heat: float = 334e3  # [j/kg] latent heat of h2o ice

    # Ejecta shielding module
    crater_cols: tuple = ('cname', 'lat', 'lon', 'psr_lat', 'psr_lon', 'diam', 
                          'age', 'age_low','age_upp', 'psr_area', 'age_ref', 
                          'prio', 'notes')
    ejecta_thickness_order: float = -3  # min: -3.5, avg: -3, max: -2.5 (Kring 1995)

    # Basin ice module
    basin_cols: tuple = ('cname', 'lat', 'lon', 'diam', 'inner_ring_diam', 
                         'bouger_diam', 'age', 'age_low', 'age_upp', 'ref')
    basin_impact_speed = 20e3  # [km/s]

    # Ballistic sedimentation module
    ballistic_teq: bool = False  # Do ballistic sed only if teq > coldtrap_max_temp
    mixing_ratio_petro: bool = True  # Use Petro and Pieter (2006) adjustment to volume fraction
    mixing_ratio_a: float = 0.0183  # (Oberbeck et al. 1975 via Eq 4 Zhang et al. 2021)  #2.913  # Fit to Ries crater ballistic sed, a
    mixing_ratio_b: float = 0.87  # (Oberbeck et al. 1975 via Eq 4 Zhang et al. 2021)  #-3.978  # Fit to Ries crater ballistic sed, b
    ice_frac: float = 0.056  # Fraction ice vs regolith (5.6% Colaprete 2010)
    heat_frac: float = 0.45  # Fraction of ballistic ke used in heating vs mixing
    basin_ejecta_temp_warm: bool = False  # Use warm ejecta temperature for basin ice
    basin_ejecta_temp_init_cold: float = 260  # [K] Initial ejecta temperature, present-cold-Moon (Fernandes & Artemieva 2012)
    basin_ejecta_temp_init_warm: float = 420  # [K] Initial ejecta temperature, ancient-warm-Moon (Fernandes & Artemieva 2012)

    # Thermal module
    specific_heat_coeffs: tuple = (-3.6125, 2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9)  # c0 to c4 (Hayne et al., 2017)

    # Secondary crater scaling (Singer et al, 2020)
    ## Regression values from Table 2 (Singer et al., 2020)
    kepler_regime: tuple = (18e3, 60e3)  # [m] diameter
    kepler_diam: float = 31e3  # [m]
    kepler_a: float = 5.1 # Kepler a value for secondary scaling law from Singer et al. 2020 [km]
    kepler_b: float = -0.33 # ± 0.10 Kepler b value for secondary scaling law from Singer et al. 2020 [km]
    copernicus_regime: tuple = (60e3, 300e3)  # [m] diameter
    copernicus_diam: float = 93e3  # [m]
    copernicus_a: float = 1.2e2 # Copernicus a value for secondary scaling law from Singer et al. 2020 [km]
    copernicus_b: float = -0.68 # ± 0.05 Copernicus b value for secondary scaling law from Singer et al. 2020 [km]
    orientale_regime: tuple = (300e3, 2500e3)  # [m] diameter
    orientale_diam: float = 660e3  # [m]
    orientale_a: float = 1.8e4 # Orientale a value for secondary scaling law from Singer et al. 2020 [km]
    orientale_b: float = -0.95 # ± 0.17 Orientale b value for secondary scaling law from Singer et al. 2020 [km]
    
    # Impact ice module
    mm_mass_rate: float = 1e6  # [kg/yr], lunar micrometeorite flux (Grun et al. 2011)
    ctype_frac: float = 0.36  # 36% of impactors are c-type (Jedicke et al., 2018)
    ctype_hydrated: float = 2/3  # 2/3 of c-types are hydrated (Rivkin, 2012)
    hydrated_wt_pct: float = 0.1  # impactors wt% H2O (Cannon 2020)
    impact_mass_retained: float = 0.165  # asteroid mass retained in impact (Ong et al., 2011)
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

    # Comet constants
    impact_speed_comet: bool = False  # Use comet impact speed distribution
    comet_ast_frac: float = 0.1  # 5-17% TODO: check citation (Joy et al 2012) 
    comet_density: float = 1300  # [kg/m^3]
    comet_hydrated_wt_pct: float = 0.5  # 50% of comet mass is hydrated
    comet_mass_retained: float = 0.065  # asteroid mass retained (Ong et al., 2011)
    halley_to_oort_ratio: float = 3  # N_Halley / N_Oort
    halley_mean_speed: float = 20e3  # [m/s] (Chyba et al., 1994; Ong et al., 2011)
    halley_sd_speed: float = 5e3  # [m/s]
    oort_mean_speed: float = 54e3  # [m/s] (Jeffers et al., 2001; Ong et al., 2011)
    oort_sd_speed: float = 5e3  # [m/s]

    # Volcanic ice module
    volc_mode: str = 'Head'  # ['Head', 'NK']

    # Head et al. (2020) mode (volc_mode == 'Head')
    volc_early: tuple = (4e9, 3e9)  # [yrs]
    volc_late: tuple = (3e9, 2e9)  # [yrs]
    volc_early_pct: float = 0.75  # 75%
    volc_late_pct: float = 0.25  # 25%
    volc_total_vol: float = 1e7 * 1e9 * 1e-9 # [m^3 yr^-1] basalt
    volc_ice_ppm: float = 10  # [ppm] 10 ppm H2O
    volc_magma_density: float = 3000  # [kg/m^3]

    # Needham & Kring (2017) mode (volc_mode == 'NK')
    nk_species: str = 'min_h2o'  # volcanic species, must be in volc_cols
    nk_cols: tuple = (
        'time', 'tot_vol', 'sphere_mass', 'min_co', 'max_co', 'min_h2o', 
        'max_h2o', 'min_h', 'max_h', 'min_s', 'max_s', 'min_sum', 'max_sum',
        'min_psurf', 'max_psurf', 'min_atm_loss', 'max_atm_loss'
    )

    # Solar Wind module
    solar_wind_mode: str = 'Benna'  # ['Benna', 'Lucey-Hurley']
    faint_young_sun: bool = True  # use faint young sun (Bahcall et al., 2001)

    # Impact gardening module Costello et al. (2020)
    overturn_depth_present: float = 0.2  # [m] Present day overturn depth for 1 overturn by secondaries at 99%
    overturn_ancient_slope: float = 1.6e-9  # [m/yr] Slope of overturn depth at higher early impact flux
    overturn_ancient_t0: float = 3e9  # [yrs] Time to start applying higher impact flux

    def __post_init__(self):
        """Set paths, model defaults and raise error if invalid type supplied."""
        setattr(self, 'seed', _get_random_seed(self))
        setattr(self, 'modelpath', _get_modelpath(self))
        setattr(self, 'datapath', _get_datapath(self))
        setattr(self, 'outpath', _get_outpath(self))
        setattr(self, 'figspath', _get_figspath(self))
        _set_mode_defaults(self, self.mode)
        _set_coldtrap_defaults(self, self.pole, self.ice_species)
        for cfg_field in fields(self):
            _make_paths_absolute(self, cfg_field, self.datapath, self.outpath)
            _enforce_dataclass_type(self, cfg_field)


    @property
    def version(self) -> str:
        return self._version


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
        for k in MODE_DEFAULTS['cannon'].keys():
            del ddict[k]
        for k, v in ddict.copy().items():
            if 'path' in k:
                ddict[k] = ''
            elif k[-3:] == '_in' in k[-4:] == '_out':
                ddict[k] = path.basename(v)
        self.to_py(outpath, self.to_string(ddict))




# Config helper functions
def _get_random_seed(cfg):
    """Return random_seed in (1, 99999) if not set in cfg."""
    try:
        seed = int(cfg.seed)
    except (TypeError, ValueError):
        seed = 0
    if not seed:
        seed = np.random.randint(1, 99999)
    return seed


def _set_mode_defaults(cfg, mode, defaults=MODE_DEFAULTS):
    """Set flags in config object for mode ['cannon', 'moonpies']."""
    for param in defaults[mode]:
        if getattr(cfg, param) is None:
            setattr(cfg, param, defaults[mode][param])


def _set_coldtrap_defaults(cfg, pole, ice_species, defaults=COLDTRAP_DEFAULTS):
    """
    Set coldtrap defaults for pole ['n', 's'] and species ['H2O', 'CO2'].
    """
    for param in defaults[pole]:
        if getattr(cfg, param) is None:
            setattr(cfg, param, defaults[pole][param])
    if getattr(cfg, 'coldtrap_max_temp') is None:
        setattr(cfg, 'coldtrap_max_temp', defaults['coldtrap_max_temp'][ice_species]) 
    if getattr(cfg, 'coldtrap_area') is None:
        setattr(cfg, 'coldtrap_area', defaults['coldtrap_area'][pole][ice_species]) 


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
        figspath = path.abspath(path.join(modelpath, '..', 'figures')) + sep
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
        # Default outpath is datapath/out/yymmdd_runname/seed/
        datapath = cfg.datapath
        run_dir = f'{cfg.run_date}_{cfg.run_name}'
        outpath = path.join(datapath, 'out', run_dir, run_seed)
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
    if field.name[-3:] == '_in':
        newpath = path.join(datapath, value)
    elif field.name[-4:] == '_out':
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
