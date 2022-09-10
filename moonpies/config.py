"""Configuration for MoonPIES model

Sets all configurable attributes, paths and model options for a MoonPIES run.
See documentation for guide to defining custom config parameters.
"""
import ast
import importlib.resources
import pprint
from os import path, sep, getcwd
from datetime import datetime
from dataclasses import dataclass, fields, field, asdict
import numpy as np
from ._version import __version__
_sudo_setattr = object.__setattr__  # Set attrs of frozen dataclass fields

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
        'ballistic_sed': True,
        'impact_gardening_costello': True,
        'impact_speed_mean': 17e3,  # [m/s] 
    }
}

POLE_DEFAULTS = {
    's': {
        'ballistic_hop_effcy': 0.054,  # Cannon et al. (2020)
        'volc_dep_effcy': 0.26,  # 25.8% Spole (Wilcoski et al., 2021)
        'coldtrap_area_H2O': 1.3e4 * 1e6,  # [m^2], (Williams et al., 2019) (1.7e4 * 1e6 Shorghofer 2020)
        'coldtrap_area_CO2': 104 * 1e6, # [m^2], (<60 K max summer T, Williams et al., 2019)
        'coldtrap_names': (
            'Haworth', 'Shoemaker', 'Faustini', 'Shackleton', 'Slater', 
            'Amundsen', 'Cabeus', 'Sverdrup', 'de Gerlache', "Idel'son L", 
            'Wiechert J', 'Cabeus B'
        ),
    },
    'n': {
        'ballistic_hop_effcy': 0.027,  # Cannon et al. (2020)
        'volc_dep_effcy': 0.15,  # 15.2% Npole (Wilcoski et al., 2021)
        'coldtrap_area_H2O': 5.3e3 * 1e6,  # [m^2], (Williams et al., 2019) (1.7e4 * 1e6 Shorghofer 2020)
        'coldtrap_area_CO2': 0,  # [m^2], TODO: compute
        'coldtrap_names': ('Fibiger', 'Hermite', 'Hermite A', 'Hevesy', 
            'Lovelace', 'Nansen A', 'Nansen F', 'Rozhdestvenskiy U', 
            'Rozhdestvenskiy W', 'Sylvester')
    },
}
@dataclass(frozen=True)
class Cfg:
    """Class to configure a mixing model run."""
    run_name: str = 'moonpies'  # Name of the current run
    version: str = __version__  # Current version of MoonPIES

    # Set in post_init
    seed: int = 0  # Note: Should not be set here - set when model is run only
    run_date: str = field(default='', compare=False)
    run_time: str = field(default='', compare=False)

    # Model output behavior
    verbose: bool = False  # Print info as model is running
    write: bool = True  # Write model outputs to a file (if False, just return)
    write_npy: bool = False  # Write large arrays to files - slow! (age_grid, ej_thickness)
    strat_after_age: bool = True  # Stratigraphy column outputs start at coldtrap age (removes pre-coldtrap layering)
    plot: bool = False  # Save strat column plots - slow!
    
    # Setup Cannon vs MoonPIES config mode and lunar pole
    mode: str = 'moonpies'  # ['moonpies', 'cannon']
    pole: str = 's'  # ['s', 'n'] TODO: only s is currently supported

    # Mode options set in __post_init__ by _set_mode_defaults()
    solar_wind_ice: bool = None
    ballistic_hop_moores: bool = None  # hop_effcy per crater (Moores et al 2016)
    ejecta_basins: bool = None
    impact_ice_basins: bool = None
    impact_ice_comets: bool = None
    use_volc_dep_effcy: bool = None  # Use volc_dep_efficiency rather than ballistic hop for volcanics
    ballistic_sed: bool = None
    impact_gardening_costello: bool = None

    # Pole options set in __post_init__ by _set_pole_defaults()
    ballistic_hop_effcy: float = None
    volc_dep_effcy: float = None
    coldtrap_area_H2O: float = None  # Depends on pole
    coldtrap_area_CO2: float = None  # Depends on pole
    coldtrap_names: tuple = None

    # Paths set in post_init if not given (attr name must end with "_path")
    data_path: str = ''  # path to import data
    out_path: str = ''  # path to save outputs
    figs_path: str = ''  # path to save figures
    
    # Files to import from data_path (attr name must end with "_in")
    crater_csv_in: str = 'crater_list.csv'
    basin_csv_in: str = 'basin_list.csv'
    nk_csv_in: str = 'needham_kring_2017_s3.csv'
    bahcall_csv_in: str = 'bahcall_etal_2001_t2.csv'
    bhop_csv_in: str = 'ballistic_hop_coldtraps.csv'
    bsed_frac_mean_in: str = 'ballistic_sed_frac_melted_mean.csv'
    bsed_frac_std_in: str = 'ballistic_sed_frac_melted_std.csv'
    mplstyle_in: str = '.moonpies.mplstyle'

    # Files to export to out_path (attr name must end with "_out")
    ej_t_csv_out: str = 'ej_columns.csv'
    ice_t_csv_out: str = 'ice_columns.csv'
    config_py_out: str = f'config_{run_name}_v{__version__}.py'
    agegrd_npy_out: str = 'age_grid.npy'
    ejmatrix_npy_out: str = 'ejecta_matrix.npy'

    # Grid and time size and resolution
    dtype = np.float32  # np.float64 (32 should be good for most purposes)
    rtol = 1e-6  # Rounding tolerance for floating point comparisons (mainly for rounding error with float time_arr)
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
    impact_angle: float = 45  # [deg]  average impact angle
    target_density: float = 1500  # [kg m^-3] (Cannon 2020)
    bulk_density: float = 2700  # [kg m^-3] 
    ice_erosion_rate: float = 0.1 # [m], 10 cm / 10 ma (Cannon 2020)
    ej_threshold: float = 4  # [crater radii] Radius of influence of a crater (-1: no threshold)
    basin_ej_threshold: float = 5 # [basin radii] Radius of influence of a basin (-1: no threshold), e.g. Liu et al. (2020)
    ej_thickness_exp: float = -3  # Exponent in ejecta thickness vs. distance (min: -3.5, avg: -3, max: -2.5; Kring 1995)
    thickness_min: float = 1e-3  # [m] minimum thickness to form a layer
    neukum_pf_new: bool = True  # [True=>2001, False=>1983] Neukum production function version (Neukum et al. 2001)
    neukum_pf_a_2001: tuple = (-3.0876, -3.557528, 0.781027, 1.021521, -0.156012, -0.444058, 0.019977, 0.086850, -0.005874, -0.006809, 8.25e-4, 5.54e-5)
    neukum_pf_a_1983: tuple = (-3.0768, -3.6269, 0.4366, 0.7935, 0.0865, -0.2649, -0.0664, 0.0379, 0.0106, -0.0022, -5.18e-4, 3.97e-5)

    # Ice constants
    # coldtrap_max_temp_H2O: float = 110  # [K]
    # coldtrap_max_temp_CO2: float = 60  # [K]
    ice_species = 'H2O'  # ['H2O', 'CO2']  # TODO: currently only H2O supported
    ice_density: float = 934  # [kg m^-3], (Cannon 2020)
    ice_melt_temp: float = 273  # [k]
    ice_latent_heat: float = 334e3  # [j/kg] latent heat of h2o ice

    # Ejecta module
    crater_cols: tuple = ('cname', 'lat', 'lon', 'psr_lat', 'psr_lon', 'diam', 
                          'age', 'age_low','age_upp', 'psr_area', 'age_ref', 
                          'prio', 'notes')

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
    ke_heat_frac: float = 0.45  # Fraction of ballistic ke used in heating vs mixing
    ke_heat_frac_speed: float = 1.45e3  # [m/s] Minimum speed for ke_heat_frac to be applied
    polar_ejecta_temp_init: float = 140  # [K] Typical polar subsurface temperature (Vasavada et al., 1999; Feng et al., 2021)
    basin_ejecta_temp_warm: bool = False  # Use warm ejecta temperature for basin ice
    basin_ejecta_temp_init_cold: float = 260  # [K] Initial ejecta temperature, present-cold-Moon (Fernandes & Artemieva 2012)
    basin_ejecta_temp_init_warm: float = 420  # [K] Initial ejecta temperature, ancient-warm-Moon (Fernandes & Artemieva 2012)
    ejecta_temp_dist_params_cold: tuple = (2.032e+06, -7.658e+03, 2.520e+02)  # quadratic fit params to Fernandes & Artemieva (2012) with x in crater radii
    ejecta_temp_dist_params_warm: tuple = (2.449e+06, -2.384e+04,  4.762e+02)  # quadratic fit params to Fernandes & Artemieva (2012) with x in crater radii

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
    impact_mass_retained: float = 0.165  # asteroid mass retained in impact (Ong et al., 2010)
    brown_c0: float = 1.568  # Brown et al. (2002)
    brown_d0: float = 2.7  # Brown et al. (2002)
    earth_moon_ratio: float = 22.5  # Earth-Moon impact ratio Mazrouei et al. (2019)
    impact_regimes: dict = field(default_factory = lambda: ({
        # regime: (rad_min, rad_max, step, sfd_slope)
        'a': (0, 0.01, None, None),  # micrometeorites (<1 mm)
        'b': (0.01, 3, 1e-4, -3.7),  # small impactors (10 mm - 3 m)
        'c': (100, 1.5e3, 1, -3.82),  # simple craters, steep sfd (100 m - 1.5 km)
        'd': (1.5e3, 15e3, 1e2, -1.8),  # simple craters, shallow sfd (1.5 km - 15 km)
        'e': (15e3, 300e3, 1e3, -1.8),  # complex craters, shallow sfd (15 km - 300 km)
    }), compare=False)  # TODO Cludge: compare=False allows Cfg to be hashable with dict attr. Should make this immutable
    # Comet constants
    is_comet: bool = False  # Use comet properties for impacts
    comet_ast_frac: float = 0.05  # 5-17% (Joy et al 2012) 
    comet_density: float = 600  # [kg/m^3] Comet Shoemaker-Levy 9 (Asphaug and Benz, 1994)
    comet_hydrated_wt_pct: float = 0.5  # 50% of comet mass is hydrated (Whipple, 1950; Ong et al., 2010)
    comet_mass_retained: float = 0.065  # asteroid mass retained (Ong et al., 2010)
    comet_speed_min: float = 10.2e3  # [km/s] minimum lunar impact speed for comet (Ong et al., 2010)
    comet_speed_max: float = 72e3  # [km/s] maximum lunar impact speed for comet (Ong et al., 2010)
    jfc_speed_mean: float = 20e3  # [m/s] (Chyba et al., 1994; Ong et al., 2010)
    jfc_speed_sd: float = 5e3  # [m/s] (Chyba et al., 1994; Ong et al., 2010)
    jfc_frac: float = 0.875  # ~7 JFC : LPC (HPC+OOC) ratio (Carrillo-Sánchez et al., 2016; Pokorný et al., 2019)
    lpc_speed_mean: float = 54e3  # [m/s] (Jeffers et al., 2001; Ong et al., 2010)
    lpc_speed_sd: float = 5e3  # [m/s] (Jeffers et al., 2001; Ong et al., 2010)
    lpc_frac: float = 0.125  # ~7 JFC : LPC (HPC+OOC) ratio (Carrillo-Sánchez et al., 2016; Pokorný et al., 2019)

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
    overturn_depth_present: float = 0.1  # [m] Present day overturn depth for 1 overturn by secondaries at 99%
    overturn_ancient_slope: float = 1.6e-9  # [m/yr] Slope of overturn depth at higher early impact flux
    overturn_ancient_t0: float = 3e9  # [yrs] Time to start applying higher impact flux

    # Private post_init methods
    def _set_seed_time(self, seed):
        """Set seed, date and time"""
        _sudo_setattr(self, 'seed', _get_random_seed(seed))
        _sudo_setattr(self, 'run_date', _get_date())
        _sudo_setattr(self, 'run_time', _get_time())


    def _set_mode_defaults(self, mode):
        """Set flags in config object for mode ['cannon', 'moonpies']."""
        for param in MODE_DEFAULTS[mode]:
            if getattr(self, param) is None:
                _sudo_setattr(self, param, MODE_DEFAULTS[mode][param])


    def _set_pole_defaults(self, pole):
        """Set defaults for lunar pole ['n', 's']."""
        for param in POLE_DEFAULTS[pole]:
            if getattr(self, param) is None:
                _sudo_setattr(self, param, POLE_DEFAULTS[pole][param])

    def _set_path_defaults(self):
        """Set defaults for paths."""
        _sudo_setattr(self, 'data_path', _get_data_path(self.data_path))
        _sudo_setattr(self, 'out_path', _get_out_path(self.out_path, self.seed, self.run_date, self.run_name))
        _sudo_setattr(self, 'figs_path', _get_figs_path(self.figs_path, self.out_path))
    
    
    def _make_paths_absolute(self, data_path, out_path):
        """
        Make all file paths absolute. 
        
        Prepend data_path to all cfg_fields ending with "_in".
        Prepend out_path to all cfg_fields ending with "_out".
        """
        for cfg_field in fields(self):
            key, value = cfg_field.name, getattr(self, cfg_field.name)
            if not isinstance(value, str):
                continue

            # Get new path with data / out path prepended
            newpath = None
            if key.endswith('_in'):
                newpath = path.join(data_path, path.basename(value))
            elif key.endswith('_out'):
                newpath = path.join(out_path, path.basename(value))
            elif key.endswith('_path'):
                newpath = value
            # If value already a valid path, do nothing
            if newpath is not None and not path.exists(value):
                newpath = path.abspath(path.expanduser(newpath))
                _sudo_setattr(self, cfg_field.name, newpath)

    def _enforce_dataclass_types(self):
        """
        Force all dataclass types to their type hint, raise error if invalid.
        
        If typehint is int and value specified in scientific notation (float), 
        converts value to int (e.g., '1e6' -> 1000000).
        """
        for cfg_field in fields(self):
            value = getattr(self, cfg_field.name)
            try:
                if not isinstance(value, cfg_field.type):
                    _sudo_setattr(self, cfg_field.name, cfg_field.type(value))
            except ValueError as e:
                msg = f'Type mismatch for {cfg_field.name} in config: ' 
                msg += f'Expected: {cfg_field.type}. Got: {type(value)}.'
                raise ValueError(msg) from e

    def __post_init__(self):
        """Set paths, model defaults and raise error if invalid type supplied."""
        self._set_seed_time(self.seed)
        self._set_mode_defaults(self.mode)
        self._set_pole_defaults(self.pole)
        self._set_path_defaults()
        self._make_paths_absolute(self.data_path, self.out_path)
        self._enforce_dataclass_types()

    # Public methods
    def to_dict(self):
        """Return dict representation of dataclass."""
        return asdict(self)


    def to_string(self):
        """Return representation of self as dict formatted python string."""
        return _dict2str(self.to_dict())


    def to_py(self, out_path):
        """Write cfg to python file at out_path."""
        _str2py(self.to_string(), out_path)


# Config helper functions
def _get_random_seed(seed):
    """Return random_seed in (1, 99999) if not set in cfg."""
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 0
    if not seed:
        seed = np.random.randint(1, 99999)
    return seed

def _get_date(fmt="%y%m%d"):
    """Return current date as string."""
    return datetime.now().strftime(fmt)

def _get_time(fmt="%H:%M:%S"):
    """Return current time as string."""
    return datetime.now().strftime(fmt)


def _get_data_path(data_path):
    """Return default data_path if not specified in cfg."""
    if data_path != '':
        return data_path
    with importlib.resources.path('moonpies', 'data') as fpath:
            data_path = fpath.as_posix()
    return data_path + sep


def _get_figs_path(figs_path, out_path):
    """Return default figs_path if not specified in cfg."""
    if figs_path != '':
        return figs_path
    figs_path = path.abspath(path.join(out_path, '..', '..', 'figures'))
    return figs_path  + sep


def _get_out_path(out_path, seed, run_date, run_name):
    """Return default out_path if not specified in cfg."""
    run_seed = f'{seed:05d}'
    if out_path != '' and run_seed not in out_path:
        # Prevent overwrites by appending run_seed/ dir to end of out_path
        if path.basename(path.normpath(out_path)).isnumeric():
            # Trim previous seed path if it exists
            out_path = path.dirname(path.normpath(out_path))
        out_path = path.join(out_path, run_seed)
    elif out_path == '':
        # Default out_path is ./out/yymmdd/runname/seed/
        out_path = path.join(getcwd(), 'out', run_date, run_name, run_seed)
    return out_path + sep


def _dict2str(d):
    """Convert dictionary to pretty printed string."""
    return pprint.pformat(d, compact=True, sort_dicts=False)


def _str2py(s, out_path):
    """Write dict representation of dataclass to python file at out_path."""
    with open(out_path, 'w', encoding="utf8") as f:
        f.write(s)
    print(f'Wrote to {out_path}')


def read_custom_cfg(cfg_path=None, seed=None):
    """
    Return Cfg from custom config file at cfg_path. Overwrite seed if given.
    """
    if cfg_path is None:
        cfg_dict = {}
    else:
        with open(path.abspath(cfg_path), 'r', encoding='utf8') as f:
            cfg_dict = ast.literal_eval(f.read())
    if seed is not None:
        cfg_dict['seed'] = seed
    return from_dict(cfg_dict)


def from_dict(cdict):
    """Return Cfg setting all provided values in cdict."""
    return Cfg(**cdict)


def make_default_cfg(out_path=''):
    """Write default cfg file to out_path."""
    if not out_path:
        out_path = path.join(getcwd(), 'myconfig.py')
    ddict = asdict(Cfg())
    del ddict['seed']
    del ddict['run_time']
    del ddict['run_date']
    for k in MODE_DEFAULTS['moonpies']:
        del ddict[k]
    for k, v in ddict.copy().items():
        if 'path' in k:
            ddict[k] = ''
        elif k[-3:] == '_in' or k[-4:] == '_out':
            ddict[k] = path.basename(v)
    _str2py(_dict2str(ddict), out_path)


if __name__ == '__main__':
    make_default_cfg()
