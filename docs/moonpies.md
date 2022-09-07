# MoonPIES Source Code

## moonpies.py

```{literalinclude} ../moonpies/moonpies.py
:language: python
:linenos:
```

## config.py

```{literalinclude} ../moonpies/config.py
:language: python
:linenos:
```

## Model details

Here, we will give an overview of the main functions in the `MoonPIES` model.

The model is divided into two primary sections: **Setup** and **Main loop**. The Setup phase prepares all stratigraphy columns, while the Main Loop steps through time to add and remove ice from each column.

### Setup

The setup phase initializes all stratigraphy columns and also pre-computes any data that is unchanged in the Main Loop.

1. `get_time_arr()`: Initialize model time array (default: 4.25 Ga to 0 Ga with 10 Ma timesteps).
2. `get_crater_list()`: Import list of large craters near the pole (optionally import large global basins also). Randomizes the ages of the imported craters within their error bars.
3. `init_strat_columns()`: Initialize a dictionary of the ice and ejecta columns of each coldtrap in cfg.coldtrap_names.


### Main Loop

The Main Loop steps through the model time array, updating the coldtrap strat_columns at each step.

For each timestep in the model, `update_strat_cols()` will do the following:

1. `get_ballistic_sed_depths()`: Get depth of ballistic sedimentation ice removal (m) for each coldtrap, if a crater was formed nearby this timestep.
2. `get_polar_ice`: Get ice thickness (m) from all sources except volcanics delivered to the polar region this timestep.
3. `get_volcanic_ice_t()`: Get volcanic ice thickness (m) deposited at this timestep.
4. `get_overturn_depth()`: Get the depth of impact gardening ice removal (m) for this timestep.

Once these quantities are computed, we go through each coldtrap to update its ice column:

1. `get_ice_coldtrap()`: Compute the amount of polar_ice that reaches this particular coldtrap
2. `update_ice_col()`: Updates ice column in the following order: ballistic sedimentation removes ice, new ice is added, impact gardening removes ice.

### MoonPIES vs Cannon mode

The flow of the main model depends on whether `cfg.mode=='moonpies'` or `cfg.mode=='cannon'`. The differences between the modes is summarized below:

**MoonPIES mode**: When `mode` is set to `moonpies` (the default), the following model configuration parameters are set:

```Python
'solar_wind_ice': True,
'ballistic_hop_moores': True,
'ejecta_basins': True,
'impact_ice_basins': True,
'impact_ice_comets': True,
'use_volc_dep_effcy': True,
'ballistic_sed': True,
'impact_gardening_costello': True, 
'impact_speed_mean': 17e3,
```

**Cannon mode**: When `mode` is set to `cannon`, the following model configuration parameters are set:

```Python
'solar_wind_ice': False,
'ballistic_hop_moores': False,
'ejecta_basins': False,
'impact_ice_basins': False,
'impact_ice_comets': False,
'use_volc_dep_effcy': False,volc_dep_efficiency
'ballistic_sed': False,
'impact_gardening_costello': False,
'impact_speed_mean': 20e3,
```

In Cannon mode, we only consider modules included in the Cannon et al. (2020) model.

### Cannon mode discrepancies

Although the `mode='cannon'` version of MoonPIES was drawn directly from Cannon et al. (2020), it is unable to reproduce the results of that model, often predicting much lower ice abundances. Some key functions were not included in the Cannon et al. (2020) supplement, limiting a direct comparison between the MoonPIES and Cannon et al. (2020) model.

Modules missing from Cannon et al. (2020) supplement which could be the source of the discrepancy are:

- Crater diameter to impact diameter scaling laws (for MoonPIES we draw scaling laws for each regime from each respective source cited by Cannon et al., 2020)
- Ejecta thickness for each polar crater (we use radial thickness with distance from Kring, 1995)
- Neukum production function version (we implemented both the 1989 and 2001 NPF and default to the 2001 version)

Other than these functions, we have tested our model against the code provided by Cannon et al. (2020) and both produce the same outputs with the same inputs, less the missing functions. We believe that discrepancies in implementation of crater scaling code is the most likely culprit.

A last possibility for the discrepancy is ambiguity of whether the Cannon et al (2020) model considers large complex craters to range in size from 15 km to 300 km (as set in the supplemental ds02.m MATLAB code), or from 15 km to 500 km (as specified in Table S2 of the supplemental information document). We attempted to reproduce supplemental Figure S3 from Cannon et al. (2020) with 10k run ensemble plots for both the 300 km and 500 km maximum crater sizes and neither results in similar ice thicknesses through time.


### Module documentation

*Coming soon!* Here we describe each module in more detail.

<!-- TODO: These are giving too long error, fix -->