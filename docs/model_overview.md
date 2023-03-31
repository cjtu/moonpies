# How MoonPIES works

This is an overview of the main functions in the `MoonPIES` model. The model is found in `moonpies.py` and can be run by calling the `main()` function. The `main()` function is divided into three parts: **Setup**, **Update Loop**, and **Output**. 

## Setup

The setup phase initializes all stratigraphy columns and also pre-computes ice delivery in each timestep and potential ice loss for each stratigraphy column in each timestep.

1. `get_time_array()`: Initialize model time array (default: 4.25 Ga to 0 Ga with 10 Ma timesteps).
2. `get_crater_basin_list()`: Import list of large craters near the pole (and optionally large global basins). Randomizes the ages of the imported craters within their error bars.
3. `init_strat_columns()`: Initialize a dictionary of the ice and ejecta columns of each coldtrap specified in cfg.coldtrap_names.
4. `get_bsed_depth()`: Compute the depth of ballistic sedimentation ice removal (m) for each coldtrap for all timesteps, if a large crater was formed within range during a particular timestep.
5. `get_overturn_time()`: Compute the depth of impact gardening ice removal (m) for all timesteps.


## Update Loop

The Update Loop steps through the model time array, updating ice lost (if any) by each coldtrap at each timestep.

For each timestep in the model, `update_strat_cols()` will:

1. `garden_ice_col()`: Remove ice due to ballistic sedimentation if a large impact formed near enough to any of the cold traps this timestep. Surface ejecta will partially preserve underlying ice layers.
2. Ice deposition: Pre-computed, but occurs first in a given timestep.
3. `remove_ice_overturn()`: Remove ice up to the impact gardening depth, if not preserved by overlying ejecta.


## Output

The output phase formats the stratigraphy columns into pandas DataFrames and saves them to CSV files. This phase also saves a copy of the model configuration file to the output directory such that the exact results can be reproduced if this file is passed in when running the model, e.g.  `moonpies --cfg <config_file.py>`.


## Additional info

For more specifics on the MoonPIES model, see the paper (coming soon), or check out the function-by-function [API documentation](api.rst).