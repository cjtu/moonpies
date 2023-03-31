# Quickstart Tutorial

## Installation

MoonPIES is available on PyPI and can be installed with pip:

```bash
pip install moonpies
```

Alternatively, you can install the latest development version by forking the main GitHub repository, see the README for more details [here](github.com/cjtu/moonpies).


## Running Moonpies


Run MoonPIES from the terminal with the moonpies command. Check the version installed with:

```bash
moonpies -v
```

To run one model simulation with default parameters and a random seed, just type `moonpies`:

```bash
moonpies
```

To choose a custom random seed, specify it as a number from 1 to 99999. Random seeds help with reproducibility, ensuring that model outputs will be consistnet if run with the same parameters and seed.

```bash
moonpies 12345
```

## Plotting Moonpies Results

Some Python functions are provided to help visualize model results. After running the model, it should have printed the output directory. In Python, use that output directory with the `plot_stratigraphy()` function from the `moonpies.plotting` module:

```Python
from moonpies.plotting import plot_stratigraphy
plot_stratigraphy('path/to/output/directory')
```

More complicated plotting functions are available in the [notebooks](https://github.com/cjtu/moonpies/tree/main/notebooks) gallery, including publication quality figures with full code available in [notebooks/figure_scripts/plot_figs.py](https://github.com/cjtu/moonpies/blob/main/notebooks/figure_scripts/). Some plots require the MoonPIES model to be run many times and then for the results to be aggregated into a single DataFrame with `aggregate.py`, also found in `notebooks/figure_scripts/`. See the following notebooks for examples:

- [2023_JGR_figs.ipynb](https://github.com/cjtu/moonpies/blob/main/notebooks/2023_JGR_figs.ipynb)
- [2023_JGR_sup_figs.ipynb](https://github.com/cjtu/moonpies/blob/main/notebooks/2023_JGR_sup_figs.ipynb)


## Configuring Moonpies


MoonPIES parameters can be customized with a simple Python configuration file. To build a custom configuration file, make a `.py` file containing a single dictionary where the keys are parameter names of the `Cfg` class found in `moonpies/config.py` and the values are the desired values (note the value type must match those in config.py). It is recommmended to choose a descriptive `run_name` to associate with your run. For example:

```Python
# File named "my_config.py"
{
   'run_name': 'no_solarwind_dense_impactors',
   'solar_wind_ice': False,
   'impactor_density': 2000  # [kg m^-3]
}
```

Then, run MoonPIES with the `-c` or `--cfg` flag:

```bash
moonpies --cfg my_config.py
```

See the `moonpies.config.Cfg` class for full list of parameters and their default values [here](https://github.com/cjtu/moonpies/blob/main/moonpies/config.py).
