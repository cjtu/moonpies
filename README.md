# MoonPIES: Moon Polar Ice and Ejecta Stratigraphy

Welcome to the *Moon Polar Ice and Ejecta Stratigraphy* (MoonPIES) model. 

**Note:** This model is not yet peer-reviewed and comes with no warranties. It should not be considered "stable". Ongoing work to improve the documentation and usability of the model may result in backwards-incompatible changes. Please direct bug reports or code feedback to the GitHub [issues board](https://github.com/cjtu/moonpies/issues) or general inquiries to Christian at [cjtu@nau.edu](mailto:cjtu@nau.edu).

## Motivation

MoonPIES models ice and ejecta at depth below lunar polar cold traps. With the imminent return of humans to the Moon through the NASA Artemis program, it is crucial to predict where we expect to find ice, a possibly invaluable lunar resource.


## Installing MoonPIES

The easiest way to get MoonPIES is with pip:

```python
pip install moonpies
```

It is currently tested on Python version 3.8+ for Windows, OS X and Linux.

To install for development, you will require [Poetry](https://python-poetry.org/). Fork or clone this repository and then from the main moonpies folder, install the dev environment with:

```python
poetry install
```

The environment can then be activated in the shell with `poetry shell` (see [poetry docs](https://python-poetry.org/docs/cli/) for more info).

## Running the model

The MoonPIES model can be run directly from the terminal / command line with the `moonpies` command. Run `moonpies --help` for options.

### Configuring a run

MoonPIES functionality is easy to tweak by specifying any of its large list of input parameters (see documentation). A configuration file can be specified as a `.py` file containing a single dictionary. For example, to change the output directory of a run, create a file called `myconfig.py` containing:

```python
{
    'out_path': '~/Downloads/'
}
```

And supply the config file when running the model:

```bash
moonpies --cfg myconfig.py
```

See documentation for full list of parameters that can be supplied in a `config.py` file.

### Random seeds

MoonPIES is designed to be reproducable when given the same random seed and input parameters (on a compatible version). By default, MoonPIES will choose a random seed in [1, 99999]. Specify a particular seed with:

```bash
moonpies 1958
```

### Outputs

MoonPIES outputs are saved by today's date, the run name, and the random seed (e.g. `out/yymmdd/run/#####/`, where `#####` is the 5-digit random seed used. For example, a seed of 1958 will produce:

```bash
out/
|- yymmdd/
|  |- mpies/
|  |  |- 01958/
|  |  |  |- ej_columns_mpies.csv
|  |  |  |- ice_columns_mpies.csv
|  |  |  |- config_mpies.py
|  |  |  |- strat_Amundsen.csv
|  |  |  |- strat_Cabeus B.csv
|  |  |  |- strat_Cabeus.csv
|  |  |  |- ...
```

The output directory will contain a `config_<run_name>.py` file which will reproduce the outputs if supplied as a config file to MoonPIES. Resulting stratigraphy columns for each cold trap are contained within the `strat_...` CSV files. Two additional CSVs with ejecta and ice columns over time show the raw model output (before outputs are collapsed into stratigraphic sequences).

**Note:** Runs with the same run name and random seed will overwrite one another. When tweaking config parameters, remember to specify a new `run_name` to ensure a unique output directory.

### Using MoonPIES in Python code

MoonPIES can be run directly from Python by importing the `moonpies` module and calling the `main()` function:

```Python
import moonpies
model_out = moonpies.main()
```

To specify custom configuration options, create a custom `Cfg` object provided by `config.py` and pass it to `moonpies.main()`. Any parameter in `config.Cfg()` can be set as an argument like so:

```Python
import config
custom_cfg = config.Cfg(solar_wind_ice=False, out_path='~/Downloads')
cannon_model_out = moonpies.main(custom_cfg)
```

Unspecified arguments will retain their defaults. Consult the full API documentation for a description of all model parameters.

### Note on versioning

As a Monte Carlo model, MoonPIES deals with random variation but is designed to be reproducible such that a particular random seed will produce the same set of random outcomes in the model. MoonPIES uses semantic versioning (e.g. major.minor.patch). Major version changes can include API-breaking changes, minor version changes will not break the API (but may break random seed reproducibility), while patch version change should preserve both the API and random seed reproducibility.

## Monte Carlo method

MoonPIES is a Monte Carlo model, meaning outputs can vary significantly from run to run. Therefore, MoonPIES should be run many times (with different random seeds) to build statistical confidence in the possible stratigraphy columns.

### Running with gnuparallel (Linux/Mac/WSL only)

To quickly process many MoonPIES runs in parallel, one can use [GNU parallel](https://www.gnu.org/software/parallel/) which is available from many UNIX package managers, e.g.:

```bash
apt install parallel  # Ubuntu / WSL
brew install parallel  # MacOS
```

**Note:** Not tested on Windows. On MacOS, requires homebrew first (see [brew.sh](https://brew.sh/)).

Now, many iterations of the model may be run in parallel. To spawn 100 runs:

`seq 100 | parallel -P-1 moonpies`

This example will start 1000 runs of MoonPIES, each with a unique random seed and output directory so that no data is overwritten. To configure your `parallel` runs:

- The number of runs is given by the `seq N` parameter (for help see [seq](https://www.unix.com/man-page/osx/1/seq/)).
- By default, `parallel` will use all available cores on your system. Specifying `-P-1` instructs GNU parallel to use all cores except one (`P-2` would use all cores except 2, etc).

## Plotting outputs

`MoonPIES` provides some functions to help visualize model outputs...

*Coming soon!*

## Authors

This model was produced by C. J. Tai Udovicic, K. Frizzell, K. Luchsinger, A. Madera, and T. Paladino with input from M. Kopp, M. Meier, R. Patterson, F. Wroblewski, G. Kodikara, and D. Kring.

## License and Citations

This code is made available under the [MIT license](https://choosealicense.com/licenses/mit/) which allows warranty-free use with proper citation. The model can be cited as:

> Authors et al. (2021). Title.

## Acknowledgements

This model was produced during the 2021 LPI Exploration Science Summer Intern Program which was supported by funding from the Lunar and Planetary Institute ([LPI](https://lpi.usra.edu)) and the Center for Lunar Science and Exploration ([CLSE](https://sservi.nasa.gov/?team=center-lunar-science-and-exploration)) node of the NASA Solar System Exploration Research Virtual Institute ([SSERVI](https://sservi.nasa.gov/)).
