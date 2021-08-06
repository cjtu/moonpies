"""MoonPIES: Moon Polar Ice and Ejecta Stratigraphy module
Main module interface
Date: 08/06/21
:Authors: K.R. Frizzell, K.M. Luchsinger, A. Madera, T.G. Paladino and C.J. Tai Udovicic
Acknowledgements: Translated & extended MATLAB model by Cannon et al. (2020).
"""
import argparse
from moonpies import moonpies as mp
from moonpies import default_config

# Get optional random seed and cfg file from cmd-line args
parser = argparse.ArgumentParser()
parser.add_argument(
    "seed",
    type=int,
    nargs="?",
    help="random seed for this run - superceeds cfg.seed",
)
parser.add_argument(
    "--cfg", "-c", nargs="?", type=str, help="path to custom my_config.py"
)
args = parser.parse_args()

# Use custom cfg from file if provided, else return default_config.Cfg
cfg_dict = default_config.read_custom_cfg(args.cfg)

# If seed given, it takes precedence over seed set in custom cfg
if args.seed is not None:
    cfg_dict["seed"] = args.seed
custom_cfg = default_config.from_dict(cfg_dict)

# Run model with chosen config options
mp.main(custom_cfg)