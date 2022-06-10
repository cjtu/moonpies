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
    help="random seed for this run - superceeds seed in config file",
)
parser.add_argument(
    "--cfg", "-c", nargs="?", type=str, help="path to custom my_config.py"
)
args = parser.parse_args()

# Get Cfg from file if provided with seed if provided
# else return the default Cfg()
cfg = default_config.read_custom_cfg(args.cfg, args.seed)

# Run model with chosen config options
mp.main(cfg)