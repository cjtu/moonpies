"""MoonPIES: Moon Polar Ice and Ejecta Stratigraphy module
Main module interface
Date: 08/06/21
:Authors: K.R. Frizzell, K.M. Luchsinger, A. Madera, T.G. Paladino and C.J. Tai Udovicic
Acknowledgements: Translated & extended MATLAB model by Cannon et al. (2020).
"""
import argparse
from pathlib import Path
from moonpies import config
if __name__ == '__main__':
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
    parser.add_argument(
        "--resume", default=False, action='store_true', help="skip if out_path dir already exists"
    )
    args = parser.parse_args()

    # Get Cfg from file (if provided), overwrite seed (if provided), else default
    cfg = config.read_custom_cfg(args.cfg, args.seed)

    if args.resume and Path(cfg.out_path).exists():
        print(f"Skipping run with seed {cfg.seed}: {cfg.out_path} exists.")

    else:
        # Run model with chosen config options
        from moonpies import moonpies as mp
        mp.main(cfg)
