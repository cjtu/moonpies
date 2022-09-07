"""
Command-line interface for running MoonPIES.

Usage:
    moonpies [options] [SEED]

Options:
    --cfg PATH  Path to custom config.py
    --resume    Skip if out_path already exists

Seed:
    Random seed for this run (overwrites seed in config file). If not given,
    a random seed will be chosen. Must be in [1, 99999].

Examples:
    moonpies
    moonpies 12345
    moonpies --cfg my_config.py 54321
"""
import argparse
from pathlib import Path
from moonpies import config

def run():
    """Command-line interface for running MoonPIES."""
    # Get optional random seed and cfg file from cmd-line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "seed",
        type=int,
        nargs="?",
        help="random seed for this run (overwrites seed in config file)",
    )
    parser.add_argument(
        "--cfg", "-c", nargs="?", type=str, help="path to custom config.py"
    )
    parser.add_argument(
        "--resume", default=False, action='store_true', 
        help="skip if out_path dir already exists"
    )
    args = parser.parse_args()

    # Get Cfg from file (if provided), overwrite seed (if provided), else default
    cfg = config.read_custom_cfg(args.cfg, args.seed)

    if args.resume and Path(cfg.out_path).exists():
        print(f"Skipping run with seed {cfg.seed}: {cfg.out_path} exists.")
        quit()

    # Run model with chosen config options
    from moonpies import moonpies as mp
    _ = mp.main(cfg)

if __name__ == '__main__':
    run()
