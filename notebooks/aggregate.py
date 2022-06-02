# Aggregate Monte Carlo runs into a single csv
import sys
import pickle
from pathlib import Path
from multiprocessing import Process, Queue
import pandas as pd
from moonpies import moonpies as mp
from moonpies import default_config

DATEDIR = Path(sys.argv[1])  # data/out/yymmdd/
# DATEDIR = Path('/home/ctaiudovicic/projects/moonpies/data/out/220601/')
TMPDIR = DATEDIR / 'tmp'
OUTDIR = DATEDIR.parent
RUN_NAMES = {
    'mpies': 'Ballistic Sedimentation',
    'no_bsed': 'No Ballistic Sedimentation'
}

# Set ice columns
COLDTRAPS = default_config.Cfg().coldtrap_names

def get_coldtrap_age(coldtrap, fcfg):
    """
    Return coldtrap age given a run config file.
    """
    mp.clear_cache()
    cfg = default_config.Cfg(**default_config.read_custom_cfg(fcfg))
    craters = mp.get_crater_list(cfg=cfg, rng=mp.get_rng(cfg))
    return craters[craters.cname==coldtrap].age.values[0]


def agg_coldtrap(coldtrap, outdir=TMPDIR):
    """
    Aggregate all Monte Carlo runs for a given coldtrap as ice layers and 
    depths, and total run ice.
    """
    data = {}
    for run_name, isbsed in RUN_NAMES.items():
        rundir = DATEDIR / run_name
        csvs = Path(rundir).rglob(f'strat_{coldtrap}.csv')
        ices, depths, times,  = [], [], []
        ice_tot, ice_6m, ice_10m, ice_100m, maxdepth = [], [], [], [], []
        for i, csv in enumerate(csvs):
            # Get crater age for this run to exclude older layers
            fcfg = csv.parent.joinpath('run_config_mpies.py')
            age = get_coldtrap_age(coldtrap, fcfg)
            
            # Pull ices and depths after coldtrap formed
            ice, time, depth = pd.read_csv(csv, usecols=[0, 3, 4]).values.T
            if (time<=age).sum() == 0:  # no layers after coldtrap formed, skip
                continue
            ices.extend(ice[time<=age])
            depths.extend(depth[time<=age])
            times.extend(time[time<=age])
            ice_tot.append(ice[time<=age].sum())
            ice_6m.append(ice[(time<=age) & (depth < 6)].sum())
            ice_10m.append(ice[(time<=age) & (depth < 10)].sum())
            ice_100m.append(ice[(time<=age) & (depth < 100)].sum())
            maxdepth.append(depth[time<=age].max())
            if i % 1000 == 0:
                print(f'{coldtrap} {isbsed} CSV {i}')
        data[isbsed] = (ices, depths, times, ice_tot, ice_6m, ice_10m, ice_100m, maxdepth)

    # Make TMPDIR if it doesn't exist
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fout = outdir / f"{coldtrap}.pickle"
    with open(fout, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickles_to_csv(tmpdir=TMPDIR, outdir=DATEDIR):
    """
    Read aggregated pickles from tmpdir and write csvs to outdir.
    """
    pickles = Path(tmpdir).rglob('*.pickle')

    layers = {}
    runs = {}
    for p in pickles:
        coldtrap = p.stem
        with open(p, 'rb') as f:
            data = pickle.load(f)
        for isbsed, d in data.items():
            ices, depths, times, ice_tot, ice_6m, ice_10m, ice_100m, maxdepth = d
            layers[(coldtrap, isbsed, 'ice')] = pd.Series(ices, dtype="float32")
            layers[(coldtrap, isbsed, 'depth')] = pd.Series(depths, dtype="float32")
            layers[(coldtrap, isbsed, 'time')] = pd.Series(times, dtype="float32")
            runs[(coldtrap, isbsed, 'total ice')] = pd.Series(ice_tot, dtype="float32")
            runs[(coldtrap, isbsed, 'total ice 6m')] = pd.Series(ice_6m, dtype="float32")
            runs[(coldtrap, isbsed, 'total ice 10m')] = pd.Series(ice_10m, dtype="float32")
            runs[(coldtrap, isbsed, 'total ice 100m')] = pd.Series(ice_100m, dtype="float32")
            runs[(coldtrap, isbsed, 'max depth')] = pd.Series(maxdepth, dtype="float32")
        
    layers_df = pd.DataFrame(layers)
    runs_df = pd.DataFrame(runs)
    layers_df.to_csv(outdir / 'layers.csv', index=False)
    runs_df.to_csv(outdir / 'runs.csv', index=False)

if __name__ == '__main__':
    # for coldtrap in COLDTRAPS[2:]:
        # agg_coldtrap(coldtrap)

    # Run agg_coldtrap in parallel (10 mins for 10k runs)
    queue = Queue()
    processes = [
        Process(target=agg_coldtrap, args=(coldtrap,)) for coldtrap in COLDTRAPS
    ]
    print('Starting processes...')
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
    print('Done aggregating. Converting to csv...')
    
    # Convert pickles to csv (30 secs for 10k runs)
    pickles_to_csv()
    print('Done')
