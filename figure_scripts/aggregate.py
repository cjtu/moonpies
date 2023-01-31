# Aggregate Monte Carlo runs into a single csv
import sys
import pickle
from pathlib import Path
from multiprocessing import Process
import numpy as np
import pandas as pd
from moonpies import moonpies as mp
from moonpies import config


def read_agg_dfs(datedir, flatten=False, count_runs=False):
    """Return aggregated layers and runs DataFrames."""
    layers = pd.read_csv(datedir / 'layers.csv', header=[0, 1, 2])
    runs = pd.read_csv(datedir / 'runs.csv', header=[0, 1, 2])
    nruns = len(runs)  # before flattening
    if flatten:
        layers = flatten_agg_df(layers)
        runs = flatten_agg_df(runs)
    if count_runs:
        return layers, runs, nruns
    else:
        return layers, runs

def flatten_agg_df(df):
    """Flatten aggregated dataframe into a single indexed dataframe."""
    dflat = df.stack(level=[0, 1]).reset_index(level=0, drop=True).reset_index()
    dflat = dflat.rename(columns={'level_0': 'coldtrap', 'level_1': 'runs'})
    return dflat

def rename_runs(df, run_names, new_names):
    """Subset aggregated df to runs in run_names and rename to new_names."""
    df = df[df.runs.isin(run_names)]
    for run_name, new_name in zip(run_names, new_names):
        df.loc[df.runs==run_name, 'runs'] = new_name
    return df


def agg_coldtrap(coldtrap, datedir, outdir, depths=(1, 3, 6, 10, 100)):
    """
    Aggregate all Monte Carlo runs for a given coldtrap as ice layers and 
    depths, and total run ice.
    """
    runs = [p for p in Path(datedir).glob('*') if p.is_dir() and p != outdir]

    data = {'layers': {}, 'runs': {}}
    for rundir in runs:
        csvs = list(Path(rundir).rglob(f'strat_{coldtrap}.csv'))
        clayers = {'ice': [], 'depth': [], 'time': []}
        cruns = {f'total ice {d}m': np.zeros(len(csvs)) for d in depths}
        cruns['total ice'] = np.zeros(len(csvs))
        cruns['maxdepth'] = np.zeros(len(csvs))

        for i, csv in enumerate(csvs):
            # Pull ice, timing and depth
            df = pd.read_csv(csv)
            ice, time, depth = df[['ice', 'time', 'depth']].values.T
            age = float(df[df.label == 'Formation age'].time)
            
            # Skip if no layers after coldtrap formed
            if (time<=age).sum() == 0:
                continue
            ice = ice[time<=age]
            depth = depth[time<=age]
            clayers['ice'].extend(ice)
            clayers['depth'].extend(depth)
            clayers['time'].extend(time)
            
            cruns['total ice'][i] = ice.sum()
            cruns['maxdepth'][i] = depth.max()

            # Compute total ice to each depth
            for dmax in depths:
                cruns[f'total ice {dmax}m'][i] = ice[depth<=dmax].sum()
                # Find layer overlapping the dmax, keep portion above dmax
                ind = np.searchsorted(depth[::-1], dmax, side='right') + 1
                if ind <= len(depth):
                    doverlap = dmax - (depth[-ind] - ice[-ind])
                    cruns[f'total ice {dmax}m'][i] += max(0, doverlap)

            if i % 1000 == 0:
                print(f'{coldtrap} {rundir.name} CSV {i}')
        data['layers'][rundir.name] = clayers
        data['runs'][rundir.name] = cruns

    # Make TMPDIR if it doesn't exist
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fout = outdir / f"{coldtrap}.pickle"
    with open(fout, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickles_to_csv(tmpdir, outdir, dtype='float32'):
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
        for run, d in data['layers'].items():
            for stat, arr in d.items():
                layers[(coldtrap, run, stat)] = pd.Series(arr, dtype=dtype)
        for run, d in data['runs'].items():
            for stat, arr in d.items():
                runs[(coldtrap, run, stat)] = pd.Series(arr, dtype=dtype)
    layers_df = pd.DataFrame(layers)
    runs_df = pd.DataFrame(runs)
    layers_df.to_csv(outdir / 'layers.csv', index=False)
    runs_df.to_csv(outdir / 'runs.csv', index=False)


def aggregate(coldtraps, tmpdir, datedir):
    """
    Aggregate all Monte Carlo runs for all coldtraps.
    """
    procs = [Process(target=agg_coldtrap, args=(c, datedir, tmpdir)) for c in coldtraps]
    print(f'Starting {len(procs)} coldtraps...')
    _ = [p.start() for p in procs]
    _ = [p.join() for p in procs]
    print('Done aggregating. Converting to csv...')
    
    # Convert pickles to csv (30 secs for 10k runs)
    pickles_to_csv(tmpdir, datedir)
    print('Done')


if __name__ == '__main__':
    COLDTRAPS = config.Cfg().coldtrap_names

    DEBUG = False
    if DEBUG:
        DATEDIR = Path('/home/ctaiudovicic/projects/moonpies/out/230122/')
        TMPDIR = DATEDIR / 'tmp'  
        for COLDTRAP in COLDTRAPS[2:]:
            agg_coldtrap(COLDTRAP, DATEDIR, TMPDIR)
            break
        quit()
    else:
        # Read DATEDIR from command line
        DATEDIR = Path(sys.argv[1])
        TMPDIR = DATEDIR / 'tmp'
        aggregate(COLDTRAPS, TMPDIR, DATEDIR)
    