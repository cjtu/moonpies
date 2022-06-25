# Aggregate Monte Carlo runs into a single csv
import sys
import pickle
from pathlib import Path
from multiprocessing import Process
import pandas as pd
from moonpies import moonpies as mp
from moonpies import default_config

# DATEDIR = "~/moonpies/data/out/yymmdd/""

def read_agg_dfs(datedir, flatten=False):
    """Return aggregated layers and runs DataFrames."""
    layers = pd.read_csv(datedir / 'layers.csv', header=[0, 1, 2])
    runs = pd.read_csv(datedir / 'runs.csv', header=[0, 1, 2])
    if flatten:
        layers = flatten_agg_df(layers)
        runs = flatten_agg_df(runs)
    return layers, runs

def flatten_agg_df(df):
    """Flatten aggregated dataframe into a single indexed dataframe."""
    dflat = df.stack(level=[0, 1]).reset_index(level=0, drop=True).reset_index()
    dflat = dflat.rename(columns={'level_0': 'coldtrap', 'level_1': 'runs'})
    return dflat

def binary_runs(df, yes='mpies', rename='bsed'):
    """Replace runs with binary Yes / No."""
    df['runs'] = df['runs'] == yes
    df['runs'] = df['runs'].replace({True: 'Yes', False: 'No'})
    if rename:
        df.rename(columns={'runs': rename}, inplace=True)
    return df

def get_coldtrap_age(coldtrap, coldtrap_csv):
    """
    Return coldtrap age given a run config file.
    """
    mp.clear_cache()
    fcfg = coldtrap_csv.parent.joinpath('run_config_mpies.py')
    cfg = default_config.read_custom_cfg(fcfg)
    craters = mp.get_crater_basin_list(cfg=cfg, rng=mp.get_rng(cfg))
    return craters[craters.cname==coldtrap].age.values[0]


def agg_coldtrap(coldtrap, datedir, outdir):
    """
    Aggregate all Monte Carlo runs for a given coldtrap as ice layers and 
    depths, and total run ice.
    """
    runs = [p for p in Path(datedir).glob('*') if p.is_dir() and p != outdir]

    data = {'layers': {}, 'runs': {}}
    for rundir in runs:
        csvs = Path(rundir).rglob(f'strat_{coldtrap}.csv')
        ices, depths, times,  = [], [], []
        ice_tot, ice_6m, ice_10m, ice_100m, maxdepth = [], [], [], [], []
        for i, csv in enumerate(csvs):
            # Get crater age for this run to exclude older layers
            age = get_coldtrap_age(coldtrap, csv)
            
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
            if i % 1000 == 0 and i > 0:
                print(f'{coldtrap} {rundir.name} CSV {i}')
        data['layers'][rundir.name] = {
            'ice': ices,
            'depth': depths,
            'time': times,
        }
        data['runs'][rundir.name] = {
            'total ice': ice_tot,
            'total ice 6m': ice_6m,
            'total ice 10m': ice_10m,
            'total ice 100m': ice_100m,
            'max depth': maxdepth
        }

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
    COLDTRAPS = default_config.Cfg().coldtrap_names

    DEBUG = False
    if DEBUG:
        DATEDIR = Path('/home/ctaiudovicic/projects/moonpies/data/out/220610/')
        TMPDIR = DATEDIR / 'tmp'  
        for COLDTRAP in COLDTRAPS[2:]:
            agg_coldtrap(COLDTRAP, DATEDIR, TMPDIR)
        quit()
    
    # Read DATEDIR from command line
    DATEDIR = Path(sys.argv[1])
    TMPDIR = DATEDIR / 'tmp'
    aggregate(COLDTRAPS, TMPDIR, DATEDIR)
    