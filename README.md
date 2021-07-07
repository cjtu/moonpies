# Moon Polar Ice and Ejecta Stratigraphy Model

Welcome to the *MoonPIES* model documentation. This will outline what the model does and how to run it.

## Motivation

This model investigates ice preserved at depth in the lunar South polar region. A recent model of ice ingress and egress from cold traps within permanently shadowed regions suggested that "gigaton ice deposits" may be preserved at depth (Cannon et al. 2020). Here, we reproduce the Cannon et al. 2020 model (hereafter *Cannon model*) and extend it to include the effects of *ballistic sedimentation*, update its treatment of *impact gardening*, and improve constraints on the largest potentially icy impactors which appear to deliver the majority of the lunar polar ice observed in the Cannon model.

## Running the model

<!-- 
# From command line (requires python, numpy, pandas)
     <seed> is any integer (default 0) and makes run deterministic
     change seed between runs to get different randomness

python mixing.py <seed>

# From Jupyter, make sure to os.chdir to location of this file:
    use import mixing to run any function in this file e.g. mixing.main()

import os
os.chdir('/home/cjtu/projects/moonpies/moonpies')
import mixing as mm
ej_cols, ice_cols, run_meta = mixing.main()

# Run with gnuparallel (6 s/run normal, 0.35 s/run 48 cores)
    parallel -P-1 uses all cores except 1

conda activate moonpies
seq 10000 | parallel -P-1 python mixing.py

# Code Profiling (pip install snakeviz)

python -m cProfile -o mixing.prof mixing.py
snakeviz mixing.prof -->

## Model workflow

The model is divided into two primary sections: *Setup* and *Main loop*. The Setup phase prepares all stratigraphy columns, while the Main Loop steps through time to add and remove ice from each column.

### Setup

The setup phase initializes all stratigraphy columns and also pre-computes any data that is unchanged in the Main Loop.

1. `read_crater_list()`: Import list of large craters near the pole.
2. `randomize_crater_ages()`: Randomly vary age of craters within their error bars.
3. `get_ejecta_thickness_matrix()`: Pre-compute ejecta thickness at each point on the model grid vs. time (3D array: $Grid X \times Grid Y \times Time$).
4. `get_volcanic_ice()`: Pre-compute ice-delivery by volcanic outgassing at each time (1D array: Time).
5. `get_ballistic_sed_matrix()`: Coming soon!
6. `init_strat_columns()`: Initialize empty ice columns and their associated ejecta thickness columns over time (1D arrays: Time)

### Main Loop

The Main Loop steps through model time from the past to the present and accumulates ice in each strat column.

1. `new_ice_mass()`: Compute the total mass of ice delivered to the polar region in this timestep.
2. `get_ice_thickness()`: Convert mass of ice to 1D thickness added to each strat column.
3. `update_ice_cols()`: Apply ice addition and removal processes for this timestep (see below)

The treatment of ice in the main loop depends on the mode of the model.

**MoonPIES mode**: If `mode == 'moonpies'`, then `update_ice_cols()` will do the following:

a. *Coming soon*: If a crater was formed at this timestep, apply ballistic sedimentation effects to all ice columns based on their distances from the impact.
b. Add new ice thickness to each strat column.
c. Remove ice to a gardening depth determined by the current model time using the Costello et al. (2020) model scaled by the historical impact flux (Ivanov et al. 2000).

**Cannon mode**: If `mode == 'cannon'`, then `update_ice_cols()` will do the following:

b. Add new ice thickness to each strat column.
c. Remove ice to a constant depth of 10 cm at all times using the method of Cannon et al. (2020)

When run in *Cannon mode*, our model outputs yield similar ice thicknesses over time as those published in Cannon et al. (2020).
