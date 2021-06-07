"""
Main mixing module (06/07/21).

Run with:

import mixing
mixing.main()
# or main(plot=True) to see plot outputs
# call individual functions with, e.g.:
# mixing.read_crater_list()

Fix the file paths below if necessary. Assumes you have folders setup like:

my_folder/
- code/
    - mixing.py
- data/
    - cannon2020_crater_ages.csv
- figs/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
R_MOON = 1737  # [km]
SIMPLE2COMPLEX = 15  # [km]

# Set Paths
FPATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
FIGPATH = os.path.abspath(FPATH + '../figs/') + os.sep

# Crater file to read and columns of csv
CRATER_CSV = os.path.abspath(FPATH + '../data/cannon2020_crater_ages.csv')
COLUMNS = ('cname', 'lat', 'lon', 'diam', 'age', 'age_low', 'age_upp')

# Set options
PLOT = False  # Show plots in main
GRDSIZE = 400  # [pixels]
GRDSTEP = 1  # [km / pixel]
GRD_X, GRD_Y = np.ogrid[-GRDSIZE:GRDSIZE:GRDSTEP, -GRDSIZE:GRDSIZE:GRDSTEP]


# Functions
def main(plot=PLOT):
    """
    Main loop for mixing model. Steps so far:
      1) read_crater_list(): Reads in crater list from CRATER_CSV above.
      2) get_ejecta_distance(): Get 3D array of dist from each crater on grid.
    """
    df = read_crater_list()
    ej_dist = get_ejecta_distances(df)
    if plot:
        plt.imshow(ej_dist[:, :, 0])

    # ToDo:
    # - compute ejecta thicknesses
    # - compute ice mixing / loss from each crater
    # - main loop 
    #   - ice gain / loss at each timestep
    #   - ejecta deposited, ice gain / mixed / loss for each crater
    return ej_dist

def read_crater_list(crater_csv=CRATER_CSV, columns=COLUMNS):
    """
    Return dataframe of craters from path to crater_csv with columns names.

    Computes x, y coordinates from S. Pole with latlon2xy.
    """
    df = pd.read_csv(crater_csv, names=columns, header=0)
    if 'rad' not in df.columns:
        df['rad'] = df['diam'] / 2
    df['x'], df['y'] = latlon2xy(df.lon, df.lat)
    df['dist2pole'] = gc_dist(0, -90, df.lon, df.lat)
    return df


def get_ejecta_distances(df, grd_x=GRD_X, grd_y=GRD_Y):
    """
    Return 3D array shape (len(grd_x), len(grd_y), len(df)) of ejecta distances 
    from each crater in df.

    Distances computed with simple dist. Distances within crater radius are NaN.
    """
    nx, ny, nz = len(grd_x.flatten()), len(grd_y.flatten()), len(df)
    ej_dist_all = np.zeros([nx, ny, nz])
    for i, crater in df.iterrows():
        X = grd_x - crater.x
        Y = grd_y - crater.y
        ej_dist = dist(crater.x, crater.y, X, Y)
        ej_dist[ej_dist < crater.rad] = np.nan
        ej_dist_all[:, :, i] = ej_dist
    return ej_dist_all


def get_ejecta_thickness(distance, simple2complex=SIMPLE2COMPLEX):
    """
    Return ejecta thickness as a function of distance.

    Complex craters eqn. 1, Kring 1995
    """

# Helper functions
def latlon2xy(lon, lat, rp=R_MOON):
    """
    Return (x, y) distance from pole of coords (lon, lat) [degrees]. 

    Returns in units of rp.
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    y = rp * np.cos(lat) * np.cos(lon)
    x = rp * np.cos(lat) * np.sin(lon)
    return x, y


def gc_dist(lon1, lat1, lon2, lat2, rp=R_MOON):
    """
    Calculate the great circle distance between two points using the Haversine formula.

    All args must be of equal length, lon and lat in decimal degrees.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return rp * 2 * np.arcsin(np.sqrt(a))


def dist(x1, y1, x2, y2):
    """Return simple distance between coordinates (x1, y1) and (x2, y2)."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


