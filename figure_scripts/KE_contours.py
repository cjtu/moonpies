#Gets contours of Kinetic Energy for all crater's ejecta blankets on the grid
#K. Frizzell
#Last updated 8/5/21

#Comment: This could be updated to be shorter/reference moonPIES


import moonpies as mp
import default_config
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd 
import rasterio
import matplotlib.ticker as ticker

def_cfg = default_config.Cfg(mode = 'moonpies')
can_cfg = default_config.Cfg(mode = 'cannon')

# Constants
G_MOON = 1.624  # [m s^-2]
R_MOON = 1737.4 * 1e3  # [m]

# Files to import (# of COLS must equal # columns in CSV)
CRATER_CSV = "crater_list.csv"
CRATER_COLS = (
    "cname",
    "lat",
    "lon",
    "diam",
    "age",
    "age_low",
    "age_upp",
    "psr_area",
    "age_ref",
    "priority",
    "notes",
)

# Model grid
GRDXSIZE = 400e3  # [m]
GRDYSIZE = 400e3  # [m]
GRDSTEP = 1e3  # [m / pixel]

def latlon2xy(lat, lon, rp=R_MOON):
    """Return (x, y) [rp units] S. Pole stereo coords from (lon, lat) [deg]."""
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = rp * np.cos(lat) * np.sin(lon)
    y = rp * np.cos(lat) * np.cos(lon)
    return x, y

def read_crater_list(crater_csv=CRATER_CSV, columns=CRATER_COLS):
    df = pd.read_csv(crater_csv, names=columns, header=0)

    # Convert units, mandatory columns
    df["diam"] = df["diam"] * 1000  # [km -> m]
    df["rad"] = df["diam"] / 2
    df["age"] = df["age"] * 1e9  # [Gyr -> yr]
    df["age_low"] = df["age_low"] * 1e9  # [Gyr -> yr]
    df["age_upp"] = df["age_upp"] * 1e9  # [Gyr -> yr]

    # Define optional columns
    if "psr_area" in df.columns:
        df["psr_area"] = df.psr_area * 1e6  # [km^2 -> m^2]
    else:
        # Estimate psr area as 90% of crater area
        df["psr_area"] = 0.9 * np.pi * df.rad ** 2

    # Define new columns
    df["x"], df["y"] = latlon2xy(df.lat, df.lon)
    df["dist2pole"] = gc_dist(0, -90, df.lon, df.lat)

    # Drop basins for now (>250 km diam)
    # TODO: handle basins somehow?
    df = df[df.diam <= 250e3]
    return df


# Functions
def ballistic_planar(theta, d, g=G_MOON):
    """
    Return ballistic speed (v) given ballistic range (d) and gravity of planet (g).
    Assumes planar surface (d << R_planet).  
    
    Parameters
    ----------
    d (num or array): ballistic range [m]
    g (num): gravitational force of the target body [m s^-2]
    theta (num): angle of impaact [radians]
    
    Returns
    -------
    v (num or array): ballistic speed [m s^-1]   
 
    """
    return np.sqrt((d * g) / np.sin(2 * theta))


def ballistic_spherical(theta, d, g=G_MOON, rp=R_MOON):
    """
    Return ballistic speed (v) given ballistic range (d) and gravity of planet (g).
    Assumes perfectly spherical planet (Vickery, 1986).
    
    Parameters
    ----------
    d (num or array): ballistic range [m]
    g (num): gravitational force of the target body [m s^-2]
    theta (num): angle of impaact [radians]
    rp (num): radius of the target body [m]
    
    Returns
    -------
    v (num or array): ballistic speed [m s^-1]   
 
    """
    tan_phi = np.tan(d / (2 * rp))
    return np.sqrt((g * rp * tan_phi) / ((np.sin(theta) * np.cos(theta)) + (np.cos(theta)**2 * tan_phi)))

def mps2kmph(v):
    """
    Return v in km/hr, given v in m/s
    
    Parameters
    ----------
    v (num or array): velocity [m s^-1]
    
    Returns
    -------
    v (num or array): velocity [km hr^-1]
    """
    
    return 3600. * v / 1000.

def thickness(d, R):
    """
    Calculate the thickness of an ejecta blanket in meters
    
    Parameters
    ----------
    d (num or array): distance from impact [m]
    R (num or array): radius of transient crater diameter [m]
    
    Returns 
    -------
    thick (num or array): ejecta thickness [m] """
    
    return 0.14 * (R**0.74) * ((d/R)**-3.)# pm 0.5

def thick2mass(thick, density=1500.):
    """
    Convert an ejecta blanket thickness to kg per meter squared, default density of the ejecta blanket from Carrier et al. 1991. Density should NOT be the bulk density of the Moon! 
    
    Parameters
    ----------
    thick (num or array): ejecta blanket thickness [m]
    density (num): ejecta blanket density [kg m^-3]
    
    Returns 
    -------
    mass (num or array): mass of the ejecta blanket [kg]
    """
    return thick * density

def mps2KE(v, m):
    """
    Convert ballistic speeds to kinetic energies in joules per meter squared
    
    Parameters
    ----------
    v (num or array): ballistic speeds, function of distance [m s^-1]
    m (num or array): mass of the ejecta blanket [kg]
    
    Returns
    -------
    KE (num or array): kinetic energy of the ejecta blanket as a function of distance [J m-2]
    """
    return 0.5 * m * v**2.
    
def ice_melted(ke, T=100, Cp=4.2, frac_mix=0.5, frac_ice=0.056, frac_rad=0.1):
    """
    Return the mass and depth of ice melted by ejecta blanket
    
    Parameters
    ----------
    ke (num or array): kinetic energy of ejecta blanket [J m^-2]
    T (num): surface temperature [K]
    Cp (num): heat capacity for regolith, 0.7-4.2 [kJ/kg/K] for H2O
    frac_mix (num): 0.5 is the percentage used for heating vs mechanical mixing
    frac_ice (num): 0.056 is the percentage of ice vs regolith (Colaprete 2010)
    frac_rad (num): 0.3-0.1 is the factor of heat not lost by radiation into space (Stopar 2018)
    
    Returns
    -------
    Mice (num or array): mass of ice melted due to ejecta [kg]
    Dice (num or array): depth of ice melted due to ejecta [m]
    """
    ke_kj = ke/1000. #J to kJ
    delT = 272. - T #heat from temp T to the melting point of water, 272 K
    Q = frac_ice*frac_rad*frac_mix*ke_kj #kJ / m^2
    L = 333. #latent heat of ice, kJ / kg
    Mice = Q/(L + Cp*delT)  
    Dice = Mice / (frac_ice*1500.) #mass / density * fraction of material that is ice
    return Mice, Dice  
    
def ejecta_blanket(R, d, theta=np.deg2rad(45)):
    """
    Return the properties of the ejecta blanket
    
    Parameters
    ----------
    Rad (num or array): crater radi[us/i] [m]
    theta (num or array): impact angle [radians]
    
    Returns 
    -------
    thick (num or array): ejecta blanket thickness as a function of distance [m]
    v (num or array): ballistic speed as a function of distance (spherical) [km hr^-1]
    ke (num or array): ejecta blanket kinetic energy as a function of distance [J m^-2]
    Mice (num or array): mass of ice melted due to ejecta [kg]
    Dice (num or array): depth of ice melted due to ejecta [m]
    """
    thick = thickness(d,R)
    v_spherical = mps2kmph(ballistic_spherical(theta, d))
    ke = mps2KE(ballistic_spherical(theta,d), thick2mass(thick))
    ke[d<R] = 0
    Mice, Dice = ice_melted(ke)
    return thick, v_spherical, ke, Mice, Dice
    
#Grid code
_DTYPE = np.float32
_GRD_Y, _GRD_X = np.meshgrid(
    np.arange(GRDYSIZE, -GRDYSIZE, -GRDSTEP, dtype=_DTYPE), 
    np.arange(-GRDXSIZE, GRDXSIZE, GRDSTEP, dtype=_DTYPE), 
    sparse=True, indexing='ij'
)

def xy2latlon(x, y, rp=1737.4e3):
    """
    Return (lat, lon) [deg] from South Polar stereo coords (x, y) [m].

    Parameters
    ----------
    x (num or arr): South Pole stereo x coordinate(s) [m]
    y (num or arr): South Pole stereo y coordinate(s) [m]
    rp (num): Radius of the planet or moon [m]

    Return
    -------
    lat (num or arr): Latitude(s) [deg]
    lon (num or arr): Longitude(s) [deg]
    """
    z = np.sqrt(rp ** 2 - x ** 2 - y ** 2)
    lat = np.rad2deg(-np.arcsin(z / rp))
    lon = np.rad2deg(np.arctan2(x, y))
    return lat, lon


def gc_dist(lon1, lat1, lon2, lat2, rp=1737.4e3):
    """
    Return great circle distance [m] from (lon1, lat1) - (lon2, lat2) [deg].

    Uses the Haversine formula adapted from C. Veness
    https://www.movable-type.co.uk/scripts/latlong.html

    Parameters
    ----------
    lon1 (num or arr): Longitude [deg] of start point
    lat1 (num or arr): Latitude [deg] of start point
    lon2 (num or arr): Longitude [deg] of end point
    lat2 (num or arr): Latitude [deg] of end point
    rp (num): Radius of the planet or moon [m]

    Return
    ------
    gc_dist (num or arr): Great circle distance(s) in meters [m]
    """
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    sin2_dlon = np.sin((lon2 - lon1) / 2) ** 2
    sin2_dlat = np.sin((lat2 - lat1) / 2) ** 2
    a = sin2_dlat + np.cos(lat1) * np.cos(lat2) * sin2_dlon
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = rp * c
    return dist


def get_gc_dist_grid(df, grdx, grdy):
    """
    Return 3D array of great circle dist between all craters in df and every
    point on the grid.

    Parameters
    ----------
    df (DataFrame):
    grdx (arr):
    grdy (arr):

    Return
    ------
    grd_dist (3D arr: NX, NY, Ndf):
    """
    ny, nx = grdy.shape[0], grdx.shape[1]
    lat, lon = xy2latlon(grdx, grdy)
    grd_dist = np.zeros((nx, ny, len(df)))
    for i in range(len(df)):
        grd_dist[:, :, i] = gc_dist(*df.iloc[i][["lon", "lat"]], lon, lat)
    return grd_dist


#call it
df = read_crater_list('D:/2021-internship/crater_list.csv')
dist = get_gc_dist_grid(df,_GRD_X,_GRD_Y)
rad = df.rad.values 
d = np.linspace(1, 450, 449) * 1000.  # [m]
thick, v_spherical, ke, Mice, Dice = ejecta_blanket(rad[None,None,:], dist)
v_planar = mps2kmph(ballistic_planar(np.deg2rad(45), d))


#plots the figure in terms of [J/m^2]
mp.plot_version(def_cfg,loc='ul')
plt.rcParams.update({
    'figure.figsize': (6, 6),
    'figure.facecolor': 'white',
    'xtick.top': True,
    'xtick.direction': 'in',
    'ytick.right': True,
    'ytick.direction': 'in',
    'axes.grid': True
})

ax = plt.subplot(1,1,1)
plotting = np.sum(ke,axis=2)
print(np.mean(plotting))
plt.contour(plotting,levels = ([int(n) for n in np.linspace(1e9,1e11,200)]),extent = (-400,400,-400,400))
plt.gca().invert_yaxis()
plt.title('Sum of KE from Impact Ejecta')
plt.xlabel('Distance [km]')
plt.ylabel('Distance [km]')
plt.colorbar(label='Kinetic Energy $[J/m^2]$',shrink=0.8)
ax.set_aspect('equal')
plt.savefig('KEcontours.png',dpi=300)


#same plot but in tons of TNT
mp.plot_version(def_cfg,loc='ul')
plt.rcParams.update({
    'figure.figsize': (6, 6),
    'figure.facecolor': 'white',
    'xtick.top': True,
    'xtick.direction': 'in',
    'ytick.right': True,
    'ytick.direction': 'in',
    'axes.grid': True
})

ax = plt.subplot(1,1,1)
plottingTNT = np.sum(ke,axis=2)
plottingTNT = plottingTNT[:,:]/int(4.184e9)
plottingTNT = plottingTNT[:,:]*int(1e6)
print(np.mean(plottingTNT))
z = plt.contour(plottingTNT,levels = ([int(n) for n in np.linspace(1e5,1e7,50)]),extent = (-400,400,-400,400))
plt.gca().invert_yaxis()
plt.title('Sum of Kinetic Energy from Impact Ejecta')
plt.xlabel('Distance [km]')
plt.ylabel('Distance [km]')
plt.colorbar(label='Kinetic Energy [Tons TNT per $km^2$]',shrink=0.8)
fmt = ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
ax.clabel(z,z.levels[0:3],inline=1,fontsize=10,fmt=fmt)
ax.set_aspect('equal')
plt.savefig('KEcontoursTNT.png',dpi=300)



#exporting to ArcGIS
lat, lon = xy2latlon(_GRD_X, _GRD_Y)


#This prints the csv as an x,y,z file
s = 'x,y,ke'
for i in range(_GRD_X.shape[1]):
    for j in range(_GRD_X.shape[1]):
        x_str = f'{_GRD_X[0][i]:.6f}'
        y_str = f'{_GRD_Y[j][0]:.6f}'
        ke_str = f'{plottingTNT[i,j]:.6f}'
        s += '\n'
        s += ','.join([x_str,y_str,ke_str])
with open('georef_KE_xy.csv','w') as f:
    f.write(s)


#this prints the csv as a lon, lat, z file
s = 'lon,lat,ke'
for i in range(lat.shape[0]):
    for j in range(lat.shape[1]):
        lon_str = f'{lon[i,j]:.6f}'
        lat_str = f'{lat[i,j]:.6f}'
        ke_str = f'{plottingTNT[i,j]:.6f}'
        s += '\n'
        s += ','.join([lon_str,lat_str,ke_str])
with open('georef_KE.csv','w') as f:
    f.write(s)

#(((TESTING)))
#adds coordinates to an image
transform = rasterio.transform.from_bounds(-180,-90,180,-80,800,800)
proj = 'PROJCS["Moon2000_spole",GEOGCS["GCS_Moon",DATUM["Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",NORTH],AXIS["Northing",NORTH]]'
profile = {'driver': 'GTiff', 'height': 800, 'width': 800, 'count': 1, 'transform': transform, 'dtype': rasterio.float64}
with rasterio.open('KEContours.tif', 'w', crs=proj, **profile) as dst:
    dst.write(plottingTNT,indexes=1)
