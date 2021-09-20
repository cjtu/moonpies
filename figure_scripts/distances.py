import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import moonpies as mp
import default_config
from scipy.interpolate import interp1d as interp

#distances = np.load("distances.npy")
#print(distances[0,:]/1000)

fig_dir = "/home/kristen/codes/code/moonpies_package/figs/"
data_dir = "/home/kristen/codes/code/moonpies_package/data/"

crater_cols = ('cname', 'lat', 'lon', 'diam', 'age', 'age_low',
                          'age_upp', 'psr_area', 'age_ref', 'prio', 'notes')

def read_crater_list(columns, rp=1737e3):
    # Convert units, mandatory columns
    df = pd.read_csv(data_dir+"crater_list.csv", names=columns, header=0)
    df["diam"] = df["diam"] * 1000  # [km -> m]
    df["rad"] = df["diam"] / 2
    df["age"] = df["age"] * 1e9  # [Gyr -> yr]
    df["age_low"] = df["age_low"] * 1e9  # [Gyr -> yr]
    df["age_upp"] = df["age_upp"] * 1e9  # [Gyr -> yr]
    if "psr_area" in df.columns:
        df["psr_area"] = df.psr_area * 1e6  # [km^2 -> m^2]
    else:
        # Estimate psr area as 90% of crater area
        df["psr_area"] = 0.9 * np.pi * df.rad ** 2
    
    df["x"], df["y"] = latlon2xy(df.lat, df.lon, rp)
    df["dist2pole"] = gc_dist(0, -90, df.lon, df.lat, rp)
    df = df[df.diam <= 250e3]
    return df

def latlon2xy(lat, lon, rp=1737e3):
    """
    Return (x, y) [m] South Polar stereo coords from (lat, lon) [deg].

    Parameters
    ----------
    lat (num or arr): Latitude(s) [deg]
    lon (num or arr): Longitude(s) [deg]
    rp (num): Radius of the planet or moon [m]

    Return
    -------
    x (num or arr): South Pole stereo x coordinate(s) [m]
    y (num or arr): South Pole stereo y coordinate(s) [m]
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = rp * np.cos(lat) * np.sin(lon)
    y = rp * np.cos(lat) * np.cos(lon)
    return x, y

def gc_dist(lon1, lat1, lon2, lat2, rp=1737e3):
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
    gc_dist = rp * c
    return gc_dist

def get_crater_distances(df, symmetric=True, dtype=None):
    """
    Return 2D array of great circle dist between all craters in df. Distance
    from a crater to itself (or repeat distances if symmetric=False) are nan.

    Mandatory
        - df : Read in crater_list.csv as a DataFrame with defined columns
        - df : Required columns defined 'lat' and 'lon'
        - See 'read_crater_list' function

    Parameters
    ----------
    df (DataFrame): Crater DataFrame, e.g., read by read_crater_list
    TODO: Symmetric :

    Returns
    -------
    out (2D array): great circle distances between all craters in df
    """
    out = np.zeros((len(df), len(df)), dtype=dtype)
    for i in range(len(df)):
        for j in range(i):
            d = gc_dist(
                *df.iloc[i][["lon", "lat"]], *df.iloc[j][["lon", "lat"]]
            )
            out[i, j] = d
    if symmetric:
        out += out.T
    out[out <= 0] = np.nan
    return out

def get_ejecta_thickness(
    distance,
    radius,
    ds2c=18e3,
    order=-3,
    dtype=None,
    mode='cannon'
):
    """
    Return ejecta thickness as a function of distance given crater radius.

    Complex craters McGetchin 1973
    """
    exp_complex = 0.74  # McGetchin 1973, simple craters exp=1
    exp = np.ones(radius.shape, dtype=dtype)
    exp[radius * 2 > ds2c] = exp_complex
    thickness = 0.14 * radius ** exp * (distance / radius) ** order
    #thickness[np.isnan(thickness)] = 0
#    if distance > 4 * radius:
#        thickness = 0
    #if mode == 'cannon':
        # TODO: should moonpies also do 4 crater radii?
        # Cannon cuts off at 4 crater radii 
        #thickness[distance > 4 * radius] = 0 
    return thickness

df = read_crater_list(crater_cols)
ej_distances = get_crater_distances(df)

#DT = np.load("temp.npy") 
#DT = DT + 250
R = df["rad"]

#d = np.linspace(1.*R,4.*R,len(DT[0]))/1000 
#e_temp = [200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400]
#dist_array = np.zeros((len(R), len(e_temp)))

c1 = 11
c2 = 0

#dtemp = interp(d[:,c1],DT[c1,:], fill_value="extrapolate")  
#ej_T = dtemp(ej_distances[c1,c2]/1000)
vf = 1 - (2.913*(ej_distances[c1,c2]/df["rad"][c1])**(-3.978))

#thick = 0.14 * (df["rad"][c1]**0.74) * ((ej_distances[c1,c2]*df["rad"][c1]/df["rad"][c1])**-3.)# pm 0.5
#thick = get_ejecta_thickness(ej_distances[c1,c2], df["rad"][c1])

#print("From", df["cname"][c1], "to", df["cname"][c2], "is volume fraction", vf, "at", ej_T, "Kelvin and ", ej_distances[c1,c2], "m with", thick, "m thickness")

#print(ej_distances[c1,c2] / (4 * df["rad"][c1]))
dists = 60*1e3*np.linspace(1.35, 4., 449) #crater radii
vf_plot = 1-(2.913*(dists**(-3.978)))

rads = df["rad"][3] 

ries = np.array([[1.375,0.9],[1.583333333,0.29], [1.916666667,0.22], [2.125,0.09], [2.233333333,0.21], [2.291666667,0.17], [2.666666667,0.07], [2.933333333,0.18], [3.041666667,0.10]])

thick = 0.14 * (rads**0.74) * ((dists/rads)**-3.)# pm 0.5

#depth = thick * (3-1) #((1/(1-vf_plot)) - 1)
print(df["rad"][3])

cfg = default_config.Cfg(mode='moonpies')
vol_frac = cfg.vol_frac_a * 1e-3 * dists ** cfg.vol_frac_b

if cfg.vol_frac_petro:
    vol_frac[vol_frac > 5] = 0.5 * vol_frac[vol_frac > 5] + 2.5

print(ej_distances[3,2])
plt.figure(1,figsize=(6,4))
plt.plot(dists, vol_frac)
#for r in range(0,9):
#    plt.scatter(ries[r][0],1-ries[r][1])
mp.plot_version(cfg, loc='lr')
plt.title("Volume Fraction Function with Points from Ries Crater")
plt.xlabel("Distance [Crater Radii]")
plt.ylabel("Volume Fraction [%Target Material]")
plt.savefig(fig_dir+"vf_func.png", dpi=300)
plt.show()

plt.figure(2, figsize=(6,4))
plt.plot(dists, thick, label="Ejecta thickness")
plt.plot(dists, np.zeros(len(thick)), label="Original Surface")
plt.plot(dists, -1*thick*vol_frac, label="Ballistic Sed. Mixing Region")
plt.axvline(ej_distances[3,2], color='red', ls='--', label='Amundsen->Faustini')
plt.axvline(ej_distances[3,1], color='orange', ls='--', label='Amundsen->Shoemaker')
plt.axvline(ej_distances[3,0], color='gold', ls='--', label='Amundsen->Haworth')
#plt.plot(dists, DT[3], label = "Ejecta Temperature [K]")
mp.plot_version(cfg, loc='lr')
plt.legend()
plt.title("Ballistic Sedimentation Mixing Region")
plt.xlabel("Distance [m]")
plt.ylabel("Thickness [m]")
plt.savefig(fig_dir+"teq_mixing_region.png", dpi=300)
plt.show()
