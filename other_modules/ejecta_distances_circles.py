import numpy as np
import pandas as pd

RAD_MOON = 1737 #km

def gc_dist(lon1, lat1, lon2, lat2, rp=RAD_MOON):
    """Return great circ dist (lon1, lat1) - (lon2, lat2) [deg] in rp units."""
    pi = 3.14159265
    deg2rad = pi/180

    dLat = deg2rad * (lat2 - lat1)
    dLon = deg2rad * (lon2 - lon1)

    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(deg2rad *lat1) * np.cos(deg2rad*lat2) * np.sin(dLon/2) *np.sin(dLon/2)
    c = 2 * np.arcsin(np.sqrt(a))

    d = rp * c

    return d

df = pd.read_csv('/home/cjtu/projects/essi21/data/crater_list.csv')
df.columns = ['cname','lat','lon', 'diam','age','age_low','age_upp','psr_area','age_ref','priority','notes']

hlat, hlon = df[df.cname == 'Haworth'][['lat', 'lon']].values.T
wlat, wlon = df[df.cname == 'Wiechert J'][['lat', 'lon']].values.T

gc_dist(hlon, hlat, wlon, wlat)

#df = pd.read_csv('data\crater_list.csv')

#hlat, hlon = df[df.cname == 'Haworth'][['lat', 'lon']].values.T
#wlat, wlon = df[df.cname == 'Wiechert J'][['lat', 'lon']].values.T

print(gc_dist(hlon, hlat, wlon, wlat))