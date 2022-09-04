import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import moonpies as mp
import config

fig_dir = "/home/kristen/codes/code/moonpies_package/figs/"
data_dir = "/home/kristen/codes/code/moonpies_package/data/"

#define constants
G_MOON = 1.624  # [m s^-2]
R_MOON = 1737 * 1e3  # [m]

def main():
    """
    Return equilibrium temperature resulting from ejecta being deposited onto each cold trap crater from every other polar crater
    Many supporting functions taken from MoonPIES (moonpies.py): latlon2xy(), gc_dist(), get_crater_distances(), read_crater_list()
    Note: plots turned off, can uncomment in fast() if desired. Additional plotting capacity in slow(), but code will take much longer to run.

    Returns
    -------
    Teq (num or array): equilibrium temperature from i into j (Teq[i,j]) [K]
    """
    crater_cols = ('cname', 'lat', 'lon', 'diam', 'age', 'age_low',
                            'age_upp', 'psr_area', 'age_ref', 'prio', 'notes')

    df = read_crater_list(crater_cols)
    ej_distances = get_crater_distances(df)

    coldtrap_indicies = [0,1,2,22,3,14,5,6,8,21,4]
    coldtrap_temps = [45.,50.,50.,100.,90.,70.,70.,60.,70.,70.,70.]

    Teq = np.zeros((len(df), len(df)))
    Teq_ct = np.zeros((len(coldtrap_temps), len(df)))

    for i in range(0,len(df)):
        for c, j in enumerate(coldtrap_indicies):
            if i != j: #can't dump ejecta into yourself
                vf = 1 - (2.913*(ej_distances[i,j]/df["rad"][i])**(-3.978))
                DT = ejecta_blanket(df["rad"][i]*1000,ej_distances[i,j])
                T_clast_i = coldtrap_temps[c]
                Teq_temp = equilibriate(vf, DT, T_clast_i)
                Teq[c,i] = Teq_temp
                Teq_ct[c,i] = Teq_temp
                #print(Teq[i, j])                
                #if Teq_temp >= 110.:
                    #print(df["cname"][i], "into", df["cname"][j], "is", Teq_temp)
    coldtrap_craters = pd.Index(['Haworth', 'Shoemaker', 'Faustini', 'Shackleton', 'Amundsen', 'Sverdrup', 'Cabeus B', 'de Gerlache', "Idel'son L", "Wiechert J", "Cabeus"], name="rows")
    names = df["cname"]
    df_teq = pd.DataFrame(Teq_ct, index=coldtrap_craters, columns=names)
    return Teq, Teq_ct, df_teq

def equilibriate(vf, DT, T_clast_i, T_ejecta_i=250):
    """
    Returns equilibrium temperature of a mixed material

    Parameters
    ----------
    vf (num): volume fraction [fraction target material]
    DT (num): delta temperature for ejecta blanket [K]
    T_clast_i (num): initial temperature of the target material [K]
    T_ejecta_i (num): initial temperature of the ejecta (pre-impact) [K]

    Returns
    -------
    T (num): equilibrium temperature of the mixed material [K]
    """
    alpha = get_alpha()
    T_surf = DT + T_ejecta_i
    T = fast(alpha, T_clast_i, T_surf, vf)
    return np.mean(T)

def fast(alpha, T_clast_i, T_surf, vf):
    """
    Allows temperature of a mixed material to evolve/equilibriate over time

    Parameters
    ----------
    vf (num): volume fraction [fraction target material]
    alpha (num): scaling factor [unitless]
    T_clast_i (num): initial temperature of the target material [K]
    T_ejecta_i (num): initial temperature of the ejecta (pre-impact) [K]

    Returns
    -------
    T (array): evolved temperature profile of the mixed material [K]
    """
    T = np.full(100, T_clast_i)#T_surf) #1,000 steps = 10 cm total, 100 micron steps
    T = vol_frac(T, T_surf, vf)
    '''
    plt.figure(1)
    plt.plot(np.linspace(0,10,100), T, 'r-')
    plt.xlabel("Distance (mm)")
    plt.ylabel("Temperature (K)")
    plt.title("Initial Temperature Distribution")
    plt.savefig("Initial_Fast_Temps.png")
    plt.show()
    '''
    T_new = np.zeros(100)
    for n in range(0,9999): #time
        T[0] += alpha* T[1] + alpha* T[2]  - (2.*alpha*T[0])
        T[-1] += alpha* T[-2] + alpha* T[-3]  - (2.*alpha*T[-1])
        for i in range(1,99): #space
            T_new[i] = (alpha*T[i-1] + alpha* T[i+1]  - (2.*alpha*T[i]))
        T += T_new
    '''
    plt.figure(2)
    plt.plot(np.linspace(0,10,100), T, 'r-')
    plt.xlabel("Distance (mm)")
    plt.ylabel("Temperature (K)")
    plt.title("Final Temperature Distribution")
    plt.savefig("Fast_Temps.png")
    plt.show()
    '''
    return T  

def ejecta_temp(ke, thick, Cp = 300, density = 1500, eff=0.5):
    """
    Return the temperature distribution of the ejecta blanket

    Parameters
    ----------
    ke (num or array): kinetic energy of the ejecta blanket [J/m^2]
    thick (num or array): thickness of the ejecta blanket [m]
    Cp (num): specific heat of the ejecta blanket [J/K/kg]
    density (num): density of the ejecta blanket [kg/m^3]
    """
    return  eff*ke/(3*thick*Cp*density)

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
    thick = thickness(d, R)
    ke = mps2KE(ballistic_spherical(theta, d), thick2mass(thick))
    DT = ejecta_temp(ke,thick)
    return DT

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

def read_crater_list(columns, rp=1737e3):
    # Convert units, mandatory columns
    df = pd.read_csv("./data/crater_list.csv", names=columns, header=0)
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

def get_alpha(L=330., density=2.5, k = 0.0125, Cp = 0.815, T_liq_c=120, T_sol_c=100, delt=0.1, delchi=0.1):
    '''
    Return alpha, a scaling parameter for thermal equilibriation

    Parameters
    ----------
    L (num): latent heat of ice melt [J/g]
    density (num): density, [g/cm^3]
    k (num): conductivity [W/cm/K]
    Cp (num): heat capacity [J/g/K]
    T_liq_c (num): liquidus of clasts [K]
    T_sol_c (num): solidus of clasts [K]
    delt (num): time step size [s], must match delchi
    delchi (num): spatial step size [cm], must match delt

    Returns
    -------
    alpha (num): scaling factor [unitless]
    '''
    Cp_L = Cp + (L/(T_liq_c-T_sol_c)) #heat capacity with latent heat
    K = k/(density*Cp_L) #thermal diffusivity
    alpha = ((K*delt)/(delchi**2))
    return alpha

def vol_frac(T, T_surf, vf):
    """
    Alters initial temperature profile to give it a volume fraction of ejecta material

    Parameters
    ----------
    T (array): initial temperature profile [K]
    vf (num): volume fraction [fraction target material]
    T_ejecta_i (num): heated temperature of the ejecta [K]

    Returns
    -------
    T (array): initial temperature profile of the mixed material [K]
    """
    if vf >= 0.93: #95%
        T[3:4] = T_surf
        T[48:50] = T_surf
        T[72:73] = T_surf
        T[93:94] = T_surf
    elif vf >= 0.85: #90%
        T[3:4] = T_surf
        T[30:31] = T_surf
        T[48:50] = T_surf
        T[59:61] = T_surf
        T[72:74] = T_surf
        T[93:95] = T_surf
    elif vf >= 0.75: #80%
        T[3:4] = T_surf
        T[8:10] = T_surf
        T[22:27] = T_surf
        T[30:32] = T_surf
        T[48:50] = T_surf
        T[59:61] = T_surf
        T[72:74] = T_surf
        T[86:90] = T_surf
        T[93:95] = T_surf
    elif vf >= 0.65: #70%
        T[2:4] = T_surf
        T[8:10] = T_surf
        T[18:20] = T_surf
        T[22:27] = T_surf
        T[30:32] = T_surf
        T[48:50] = T_surf
        T[53:55] = T_surf
        T[59:61] = T_surf
        T[66:69] = T_surf
        T[72:74] = T_surf
        T[86:90] = T_surf
        T[93:97] = T_surf
    elif vf >= 0.55: #60%
        T[2:5] = T_surf
        T[8:10] = T_surf
        T[18:22] = T_surf
        T[22:27] = T_surf
        T[30:35] = T_surf
        T[48:50] = T_surf
        T[53:56] = T_surf
        T[59:63] = T_surf
        T[66:69] = T_surf
        T[72:75] = T_surf
        T[86:90] = T_surf
        T[93:97] = T_surf
    elif vf >= 0.45: #50%
        T[2:5] = T_surf
        T[8:12] = T_surf
        T[18:22] = T_surf
        T[22:27] = T_surf
        T[30:38] = T_surf
        T[48:50] = T_surf
        T[53:57] = T_surf
        T[59:63] = T_surf
        T[66:69] = T_surf
        T[72:77] = T_surf
        T[86:91] = T_surf
        T[93:98] = T_surf
    elif vf >= 0.35: #40%
        T[2:5] = T_surf
        T[8:14] = T_surf
        T[18:22] = T_surf
        T[22:27] = T_surf
        T[30:40] = T_surf
        T[48:50] = T_surf
        T[53:57] = T_surf
        T[59:64] = T_surf
        T[66:69] = T_surf
        T[72:79] = T_surf
        T[83:91] = T_surf
        T[93:98] = T_surf
    elif vf < 0.35:
        exit()
    return T

def slow(alpha, T_clast_i, T_surf, vf):
    """
    Allows temperature of a mixed material to evolve/equilibriate over time

    Parameters
    ----------
    vf (num): volume fraction [fraction target material]
    alpha (num): scaling factor [unitless]
    T_clast_i (num): initial temperature of the target material [K]
    T_ejecta_i (num): initial temperature of the ejecta (pre-impact) [K]

    Returns
    -------
    T (array): evolved temperature profile of the mixed material [K]
    """
    T = np.full((100, 1000), T_clast_i)#T_surf) #1,000 steps = 10 cm total, 100 micron steps
    T = vol_frac(T, T_surf, vf)
    T_new = np.zeros(100, dtype='float64')
    for n in range(0,999): #time
        T[0,n+1] = T[0,n] + alpha* T[1,n] + alpha* T[2,n]  - (2.*alpha*T[0,n])
        T[-1,n+1] = T[-1,n] + alpha* T[-2,n] + alpha* T[-3,n]  - (2.*alpha*T[-1,n])
        for i in range(1,99): #space
            T_new[i] = (alpha*T[i-1,n] + alpha* T[i+1,n]  - (2.*alpha*T[i,n]))
        T[:,n+1] = T[:,n] + T_new#print(alpha*T[i-1,n] + alpha* T[i+1,n], 2.*alpha*T[i,n])  
        # 
    return T  

def test_melt():
    vf = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    T_clast_i = np.array([50, 60, 70, 80, 90, 100])
    #vf = 0.95
    DT = 50
    #T_clast_i = 100.
    alpha = get_alpha()
    T_surf = DT + 250.
    maxT = np.empty((100, len(vf), len(T_clast_i)))
    maxTlen = np.empty((len(vf), len(T_clast_i)))
    fraction = np.empty((len(vf), len(T_clast_i)))
    for i, f in enumerate(vf):
        for j, te in enumerate(T_clast_i):
            T = slow(alpha, te, T_surf, f)
            maxT[:,i,j] = np.max(T, axis=1)
            maxtemp = np.max(T, axis=1)
            maxTlen[i,j] = len(maxtemp[maxtemp>110])
            fraction[i,j] = maxTlen[i,j]/(100*(1-vf[i])) 
    
    cfg = config.Cfg(mode='moonpies')

    plt.figure(1, figsize=(6,4))
    plt.plot(np.linspace(0,10,100), T[:,0], 'r-')
    mp.plot_version(cfg, loc='ll')
    plt.xlabel("Distance (mm)")
    plt.ylabel("Temperature (K)")
    plt.title("Initial Temperature Distribution")
    plt.savefig(fig_dir+"Initial_Temps_test.png")
    #plt.show()

    plt.figure(2, figsize=(6,4))
    plt.plot(np.linspace(0,10,100), T[:,-1], 'r-')
    mp.plot_version(cfg, loc='ll')
    plt.xlabel("Distance (mm)")
    plt.ylabel("Temperature (K)")
    plt.title("Final Temperature Distribution")
    plt.savefig(fig_dir+"Final_Temps_test.png")    
    #plt.show()

    plt.figure(3, figsize=(6,4))
    plt.plot(vf*100, fraction, 'r-')
    mp.plot_version(cfg, loc='ll')
    plt.xlabel("Volume Fraction [% Target Material]")
    plt.ylabel("Length Units Melted")
    plt.title("Number of Ejecta Length Units Heated")
    #plt.show()
    plt.savefig(fig_dir+"max_temps_test_vf_frac.png", dpi=300)

    plt.figure(4, figsize=(6,4))
    plt.plot(vf*100, maxTlen, 'r-')
    mp.plot_version(cfg, loc='ll')
    plt.xlabel("Volume Fraction [% Target Material]")
    plt.ylabel("% Target Pixels Heated")
    plt.title("Number of Pixels Heated Above 110 K")
    #plt.show()
    plt.savefig(fig_dir+"max_temps_len.png", dpi=300)   

    plt.figure(5, figsize=(6,4))
    plt.plot(np.linspace(0,10,100), maxT[:,0,:], 'r-')
    mp.plot_version(cfg, loc='ll')
    plt.xlabel("Distance [mm]")
    plt.ylabel("Maximum Temperature [K]")
    plt.title("Maximum Temperature vs clast temp for 60% volume fraction")
    #plt.show()
    plt.savefig(fig_dir+"max_temps_test_clastT.png", dpi=300)
    '''
    plt.figure(6)
    plt.plot(np.linspace(0,10,100), T[:,:])
    plt.xlabel("Distance (mm)")
    plt.ylabel("Temperature (K)")
    plt.title("Evolving Temperature Distribution")
    plt.savefig(fig_dir+"Evolving_Temps_test.png")    
    plt.show()
    '''
    return T

#Teq, Teq_ct, df = main()
#np.savetxt("Teqs.csv", Teq, delimiter=',') #all craters into all craters, 24x24 grid
#np.savetxt("bal_sed_cold_traps.csv", Teq_ct, delimiter=',') #all craters into all cold traps, 24x10 grid

#df.to_csv('Teqs_cold_trap.csv')

t = test_melt()