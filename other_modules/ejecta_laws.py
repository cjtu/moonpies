import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Constants
G_MOON = 1.624  # [m s^-2]
R_MOON = 1737 * 1e3  # [m]

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
    thick = np.empty((len(R), len(d)))
    ke = np.empty((len(R), len(d)))
    #Mice = np.empty((len(R), len(d)))
    #Dice = np.empty((len(R), len(d))) 
    v_spherical = mps2kmph(ballistic_spherical(theta, d))
    for r in range(0, len(R)):
        #d=np.linspace(10,4.*R[r],500) #custom distance array for each crater
        thick[r,:] = thickness(np.linspace(1.*R[r],4.*R[r],len(d)), R[r])
    
        ke[r,:] = mps2KE(ballistic_spherical(theta, np.linspace(1.*R[r],4.*R[r],len(d))), thick2mass(thick[r,:]))
    Mice, Dice = ice_melted(ke)
    return thick, v_spherical, ke, Mice, Dice
    
    
Rad = pd.read_csv('../data/crater_list.csv',usecols=[3])#np.array([25.7, 25.9, 21.25, 28.85, 29.6, 16.35])*1000. #UPDATE
d = np.linspace(1, 450, 449) * 1000.  # [m]
R = (Rad['Diam (km)']) * 1000. /2.
thick, v_spherical, ke, Mice, Dice = ejecta_blanket(R, d)

v_planar = mps2kmph(ballistic_planar(np.deg2rad(45), d))


###PLOT
f, ax = plt.subplots(figsize=(5, 5))
f.suptitle('Ballistic speed of ejecta vs distance traveled')

d_km = d / 1000
ax.plot(d_km, v_planar, label='planar')
ax.plot(d_km, v_spherical, label='spherical')

# inset
axins = ax.inset_axes([0.5, 0.1, 0.4, 0.4])
axins.plot(d_km, v_planar, label='planar')
axins.plot(d_km, v_spherical, label='spherical')
x1, x2, y1, y2 = (400, 450, 2700, 3100)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
ax.indicate_inset_zoom(axins, edgecolor="black")
ax.legend()

ax.set_xlabel('distance [km]')
ax.set_ylabel('speed [$km/h$]')
plt.savefig("speedvsdistance.png", dpi=300)


f2, axs = plt.subplots(2,2)#,figsize=(5, 5))
axs[0,0].set_title('Kinetic energy')
for i in range(0,len(R)-3):
    axs[0,0].semilogy(np.linspace(1.*R[i],4.*R[i],len(d))/1000., ke[i,:], label='Kinetic Energy') #d/1000., ke[i,:], label='Kinetic Energy')#
axs[0,0].set_xlabel('distance [km]')
axs[0,0].set_ylabel('kinetic energy [$J/m^{2}$]')
axs[0,0].grid(True, which="both")

axs[0,1].set_title('Thickness')
for i in range(0,len(R)-3):
    axs[0,1].semilogy(np.linspace(1.*R[i],4.*R[i],len(d))/1000., thick[i,:], label='Thickness') # d/1000.
axs[0,1].set_xlabel('distance [km]')
axs[0,1].set_ylabel('thickness [$m$]')
axs[0,1].grid(True, which="both")

axs[1,0].set_title('Ballistic Sedimentation Regime')
for i in range(0,len(R)-3):
    Bally = ke[i,ke[i,:] >= 10**9.5]
    xtemp = np.linspace(1.*R[i],4.*R[i],len(d))/1000.
    Ballx = xtemp[ke[i,:] >= 10**9.5]
    axs[1,0].semilogy(Ballx, Bally, label='depth affected [$m$]')
axs[1,0].set_xlabel('distance [km]')
axs[1,0].set_ylabel('depth [$m$]')
axs[1,0].grid(True, which="both")

axs[1,1].set_title('Depth of Ice Melted')
for i in range(0,len(R)-3):
    axs[1,1].semilogy(np.linspace(1.*R[i],4.*R[i],len(d))/1000., Mice[i,:]/(0.056*1500.), label='Depth of ice melted') #
axs[1,1].set_xlabel('distance [km]')
axs[1,1].set_ylabel('depth melted [$m$]')
axs[1,1].grid(True, which="both")
plt.tight_layout()
plt.savefig("Distance_ejecta.png", dpi=300)


f3, axs = plt.subplots(2,2)#,figsize=(5, 5))
axs[0,0].set_title('Kinetic energy')

for i in range(0,len(R)-3):
    axs[0,0].semilogy(np.linspace(1.*R[i],4.*R[i],len(d))/R[i], ke[i,:], label='Kinetic Energy')
axs[0,0].set_xlabel('distance [Crater Radii]')
axs[0,0].set_ylabel('kinetic energy [$J/m^{2}$]')
axs[0,0].grid(True, which="both")

axs[0,1].set_title('Thickness')
for i in range(0,len(R)-3):
    axs[0,1].semilogy(np.linspace(1.*R[i],4.*R[i],len(d))/R[i], thick[i,:], label='Thickness')
axs[0,1].set_xlabel('distance [Crater Radii]')
axs[0,1].set_ylabel('thickness [$m$]')
axs[0,1].grid(True, which="both")

axs[1,0].set_title('Ballistic Sedimentation Regime')
for i in range(0,len(R)-3):
    Bally = ke[i,ke[i,:] >= 10**9.5]
    xtemp = np.linspace(1.*R[i],4.*R[i],len(d))/R[i]
    Ballx = xtemp[ke[i,:] >= 10**9.5]
    axs[1,0].semilogy(Ballx, Bally, label='depth affected [$m$]')
axs[1,0].set_xlabel('distance [Crater Radii]')
axs[1,0].set_ylabel('depth [$m$]')
axs[1,0].grid(True, which="both")

axs[1,1].set_title('Depth Melted')
for i in range(0,len(R)-3):
    axs[1,1].semilogy(np.linspace(1.*R[i],4.*R[i],len(d))/R[i], Mice[i,:]/(0.056*1500.), label='Depth of ice melted')
axs[1,1].set_xlabel('distance [Crater Radii]')
axs[1,1].set_ylabel('depth melted [$m$]')
axs[1,1].grid(True, which="both")
plt.tight_layout()
plt.savefig("Distance_ejecta_R.png", dpi=300)
