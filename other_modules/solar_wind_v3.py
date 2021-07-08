import numpy as np

COLDTRAP_AREA = 1.3e4 * 1e6
TIMESTEP = 10e6 
t = 2.1e9
RAD_MOON = 1737e3
SA_MOON = 4 * np.pi * RAD_MOON ** 2

def find_closest(arr,val):
    '''
    Returns index of closest value in array to specified value of interest.
    '''
    abs_diffs = np.abs(arr-val)


    return abs_diffs.argmin()


    

def sun_scalar(time, supply_rate):
    '''
    Returns the inputted supply rate multipled by the solar luminosity factor given from Bachall, 2001. 
    '''

    # Bachall, 2001
    sol_lum = np.array([0.677, 0.721, 0.733, 0.744, 0.754, 0.764,
                    0.775, 0.786, 0.797, 0.808, 0.820, 0.831, 
                    0.844, 0.856, 0.869, 0.882, 0.896, 0.910, 
                    0.924, 0.939, 0.954, 0.970, 0.986, 1.0]) # solar luminance values through time w.r.t to current value
                    
    sol_t = np.linspace(4.6, 0, len(sol_lum)) * 1e9 # years


    closest_idx = find_closest(sol_t, time)

    return supply_rate * sol_lum[closest_idx]
    

def get_solar_wind_ice(time_arr, cfg):
    """"""
    # apply equation based on cfg.sw_mode
    # get total mass [kg] at each time in time_arr
    # return sw_mass_vs_time

    # everything else handled by get_ice_thickness(global_mass) -> polar_ice_thickness


def solar_wind(time, sol_wind_mode, fys=True, coldtrap_area=COLDTRAP_AREA, ts = TIMESTEP,  sa = SA_MOON):
    '''
    Returns kg of ice in cold trap area at south pole. 
    Does not account for movement of ice, only deposition given a supply rate from either Benna et al. 2019 or Hurely et al. 2017

    For "Benna" mode, the water supply rate is 2 g/s, which is then converted to a global supply rate over the surface area of the moon. 
    For Lucey-Hurley mode, the H2 supply rate is 30 g/s and is converted to a global supply rate over the surface area of the moon assuming 1 part per thousand is eventually converted to water after Lucey et al. 2020. 

    The supply rate is then converted to a planar density over some timstep and is multipled by the area of the south polar cold traps to obtain a mass of deposited ice. 
    If fys = True, this value is run through the faint young sun function which scales this value by the luminosity of the sun at the current timestep. 

    '''

    # @Christian, since these values will never change every time this function is called, we have the option to just hard code them in. Maybe if we find this function is slowing
    # things down. Otherwise, I kept it like this to improve readability. 

    if sol_wind_mode == "Benna":
        volatile_supply_rate = 2 / 1000 / sa #  2 [g/s] * 1/1000 [kg/g] / moon_surf_area [m^2] = 5.273e-17 kg/(m^2 s)   |   Benna et al. 2019; Arnold, 1979; Housley et al. 1973 
    elif sol_wind_mode == "Lucey-Hurley":
        volatile_supply_rate = 30 / 1000 / 1000 / sa  # 30 [g/s] * 1/1000 [kg/g] * 1/1000 [ppt] / moon_surf_area [m^2] = 7.912e-19 kg/(m^2 s)  |  Lucey et al. 2020, Hurley et al. 2017

    supply_rate_ts = volatile_supply_rate * 60 * 60 * 24 * 365 * ts # convert to kg/m^2 (planar density per timestep)

    supply_rate_sp = supply_rate_ts * coldtrap_area  # kg of ice in cold trap areas at pole. 

    # Faint young sun
    if fys == True: 
        out = sun_scalar(time, supply_rate_sp)
    else:
        out = supply_rate_sp

    
    return out 

solar_ice = solar_wind(t,'Lucey-Hurley')
