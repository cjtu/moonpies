import numpy as np

COLDTRAP_AREA = 1.3e4 * 1e6
TIMESTEP = 10e6 
t = 2.1e9

def find_closest(arr,val):
    '''
    Returns index of closest value in array to specified value of interest.
    '''
    abs_diffs = np.abs(arr-val)


    return abs_diffs.argmin()


    

def sun_scalar(time, supply_rate):
    '''
    Returns the inputted supply rate multipled by the solar luminosity factor given from Bachall, 2011. 
    '''
    # Bachall, 2001
     
    sol_lum = np.array([0.677, 0.721, 0.733, 0.744, 0.754, 0.764,
                    0.775, 0.786, 0.797, 0.808, 0.820, 0.831, 
                    0.844, 0.856, 0.869, 0.882, 0.896, 0.910, 
                    0.924, 0.939, 0.954, 0.970, 0.986, 1.0]) # solar luminance values through time w.r.t to current value
                    
    sol_t = np.linspace(4.6, 0, len(sol_lum)) * 1e9 # years


    closest_idx = find_closest(sol_t, time)

    return supply_rate * sol_lum[closest_idx]
    




def solar_wind(time, coldtrap_area=COLDTRAP_AREA, ts = TIMESTEP, fys=True):
    '''
    Returns kg of ice in cold trap area at south pole. Does not account for movement of ice, only deposition given a supply rate from Reiss et al. 2021 
    '''
    volatile_supply_rate = 10e-15 #kg/(m^2 s) Reiss et al. 2021

    supply_rate_ts = volatile_supply_rate * 60 * 60 * 24 * 365 * ts # convert to kg/m^2 using the timestep used in greater program to get planar density per timestep

    supply_rate_sp = supply_rate_ts * coldtrap_area  # kg of ice in cold trap areas at pole. 

    # Faint young sun
    if fys == True: 
        out = sun_scalar(time, supply_rate_sp)
    else:
        out = supply_rate_sp

    
    return out 

solar_ice = solar_wind(t)
