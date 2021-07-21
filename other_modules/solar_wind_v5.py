import numpy as np
import pandas as pd 

timestart = 4.25e9
timeend = 0
timestep = 10e6
n = int((timestart - timeend) / timestep)

TIME_ARR = np.linspace(timestart, timestep, n, dtype=np.float32)

# TODO: make csv of sol lum. Read in. 
sol_fn = '/Users/tylerpaladino/Documents/ISU/LPI_NASA/data/Bachall_sol_lum.csv'
sol_cols = ('Age', 'Luminosity')

def read_sol_lum(sol_lum_csv = sol_fn, columns = sol_cols):
    df = pd.read_csv(sol_lum_csv, names=columns, header=1)
    df = df[[columns[0], columns[1]]]
    df[columns[0]] = (4.57 - df[columns[0]]) * 1e9 # [Gyr -> yr]
    return df


def sun_scalar(supply_rate, time_arr):
    '''
    Returns the inputted supply rate multipled by the solar luminosity factor given from Bachall, 2001. 
    '''

    # Bachall, 2001
    df_sol = read_sol_lum()

    # Multiply supply rate by ones length of time array to get supply rate through entire model time. 
    sol_wind_mass = np.ones(len(time_arr)) * supply_rate

    # [::-1] reverses order of array since interp wants things increasing. We flip everything back later
    interp_lum = np.interp(-time_arr, -df_sol['Age'], df_sol['Luminosity'])
    
    # Multiply flipped interpolated luminosities by the sol_wind_mass array to get scaled solar wind mass. 
    sol_wind_mass_scaled = interp_lum * sol_wind_mass

    return sol_wind_mass_scaled

def solar_wind(sol_wind_mode, time_arr = TIME_ARR, fys=True, ts = timestep):
    '''
    Returns array with time of kg of ice throughout the lunar surface. 
    Does not account for movement of ice, only deposition given a supply rate from either Benna et al. 2019 or Hurely et al. 2017

    For "Benna" mode, the water supply rate is 2 g/s, 
    For Lucey-Hurley mode, the H2 supply rate is 30 g/s but is converted to water assuming 1 part per thousand is eventually converted to water after Lucey et al. 2020. 

    If fys = True, the array is run through the faint young sun function which scales this value by the luminosity of the sun at all timesteps

    '''
    if sol_wind_mode == "Benna":
        volatile_supply_rate = 2 * 1e-3 # [g/s] * [kg/g] = [kg/s]  |   Benna et al. 2019; Arnold, 1979; Housley et al. 1973 
    elif sol_wind_mode == "Lucey-Hurley":
        volatile_supply_rate = 30 * 1e-3 * 1e-3 # [g/s] * [kg/g] * [1/1000] [ppt] = [kg/s] |  Lucey et al. 2020, Hurley et al. 2017

    supply_rate_ts = volatile_supply_rate * 60 * 60 * 24 * 365 * ts # convert to kg per timestep

    # Faint young sun
    if fys == True: 
        solar_wind_ice = sun_scalar(supply_rate_ts, time_arr)
    else:
        solar_wind_ice = np.ones(len(time_arr)) * supply_rate_ts

    
    return solar_wind_ice

solar_ice = solar_wind('Lucey-Hurley')
