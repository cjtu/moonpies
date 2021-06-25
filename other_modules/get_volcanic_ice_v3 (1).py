#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:58:29 2021

@author: tylerpaladino
"""
import numpy as np
import pandas as pd


VOLCANIC_CSV = '/Users/tylerpaladino/Documents/ISU/LPI_NASA/Needham_tab.xlsx'
TIMESTEP = 10e6  # [yr]
TIMESTART = 4.3e9  # [yr]


def get_volcanic_ice(fvolcanic=VOLCANIC_CSV, dt=TIMESTEP, timestart=TIMESTART, scheme='NK', pole_perc=.01, col='min_H2O'):
    """
    Created on Sun Jun  6 11:58:29 2021
    Function to read in Needham and Kring, (2017) transient atmosphere data. Outputs wrangled data that's been scaled
    appropriately based on polar area and a constant of how much material may have made it to the pole.

    inputs:
    fvolcanic - string dictating path of N*K data file
    dt - timestep in main model
    timestart - initial start time given in main model
    scheme - Either 'NK' , 'Head', or 'Cannon'. Head and Cannon will output the same thing
    pole_perc - the percentage of material that makes it to the pole (given as a decimal)
    col - The atmospheric species that is being deposited. Choices include 'min_CO', 'max_CO', 'min_H2O', 'max_H2O',
            'min_H', 'max_H', 'min_S', 'max_S'

    outputs:
    out - an array the same length as TIME_ARR filled with values of a volatile species deposited in grams.


    @author: tylerpaladino
    """
    TIME_ARR = np.linspace(timestart, dt, int(timestart / dt))
    tstart_ga = timestart/1e9
    out = np.zeros(len(TIME_ARR))
    if scheme.upper() == 'NK':


        r = 1737  # km
        theta = 6 * (np.pi/180)  # radians
        AOI = (2 * np.pi * r**2) * (1-np.cos(theta))  # surface area of south pole region
        # km^2 Poli︠a︡nin, A. D., & Manzhirov, A. V. (2007). Handbook of mathematics for engineers and scientists. Boca Raton, FL: Chapman & Hall/CRC.

        # surface area of entire moon
        surf_area_moon = 4 * np.pi * r**2  # km^2

        AOI_frac = AOI/surf_area_moon

        # Read in transient atmosphere data
        # fvolcanic = '/Users/tylerpaladino/Documents/ISU/LPI_NASA/Needham_tab.xlsx'
        cols = ('age', 'tot_vol', 'sphere_mass', 'min_CO', 'max_CO',
                'min_H2O', 'max_H2O', 'min_H', 'max_H', 'min_S', 'max_S', 'min_sum',
                'max_sum', 'min_psurf', 'max_psurf', 'min_atm_loss', 'max_atm_loss')

        df = pd.read_excel(fvolcanic, names=cols, skiprows=3)

        # Sort in descending order
        sdf = df.sort_values('age', ascending=False).reset_index().drop(columns=['index'])

        # Either remove old ages outside of range specified in input args or add on
        # ages to beginning of df to match input args
        if tstart_ga <= max(df.age):
            new_df = sdf.loc[list(sdf.age).index(tstart_ga):]
            new_df.reset_index(drop=True)
        else:
            year_diff = tstart_ga - max(df.age)
            num_rows = int(year_diff/0.1)  # 0.1 is interval between years in atmosphere data

            ages_needed = np.linspace(max(TIME_ARR)/1e9,
                                      max(df.age),
                                      num_rows,
                                      endpoint=False)

            new_array = np.zeros((len(ages_needed), df.shape[1]))
            # Place new ages into zeros array in correct position (0th)
            new_array[:, 0] = ages_needed

            new_df_row = pd.DataFrame(new_array, columns=cols)
            # Concatenate
            new_df = pd.concat([new_df_row, sdf]).reset_index(drop=True)

        # Add on min of TIME_ARR to end of df
        temp = pd.DataFrame([np.zeros(new_df.shape[1])], columns=cols)

        new_df = new_df.append(temp, ignore_index=True)

        # Insert age in years to df
        new_df.insert(1, 'age_yrs', new_df.age*1e9, True)  # Add in column of ages in years

        # Figure out how many data entries we need to create in new array to match TIME_ARR
        div_num = int(round(new_df.age_yrs[0]-new_df.age_yrs[1])/dt)
        # Loop through new_df and split up values based on div num into equal parts into out var.
        for i, row in new_df.iterrows():
            out[i*div_num:(i+1) * div_num] = row[col]/div_num

        return out * AOI_frac * pole_perc

    elif scheme.upper() == 'HEAD' or scheme == 'Cannon':
        tot_eruption_vol = 10**7  # km^3 basalt
        outgassed_H2O = 10  # ppm
        magma_rho = 3000  # kg/m^3

        tot_H2O_dep = tot_eruption_vol * (1000**3) * magma_rho * (outgassed_H2O/1E6) * 1000 # Deposited ice in grams

        volc_epoch_idx = np.zeros(3, dtype=int)

        H2O_dep_arr = np.array([tot_H2O_dep*.75, tot_H2O_dep*.25])  # 75 % and 25 %

        volc_epoch_idx[0] = (timestart - 4*1e9)/dt
        volc_epoch_idx[1] = (timestart - 3*1e9)/dt
        volc_epoch_idx[2] = (timestart - 2*1e9)/dt
        num_rows = volc_epoch_idx[1] - volc_epoch_idx[0] # How much to divide the total deposited value in each era.

        for row, val in enumerate(TIME_ARR):
            if volc_epoch_idx[0] <= row < volc_epoch_idx[1]:
                out[row] = H2O_dep_arr[0]/num_rows
            elif volc_epoch_idx[1] <= row < volc_epoch_idx[2]:
                out[row] = H2O_dep_arr[1]/num_rows

        return out

test = get_volcanic_ice(scheme='HEAD')
