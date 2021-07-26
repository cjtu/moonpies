"""Test MoonPIES module."""
import numpy as np
from moonpies import default_config
from moonpies import moonpies as mm

CFG = default_config.Cfg()

def test_volcanic_ice_head():
    
    timestart = 4.25e9
    timeend = 0
    timestep = 10e6
    n = int((timestart - timeend) / timestep)

    TIME_ARR = np.linspace(timestart, timestep, n, dtype=np.float32)
    hopEfficS = 0.054

    age = 3.65

    if age >= 2.01 and age <=4:
    
        if age >=3.01:
            iceMassVulxS = 1E7*.75*3000*(1000**3)*(10/1E6)*(1E7/1E9)
        else:
            iceMassVulxS = 1E7*.25*3000*(1000**3)*(10/1E6)*(1E7/1E9)
            
    else:
        iceMassVulxS = 0

    expected = iceMassVulxS


    head_volc_arr = mm.volcanic_ice_head(
        TIME_ARR,
        CFG.timestep,
        CFG.volc_early,
        CFG.volc_late,
        CFG.volc_early_pct,
        CFG.volc_late_pct,
        CFG.volc_total_vol,
        CFG.volc_h2o_ppm,
        CFG.volc_magma_density,
        CFG.dtype,
    )
    time_of_interest = np.where(TIME_ARR == age*1e9)
    actual = list(head_volc_arr[time_of_interest])
    print(format(actual[0],'.2E'))
    print(format(expected,'.2E'))
    # np.testing.assert_allclose(actual, expected, rtol=0.35)

    np.testing.assert_approx_equal(actual[0],expected)
# def test_volcanic_ice_NK():
#     cfg = default_config.Cfg(mode='cannon')
#     time_Arr = np.array([3.9e9 3.8e9])
#     actual = mm.volcanic_ice_NK(time_arr, cfg)
#     expected=np.array([0.0, 0.0])