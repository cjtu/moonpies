"""Test MoonPIES module."""
import numpy as np
from moonpies import default_config
from moonpies import moonpies as mm


CFG = default_config.Cfg()


# def test_final2transient():
#     """Test final2transient."""
#     actual = mm.final2transient(np.array([1500, 15e3, 299e3]))
#     expected = (1200, 12e3, 173125.614057)
#     np.testing.assert_array_almost_equal(actual, expected)


def test_get_ice_thickness():
    global_ice = 20e3
    polar_ice_mass = global_ice * 0.054
    ice_volume = polar_ice_mass / 934
    ice_thickness = ice_volume / (1.3e4 * 1e6)
    actual = mm.get_ice_thickness(global_ice, CFG)
    expected = ice_thickness
    np.testing.assert_almost_equal(actual,expected)




def test_get_ice_thickness_cannon(): 
    ice_volc = 10e10
    ice_impact = 10e15
    total_ice = ice_volc + ice_impact

    #IceHopEfficiciency, how much ice gets to poles
    sp_ice_volc_mass = ice_volc*0.054 
    sp_ice_impact_mass = ice_impact*0.054

    #Total Ice Mass
    sp_total_ice = sp_ice_volc_mass + sp_ice_impact_mass 

    #Ice Mass to Volume
    sp_ice_volume = sp_total_ice / 934

    #Ice Volume to Thickness
    sp_ice_thickness = sp_ice_volume / ((1.30e4)*1000*1000)
    actual = mm.get_ice_thickness(total_ice, CFG)
    expected = sp_ice_thickness
    print(actual)

    np.testing.assert_almost_equal(actual,expected)

