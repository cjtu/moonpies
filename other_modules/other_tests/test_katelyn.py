"""Test MoonPIES module."""
import numpy as np
from . import default_config
from . import moonpies as mm

CFG = default_config.Cfg()

def cannon2020_ds01_ice_erosion(totalIceS,t,ej_col):
    """Cannon ds01 impact gardening module"""
    erosionBase = 0
    for time_idx in range(0,t+1): #set erosionBase which could be set on previous timestep
        if ej_col[time_idx]>0.4:
            erosionBase = time_idx
    iceEroded = 0.1       
    layer = t
    while iceEroded > 0:
        if t > erosionBase:
            if totalIceS[layer] >= iceEroded:
                totalIceS[layer] = totalIceS[layer]-iceEroded
                iceEroded = 0
            else:
                iceEroded = iceEroded-totalIceS[layer]
                totalIceS[layer] = 0
                layer = layer-1
        else:
            break
    return totalIceS

def test_erode_ice_cannon_zero_ejecta():
    """Testing for no ejecta deposited"""
    ice_col = np.array([1,1]) #ice_arr
    ej_col = np.array([0,0]) #test the amount of ejecta
    totalIceS = ice_col.copy()
    t = len(ice_col)-1 #t = index of current timestep (length of ice col)
    actual = mm.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    # print(actual,expected)
    # raise
    np.testing.assert_array_almost_equal(actual,expected)

def test_erode_ice_cannon_ejecta_layer_lessthan_gard_depth():
    """testing for thin ejecta layer < gardening depth (0.1)"""
    ice_col = np.array([1,1]) #ice_arr
    ej_col = np.array([0.05,0.05]) #test the amount of ejecta
    totalIceS = ice_col.copy()
    t = len(ice_col)-1 #t = index of current timestep (length of ice col)
    actual = mm.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    np.testing.assert_array_almost_equal(actual,expected)

def test_erode_ice_cannon_ejecta_layer_equals_gard_depth():
    """testing for ejecta layer = gardening depth (0.1)"""
    ice_col = np.array([1,1]) #ice_arr
    ej_col = np.array([0.1,0.1]) #test the amount of ejecta
    totalIceS = ice_col.copy()
    t = len(ice_col)-1 #t = index of current timestep (length of ice col)
    actual = mm.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    # print(actual,expected)
    # raise
    np.testing.assert_array_almost_equal(actual,expected)

def test_erode_ice_cannon_thicker_than_gard_depth_thinner_than_shield():
    """testing for ejecta layer > gardening depth (0.1) but < ejecta shield (0.4)"""
    ice_col = np.array([1,1]) #ice_arr
    ej_col = np.array([0.2,0.2]) #test the amount of ejecta
    totalIceS = ice_col.copy()
    t = len(ice_col)-1 #t = index of current timestep (length of ice col)
    actual = mm.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    # print(actual,expected)
    # raise
    np.testing.assert_array_almost_equal(actual,expected)

def test_erode_ice_cannon_thick_ejecta():
    """testing for ejecta layer > ejecta shield (0.4)"""
    ice_col = np.array([1,1]) #ice_arr
    ej_col = np.array([0.5,0.5]) #test the amount of ejecta
    totalIceS = ice_col.copy()
    t = len(ice_col)-1 #t = index of current timestep (length of ice col)
    actual = mm.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    #print(actual,expected)
    #raise
    np.testing.assert_array_almost_equal(actual,expected)

def test_erode_ice_cannon_ej_equals_ejecta_shield():
    """Tests for ejecta layer = ejecta shield (0.4)"""
    ice_col = np.array([1,1]) 
    ej_col = np.array([0.4,0.4])
    totalIceS = ice_col.copy()
    t = len(ice_col)-1 
    actual = mm.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    # print(actual,expected)
    # raise    
    np.testing.assert_array_almost_equal(actual,expected)

def test_erode_ice_cannon_long_cols():
    """Testing longer ice and ejecta cols"""
    ice_col = np.array([1,1,2,1,3,1]) #ice_arr
    ej_col = np.array([0.5,0.5,0.5,0.1,0,0.3]) #test the amount of ejecta
    totalIceS = ice_col.copy()
    t = len(ice_col)-1 #t = index of current timestep (length of ice col)
    actual = mm.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    # print(actual,expected)
    # raise    
    np.testing.assert_array_almost_equal(actual,expected)

def test_erode_ice_cannon_thin_ice_sheets():
    """testing for thin ice layers"""
    ice_col = np.array([0.05,0.1,0,0.05,0.2,0,0.1]) #ice_arr
    ej_col = np.array([0.1,0.1,0,0,1,0,0.1]) #test the amount of ejecta
    totalIceS = ice_col.copy()
    t = len(ice_col)-1 #t = index of current timestep (length of ice col)
    actual = mm.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    #print(actual,expected)
    #raise     
    np.testing.assert_array_almost_equal(actual,expected)

def test_erode_ice_cannon_thick_ejecta_on_top():
    """testing for a thick ejecta layer on top"""
    ice_col = np.array([0.05,0.1,0,0.05,0.2,0,0]) #ice_arr
    ej_col = np.array([0.1,0.1,0,0,1,0,20]) #test the amount of ejecta
    totalIceS = ice_col.copy()
    t = len(ice_col)-1 #t = index of current timestep (length of ice col)
    actual = mm.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    #print(actual,expected)
    #raise     
    np.testing.assert_array_almost_equal(actual,expected)