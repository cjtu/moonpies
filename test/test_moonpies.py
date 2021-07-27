"""Test MoonPIES module."""
import numpy as np
from unittest.mock import patch
from moonpies import default_config
from moonpies import moonpies as mp

CFG = default_config.Cfg()


def test_volcanic_ice_head():
    """Test volcanic_ice_head()."""
    time_arr = mp.get_time_array(CFG)
    time_arr_rounded = np.rint(time_arr/1e7) / 100
    iceMassVulxS = np.zeros(len(time_arr))

    for t, age in enumerate(time_arr_rounded):
        # Cannon volcanic ice mass [kg]
        if age > 2.00 and age < 4.01:
            if age > 3.00:
                print(f'{round(age, 2):.12f}')
                iceMassVulxS[t] = 1E7*.75*3000*(1000**3)*(10/1E6)*(1E7/1E9)
            else:
                iceMassVulxS[t] = 1E7*.25*3000*(1000**3)*(10/1E6)*(1E7/1E9)
        else:
            iceMassVulxS[t] = 0
    expected = iceMassVulxS

    actual = mp.volcanic_ice_head(
        time_arr,
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
    diff =(actual - expected) > 1e5 
    print(np.where(diff))
    print(actual[diff], expected[diff])
    np.testing.assert_allclose(actual, expected)


def test_impact_flux():
    """Test impact_flux ratios."""
    # Test first timestep
    t_yrs = 4.25e9
    actual = mp.impact_flux(t_yrs) / mp.impact_flux(0)

    # Cannon 2020 ds02 eq
    t_ga = t_yrs * 1e-9
    expected = (3.76992e-13 * (np.exp(6.93 * t_ga)) + 8.38e-4) / (
        3.76992e-13 * (np.exp(6.93 * 0)) + 8.38e-4
    )
    np.testing.assert_approx_equal(actual, expected)

    # Test 3 Ga
    t_yrs = 3e9
    actual = mp.impact_flux(t_yrs) / mp.impact_flux(0)
    # Cannon 2020 ds02 eq
    t_ga = t_yrs * 1e-9
    expected = (3.76992e-13 * (np.exp(6.93 * t_ga)) + 8.38e-4) / (
        3.76992e-13 * (np.exp(6.93 * 0)) + 8.38e-4
    )
    np.testing.assert_allclose(actual, expected)


def test_ice_micrometeorites():
    """Test ice_micrometeorites."""
    t = 4.25e9
    actual = mp.ice_micrometeorites(t)
    # Cannon 2020 ds02 Regime A: micrometeorites
    totalImpactorWater = 1e6 * 1e7 * 0.1
    totalImpactorWater = (
        totalImpactorWater * mp.impact_flux(t) / mp.impact_flux(0)
    )
    expected = totalImpactorWater * 0.165
    np.testing.assert_allclose(actual, expected)

    t = 3e9
    actual = mp.ice_micrometeorites(t)
    # Cannon 2020 ds02 Regime A: micrometeorites
    totalImpactorWater = 1e6 * 1e7 * 0.1
    totalImpactorWater = (
        totalImpactorWater * mp.impact_flux(t) / mp.impact_flux(0)
    )
    expected = totalImpactorWater * 0.165
    np.testing.assert_allclose(actual, expected)

    t = 0
    actual = mp.ice_micrometeorites(t)
    # Cannon 2020 ds02 Regime A: micrometeorites
    totalImpactorWater = 1e6 * 1e7 * 0.1
    totalImpactorWater = (
        totalImpactorWater * mp.impact_flux(t) / mp.impact_flux(0)
    )
    expected = totalImpactorWater * 0.165
    np.testing.assert_allclose(actual, expected)


def test_get_impactors_brown():
    """Test get_impactors_brown()."""
    mindiam = 0.01
    maxdiam = 3
    ts = 10e6
    actual = mp.get_impactors_brown(mindiam, maxdiam, ts)
    # Cannon 2020 ds02 Regime B: small impactors
    c = 1.568
    d = 2.7
    impactorNumGtLow = 10 ** (c - d * np.log10(mindiam))
    impactorNumGtHigh = 10 ** (c - d * np.log10(maxdiam))
    impactorNum = impactorNumGtLow - impactorNumGtHigh
    impactorNum = impactorNum * 1e7
    impactorNum = impactorNum / 22.5
    expected = impactorNum
    np.testing.assert_approx_equal(actual, expected)


def test_ice_small_impactors():
    """Test ice_small_impactors."""
    t = 4.25e9
    diams, ncraters = mp.get_impactor_pop(t, "b", CFG.timestep, CFG.diam_range, CFG.sfd_slopes)
    actual = mp.ice_small_impactors(diams, ncraters, CFG)

    # Brown (tested above)
    mindiam = diams[0]
    maxdiam = diams[-1]
    impactorNum = mp.get_impactors_brown(mindiam, maxdiam, CFG.timestep)
    impactorDiams = diams

    # Cannon 2020 ds02 Regime B: small impactors
    impactorNum = impactorNum * mp.impact_flux(t) / mp.impact_flux(0)
    sfd = impactorDiams ** -3.7
    impactors = sfd * (impactorNum / np.sum(sfd))
    np.testing.assert_allclose(ncraters, impactors, rtol=5e-7)

    # Cannon 2020: convert to mass
    impactorMasses = 1300 * (4 / 3) * np.pi * (impactorDiams / 2) ** 3
    totalImpactorMass = np.sum(impactorMasses * impactors)
    totalImpactorWater = totalImpactorMass * 0.36 * (2 / 3) * 0.1
    expected = totalImpactorWater * 0.165
    np.testing.assert_allclose(actual, expected)


def test_get_crater_pop_C():
    """Test get_crater_pop on regime C."""
    t = 4.25e9
    regime = "c"
    rng = 0
    diams, ncraters = mp.get_crater_pop(t, 
                regime, 
                CFG.timestep,
                CFG.diam_range,
                CFG.sfd_slopes,
                CFG.sa_moon,
                CFG.ivanov2000,
                rng=rng)

    # Cannon regime C
    craterDiams = diams  # m
    craterNum = mp.neukum(craterDiams[0], CFG.ivanov2000) - mp.neukum(craterDiams[-1], CFG.ivanov2000)
    craterNum = craterNum * (1e7)
    craterNum = craterNum * CFG.sa_moon
    craterNum = craterNum * mp.impact_flux(t) / mp.impact_flux(0)

    sfd = craterDiams ** -3.82
    craters = sfd * (craterNum / sum(sfd))
    np.testing.assert_allclose(ncraters, craters, rtol=5e-7)


def test_ice_small_craters():
    """Test ice_small_craters."""
    t = 4.25e9
    regime = "c"
    rng = 0
    diams, ncraters = mp.get_crater_pop(t, 
                regime, 
                CFG.timestep,
                CFG.diam_range,
                CFG.sfd_slopes,
                CFG.sa_moon,
                CFG.ivanov2000,
                rng=rng)
    actual = mp.ice_small_craters(diams, ncraters, regime, CFG)

    # Cannon regime C
    v = 20e3  # [m/s]
    impactorDiams = mp.diam2len(diams, v, regime, CFG)
    # impactorDiams = dToL_C(craterDiams*1000,20)
    impactorMasses = 1300 * (4 / 3) * np.pi * (impactorDiams / 2) ** 3
    totalImpactorMass = sum(impactorMasses * ncraters)
    totalImpactorWater = totalImpactorMass * 0.36 * (2 / 3) * 0.1

    iceMassRegimeC = totalImpactorWater * 0.165
    expected = iceMassRegimeC
    np.testing.assert_approx_equal(actual, expected)


def test_ice_large_craters_D():
    """Test ice_large_craters regime D."""
    regime = "d"
    rng = 0

    # Cannon 2020: Regime D
    def cannon_D(crater_diams, impactor_speeds):
        impactorSpeeds = impactor_speeds * 1e-3  # [km/s]
        impactorDiams = mp.diam2len(crater_diams, impactor_speeds, regime, CFG)
        impactorMasses = 1300 * (4 / 3) * np.pi * (impactorDiams / 2) ** 3
        waterRetained = np.zeros(len(impactorSpeeds))
        for s in range(len(waterRetained)):
            if impactorSpeeds[s] < 10:
                waterRetained[s] = 0.5
            else:
                waterRetained[s] = 36.26 * np.exp(-0.3464 * impactorSpeeds[s])
        waterRetained[waterRetained < 0] = 0
        waterMasses = impactorMasses * waterRetained * 0.1
        iceMasses = np.zeros(len(waterMasses))
        for w in range(len(waterMasses)):
            iceMasses[w] = waterMasses[w]
        iceMassRegimeD = np.sum(iceMasses)
        return iceMassRegimeD

    # Test 4.25 Ga
    t = 4.25e9
    diams = mp.get_crater_pop(t, 
                regime,
                CFG.timestep,
                CFG.diam_range,
                CFG.sfd_slopes,
                CFG.sa_moon,
                CFG.ivanov2000,
                rng=rng)
    crater_diams = mp.get_random_hydrated_craters(diams)
    impactor_speeds = mp.get_random_impactor_speeds(len(crater_diams))
    actual = mp.ice_large_craters(crater_diams, impactor_speeds, regime, CFG)
    expected = cannon_D(crater_diams, impactor_speeds)
    np.testing.assert_approx_equal(actual, expected)

    # Test 4.25 Ga
    t = 3e9
    diams = mp.get_crater_pop(t, 
                regime,
                CFG.timestep,
                CFG.diam_range,
                CFG.sfd_slopes,
                CFG.sa_moon,
                CFG.ivanov2000,
                rng=rng)
    crater_diams = mp.get_random_hydrated_craters(diams)
    impactor_speeds = mp.get_random_impactor_speeds(len(crater_diams))
    actual = mp.ice_large_craters(crater_diams, impactor_speeds, regime, CFG)
    expected = cannon_D(crater_diams, impactor_speeds)
    np.testing.assert_approx_equal(actual, expected)


def test_ice_large_craters_E():
    """Test ice_large_craters regime E."""
    regime = "e"
    rng = 0

    # Cannon 2020 Regime E
    def cannon_E(crater_diams, impactor_speeds):
        impactorSpeeds = impactor_speeds * 1e-3  # [km/s]
        impactorDiams = mp.diam2len(crater_diams, impactor_speeds, regime, CFG)
        impactorMasses = 1300 * (4 / 3) * np.pi * (impactorDiams / 2) ** 3

        waterRetained = np.zeros(len(impactorSpeeds))

        for s in range(len(waterRetained)):
            if impactorSpeeds[s] < 10:
                waterRetained[s] = 0.5
            else:
                waterRetained[s] = 36.26 * np.exp(-0.3464 * impactorSpeeds[s])
        waterRetained[waterRetained < 0] = 0
        waterMasses = impactorMasses * waterRetained * 0.1
        iceMasses = np.zeros(len(waterMasses))
        for w in range(len(waterMasses)):
            iceMasses[w] = waterMasses[w]
        iceMassRegimeE = sum(iceMasses)
        return iceMassRegimeE

    # Test 4.25 Ga
    t = 4.25e9
    diams = mp.get_crater_pop(t, 
                regime,
                CFG.timestep,
                CFG.diam_range,
                CFG.sfd_slopes,
                CFG.sa_moon,
                CFG.ivanov2000,
                rng=rng)
    crater_diams = mp.get_random_hydrated_craters(diams)
    impactor_speeds = mp.get_random_impactor_speeds(len(crater_diams))
    actual = mp.ice_large_craters(crater_diams, impactor_speeds, regime, CFG)
    expected = cannon_E(crater_diams, impactor_speeds)
    np.testing.assert_approx_equal(actual, expected)

    # Test 3 Ga
    t = 3
    diams = mp.get_crater_pop(t, 
                regime,
                CFG.timestep,
                CFG.diam_range,
                CFG.sfd_slopes,
                CFG.sa_moon,
                CFG.ivanov2000,
                rng=rng)
    crater_diams = mp.get_random_hydrated_craters(diams)
    impactor_speeds = mp.get_random_impactor_speeds(len(crater_diams))
    actual = mp.ice_large_craters(crater_diams, impactor_speeds, regime, CFG)
    expected = cannon_E(crater_diams, impactor_speeds)
    np.testing.assert_approx_equal(actual, expected)


def test_get_ice_thickness():
    global_ice = 20e3
    polar_ice_mass = global_ice * 0.054
    ice_volume = polar_ice_mass / 934
    ice_thickness = ice_volume / (1.3e4 * 1e6)
    actual = mp.get_ice_thickness(global_ice, CFG)
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
    actual = mp.get_ice_thickness(total_ice, CFG)
    expected = sp_ice_thickness
    np.testing.assert_almost_equal(actual,expected)


# Test Cannon impact gardening
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
    actual = mp.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
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
    actual = mp.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    np.testing.assert_array_almost_equal(actual,expected)

def test_erode_ice_cannon_ejecta_layer_equals_gard_depth():
    """testing for ejecta layer = gardening depth (0.1)"""
    ice_col = np.array([1,1]) #ice_arr
    ej_col = np.array([0.1,0.1]) #test the amount of ejecta
    totalIceS = ice_col.copy()
    t = len(ice_col)-1 #t = index of current timestep (length of ice col)
    actual = mp.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
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
    actual = mp.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
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
    actual = mp.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
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
    actual = mp.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
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
    actual = mp.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
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
    actual = mp.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
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
    actual = mp.erode_ice_cannon(ice_col, ej_col, t, overturn_depth=0.1, ej_shield=0.4)
    expected = cannon2020_ds01_ice_erosion(totalIceS,t,ej_col) #final ice col from cannon
    #print(actual,expected)
    #raise     
    np.testing.assert_array_almost_equal(actual,expected)


def test_get_diam_array():
    """Test get_diam_array"""
    for regime in ("b", "c", "d", "e"):
        low, upp, step = CFG.diam_range[regime]
        actual = mp.get_diam_array(low, upp, step)
        expected = np.linspace(low, upp, int((upp - low) / step) + 1)
        np.testing.assert_array_almost_equal(actual, expected)


def test_final2transient():
    """Test final2transient."""
    actual = mp.final2transient(np.array([1500, 15e3, 299e3]))
    expected = (1200, 12e3, 173125.614057)
    np.testing.assert_array_almost_equal(actual, expected)


def test_diam2len_prieur():
    """Test diam2len_prieur."""

    pass


def test_diam2len_collins():
    """Test diam2len_collins."""
    # Test against Melosh 1989 Purdue Impact Calculator
    # Pi scaling into hard rock target
    diams = np.array([1.2e3, 12e3])
    actual = mp.diam2len_collins(diams, 20e3)
    expected = (3.87e1, 7.42e2)
    np.testing.assert_allclose(actual, expected, rtol=0.005)


def test_diam2len_johnson():
    """Test diam2len_johnson."""
    # Testing fig 6, Johnson et al. 2016
    diams = np.array([6.5e3, 20e3, 100e3])
    actual = mp.diam2len_johnson(
            diams,
            20e3,
            1300, 
            1300,            
            CFG.grav_moon,
            CFG.impact_angle,
            CFG.simple2complex,)
    expected = (0.37e3, 1.4e3, 8.5e3)
    np.testing.assert_allclose(actual, expected, rtol=0.35)


def test_diam2len_potter():
    """Test diam2len_potter."""
    # Testing fig 6, Potter et al. 2016
    diams = np.array([208629, 255489, 343091])
    actual = mp.diam2len_potter(
            diams,
	    v=17e3,
	    rho_i=1300,
	    rho_t=1500,
	    g=1.62,
	    )
    expected = ([41113, 55351, 85316]) # TODO
    np.testing.assert_allclose(actual, expected, rtol=0.35)


def test_garden_ice_column():
    """Test garden_ice_column."""

    # First garden ice, then protect with ejecta TODO: check
    ice_col = np.array([10])
    ejecta_col = np.array([10])
    overturn_depth = 10
    new_ice_col = mp.garden_ice_column(ice_col, ejecta_col, overturn_depth)
    expected = [0]
    np.testing.assert_array_almost_equal(new_ice_col, expected)


    ice_col = np.array([10])
    ejecta_col = np.array([5])
    overturn_depth = 3
    new_ice_col = mp.garden_ice_column(ice_col, ejecta_col, overturn_depth)
    expected = [7]
    np.testing.assert_array_almost_equal(new_ice_col, expected)


    ice_col = np.array([10, 10])
    ejecta_col = np.array([5, 0])
    overturn_depth = 15
    new_ice_col = mp.garden_ice_column(ice_col, ejecta_col, overturn_depth)
    expected = [5, 0]
    np.testing.assert_array_almost_equal(new_ice_col, expected)

    ice_col = np.array([6, 4, 2])
    ejecta_col = np.array([0, 2, 1])
    overturn_depth = 10
    new_ice_col = mp.garden_ice_column(ice_col, ejecta_col, overturn_depth)
    expected = [0, 4, 0]
    np.testing.assert_array_almost_equal(new_ice_col, expected)


# Acceptance test for Cannon ds01
@patch('moonpies.moonpies.get_ejecta_thickness_matrix')
@patch('moonpies.moonpies.total_impact_ice')
def test_cannon_ds01(mock_total_impact_ice, mock_ej_thick_matrix):
    """
    Test Cannon mode produces same ice cols as main Cannon model.
    
    # Mock ejecta thicknesses [m] at following timesteps
    # ejS[0] = 0.05  # 4.20 Ga
    # ejS[1] = 0.3  # 4.15 Ga
    # ejS[2] = 0.4  # 4.10 Ga
    # ejS[3] = 0.5  # 4.05 Ga

    # Compare to changes in ice column outputs in MatLAB
    # First row: 4.3474 m
    # 4.2 Ga: no change (1st ej layer)
    # 4.15 Ga: no change (2nd ej layer)
    # 4.10 Ga: no change (3rd ej layer)
    # 4.05 Ga: 4.3574 m for 1 timestep (4th ej layer)
    # 4.0 Ga: 4.3574 m until 3.0 Ga (early volcanism
    # 3.0 Ga: 4.3507 m until 2.0 Ga (late volcanism)
    # 2.0 Ga: 4.3474 m until 0 Ga
    """
    cfg = default_config.Cfg()
    cfg.mode = "cannon"
    ncraters = len(mp.read_crater_list(cfg.crater_csv_in, cfg.crater_cols))
    ntime = len(mp.get_time_array(cfg))

    # Mock ejecta thickness the same way as in Cannon
    ej_thickness = np.zeros((ntime, ncraters))
    bsed_time = np.zeros((ntime, len(cfg.coldtrap_craters)))
    ej_thickness[5] = 0.05
    ej_thickness[10] = 0.3
    ej_thickness[15] = 0.4
    ej_thickness[20] = 0.5
    mock_ej_thick_matrix.return_value = (ej_thickness, ['']*ntime, bsed_time)

    # Mock total_impact_ice same way as in Cannon (1e15 at every timestep)
    mock_total_impact_ice.return_value = 1e15

    out = mp.main(cfg)
    expected = out[1].Haworth.values  # output: ej_col, ice_col

    # Test first timestep
    np.testing.assert_approx_equal(expected[0], 4.3474, 4)

    # First ejecta layer (0.05) doesn't block gardening
    np.testing.assert_approx_equal(expected[4], 4.3474, 4)
    np.testing.assert_approx_equal(expected[5], 4.3474, 4)
    np.testing.assert_approx_equal(expected[6], 4.3474, 4)

    # Second ejecta layer (0.3) doesn't block gardening in Cannon (but should)
    np.testing.assert_approx_equal(expected[9], 4.3474, 4)
    np.testing.assert_approx_equal(expected[10], 4.3474, 4)
    np.testing.assert_approx_equal(expected[11], 4.3474, 4)

    # Third ejecta layer (0.4) blocks gardening
    np.testing.assert_approx_equal(expected[14], 4.3474, 4)
    np.testing.assert_approx_equal(expected[15], 4.3474, 4)
    np.testing.assert_approx_equal(expected[16], 4.3474, 4)

    # Fourth ejecta layer (0.5) block gardening
    np.testing.assert_approx_equal(expected[19], 4.3474, 4)
    np.testing.assert_approx_equal(expected[20], 4.4474, 4)
    np.testing.assert_approx_equal(expected[21], 4.3474, 4)

    # Onset of early volcanism at 4.0 Ga
    np.testing.assert_approx_equal(expected[24], 4.3474, 4)
    np.testing.assert_approx_equal(expected[25], 4.3574, 4)
    np.testing.assert_approx_equal(expected[26], 4.3574, 4)

    # Onset of late volcanism at 3.0 Ga
    np.testing.assert_approx_equal(expected[124], 4.3574, 4)
    np.testing.assert_approx_equal(expected[125], 4.3507, 4)
    np.testing.assert_approx_equal(expected[126], 4.3507, 4)

    # Late volcanism stops at 2.0 Ga
    np.testing.assert_approx_equal(expected[224], 4.3507, 4)
    np.testing.assert_approx_equal(expected[225], 4.3474, 4)
    np.testing.assert_approx_equal(expected[226], 4.3474, 4)

    # Test final timestep
    np.testing.assert_approx_equal(expected[0], 4.3474, 4)