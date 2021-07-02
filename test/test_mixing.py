"""Test mixing module."""
from essi21 import mixing as mm
import numpy as np


def test_impact_flux():
    """Test impact_flux ratios."""
    # Test first timestep
    t_yrs = 4.25e9
    actual = mm.impact_flux(t_yrs) / mm.impact_flux(0)

    # Cannon 2020 ds02 eq
    t_ga = t_yrs * 1e-9
    expected = (3.76992e-13 * (np.exp(6.93 * t_ga)) + 8.38e-4) / (
        3.76992e-13 * (np.exp(6.93 * 0)) + 8.38e-4
    )
    np.testing.assert_approx_equal(actual, expected)

    # Test 3 Ga
    t_yrs = 3e9
    actual = mm.impact_flux(t_yrs) / mm.impact_flux(0)
    # Cannon 2020 ds02 eq
    t_ga = t_yrs * 1e-9
    expected = (3.76992e-13 * (np.exp(6.93 * t_ga)) + 8.38e-4) / (
        3.76992e-13 * (np.exp(6.93 * 0)) + 8.38e-4
    )
    np.testing.assert_allclose(actual, expected)


def test_ice_micrometeorites():
    """Test ice_micrometeorites."""
    t = 4.25e9
    actual = mm.ice_micrometeorites(t)
    # Cannon 2020 ds02 Regime A: micrometeorites
    totalImpactorWater = 1e6 * 1e7 * 0.1
    totalImpactorWater = (
        totalImpactorWater * mm.impact_flux(t) / mm.impact_flux(0)
    )
    expected = totalImpactorWater * 0.165
    np.testing.assert_allclose(actual, expected)

    t = 3e9
    actual = mm.ice_micrometeorites(t)
    # Cannon 2020 ds02 Regime A: micrometeorites
    totalImpactorWater = 1e6 * 1e7 * 0.1
    totalImpactorWater = (
        totalImpactorWater * mm.impact_flux(t) / mm.impact_flux(0)
    )
    expected = totalImpactorWater * 0.165
    np.testing.assert_allclose(actual, expected)

    t = 0
    actual = mm.ice_micrometeorites(t)
    # Cannon 2020 ds02 Regime A: micrometeorites
    totalImpactorWater = 1e6 * 1e7 * 0.1
    totalImpactorWater = (
        totalImpactorWater * mm.impact_flux(t) / mm.impact_flux(0)
    )
    expected = totalImpactorWater * 0.165
    np.testing.assert_allclose(actual, expected)


def test_get_impactors_brown():
    """Test get_impactors_brown()."""
    mindiam = 0.01
    maxdiam = 3
    actual = mm.get_impactors_brown(mindiam, maxdiam)
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
    diams, ncraters = mm.get_impactor_pop(t, "B")
    actual = mm.ice_small_impactors(diams, ncraters)

    # Brown (tested above)
    mindiam = diams[0]
    maxdiam = diams[-1]
    impactorNum = mm.get_impactors_brown(mindiam, maxdiam)
    impactorDiams = diams

    # Cannon 2020 ds02 Regime B: small impactors
    impactorNum = impactorNum * mm.impact_flux(t) / mm.impact_flux(0)
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
    diams, ncraters = mm.get_crater_pop(t, regime="C")

    # Cannon regime C
    craterDiams = diams  # m
    craterNum = mm.neukum(craterDiams[0]) - mm.neukum(craterDiams[-1])
    craterNum = craterNum * (1e7)
    craterNum = craterNum * mm.SA_MOON
    craterNum = craterNum * mm.impact_flux(t) / mm.impact_flux(0)

    sfd = craterDiams ** -3.82
    craters = sfd * (craterNum / sum(sfd))
    np.testing.assert_allclose(ncraters, craters, rtol=5e-7)


def test_ice_small_craters():
    """Test ice_small_craters."""
    t = 4.25e9
    regime = "C"
    diams, ncraters = mm.get_crater_pop(t, regime=regime)
    actual = mm.ice_small_craters(diams, ncraters, regime)

    # Cannon regime C
    v = 20e3  # [m/s]
    impactorDiams = mm.diam2len(diams, v, regime)
    # impactorDiams = dToL_C(craterDiams*1000,20)
    impactorMasses = 1300 * (4 / 3) * np.pi * (impactorDiams / 2) ** 3
    totalImpactorMass = sum(impactorMasses * ncraters)
    totalImpactorWater = totalImpactorMass * 0.36 * (2 / 3) * 0.1

    iceMassRegimeC = totalImpactorWater * 0.165
    expected = iceMassRegimeC
    np.testing.assert_approx_equal(actual, expected)


def test_ice_large_craters_D():
    """Test ice_large_craters regime D."""
    regime = "D"

    # Cannon 2020: Regime D
    def cannon_D(crater_diams, impactor_speeds):
        impactorSpeeds = impactor_speeds * 1e-3  # [km/s]
        impactorDiams = mm.diam2len(crater_diams, impactor_speeds, regime)
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
    diams = mm.get_crater_pop(t, regime=regime)
    crater_diams = mm.get_random_hydrated_craters(diams)
    impactor_speeds = mm.get_random_impactor_speeds(len(crater_diams))
    actual = mm.ice_large_craters(crater_diams, impactor_speeds, regime)
    expected = cannon_D(crater_diams, impactor_speeds)
    np.testing.assert_approx_equal(actual, expected)

    # Test 4.25 Ga
    t = 3e9
    diams = mm.get_crater_pop(t, regime=regime)
    crater_diams = mm.get_random_hydrated_craters(diams)
    impactor_speeds = mm.get_random_impactor_speeds(len(crater_diams))
    actual = mm.ice_large_craters(crater_diams, impactor_speeds, regime)
    expected = cannon_D(crater_diams, impactor_speeds)
    np.testing.assert_approx_equal(actual, expected)


def test_ice_large_craters_E():
    """Test ice_large_craters regime E."""
    regime = "E"

    # Cannon 2020 Regime E
    def cannon_E(crater_diams, impactor_speeds):
        impactorSpeeds = impactor_speeds * 1e-3  # [km/s]
        impactorDiams = mm.diam2len(crater_diams, impactor_speeds, regime)
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
    diams = mm.get_crater_pop(t, regime=regime)
    crater_diams = mm.get_random_hydrated_craters(diams)
    impactor_speeds = mm.get_random_impactor_speeds(len(crater_diams))
    actual = mm.ice_large_craters(crater_diams, impactor_speeds, regime)
    expected = cannon_E(crater_diams, impactor_speeds)
    np.testing.assert_approx_equal(actual, expected)

    # Test 3 Ga
    t = 3
    diams = mm.get_crater_pop(t, regime=regime)
    crater_diams = mm.get_random_hydrated_craters(diams)
    impactor_speeds = mm.get_random_impactor_speeds(len(crater_diams))
    actual = mm.ice_large_craters(crater_diams, impactor_speeds, regime)
    expected = cannon_E(crater_diams, impactor_speeds)
    np.testing.assert_approx_equal(actual, expected)


def test_get_diam_array():
    """Test get_diam_array"""
    for regime in ("B", "C", "D", "E"):
        actual = mm.get_diam_array(regime)
        low, upp, step = mm.DIAM_RANGE[regime]
        expected = np.linspace(low, upp, int((upp - low) / step) + 1)
        np.testing.assert_array_almost_equal(actual, expected)


def test_final2transient():
    """Test final2transient."""
    actual = mm.final2transient(np.array([1500, 15e3, 299e3]))
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
    actual = mm.diam2len_collins(diams, 20e3)
    expected = (3.87e1, 7.42e2)
    np.testing.assert_allclose(actual, expected, rtol=0.005)


def test_diam2len_johnson():
    """Test diam2len_johnson."""
    # Testing fig 6, Johnson et al. 2016
    diams = np.array([6.5e3, 20e3, 100e3])
    actual = mm.diam2len_johnson(diams, 1300, 1300)
    expected = (0.37e3, 1.4e3, 8.5e3)
    np.testing.assert_allclose(actual, expected, rtol=0.35)


def test_garden_ice_column():
    """Test garden_ice_column."""
    ice_col = np.array([10])
    ejecta_col = np.array([10])
    overturn_depth = 10
    new_ice_col = mm.garden_ice_column(ice_col, ejecta_col, overturn_depth)
    expected = [10]
    np.testing.assert_array_almost_equal(new_ice_col, expected)


    ice_col = np.array([10])
    ejecta_col = np.array([5])
    overturn_depth = 10
    new_ice_col = mm.garden_ice_column(ice_col, ejecta_col, overturn_depth)
    expected = [5]
    np.testing.assert_array_almost_equal(new_ice_col, expected)


    ice_col = np.array([10, 10])
    ejecta_col = np.array([5, 0])
    overturn_depth = 15
    new_ice_col = mm.garden_ice_column(ice_col, ejecta_col, overturn_depth)
    expected = [10, 0]
    np.testing.assert_array_almost_equal(new_ice_col, expected)

    ice_col = np.array([6, 4, 2])
    ejecta_col = np.array([0, 2, 1])
    overturn_depth = 10
    new_ice_col = mm.garden_ice_column(ice_col, ejecta_col, overturn_depth)
    expected = [5, 0, 0]
    np.testing.assert_array_almost_equal(new_ice_col, expected)
