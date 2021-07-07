# Ballistic sedimentation module
def ballistic_sed_ice_column(c, ice_column, ballistic_sed_matrix):
    """Return ice column with ballistic sed grid applied"""
    # ballistic_sed_grid = ballistic_sed_matrix[:, :, c]
    # TODO: add code from Kristen
    return ice_column


# def simple2complex_diam(
#     gravity,
#     density=2700,
#     s2c_moon=18e3,
#     g_moon=1.62,
#     rho_moon=2700,
# ):
#     """
#     Return simple-to-complex transition diameter given gravity of body [m s^-2]
#     and density of target [kg m^-3] (Melosh 1989).

#     Parameters
#     ----------
#     gravity [num] : Gravity of a planetary body [m s^-2]
#     density [num] : Density of the target material [kg m^-3]
#     s2c_moon [num] : Lunar simple-to-complex transition diamter [m]
#     g_moon [num] : Gravity of the Moon [m s^-2]
#     rho_moon [num] : Bulk density of the Moon [kg m^-3]

#     Return
#     ------
#     simple2complex (num): Simple-to-complex transition diameter [m]
#     """
#     return g_moon * rho_moon * s2c_moon / (gravity * density)


# def complex2peakring_diam(
#     gravity,
#     density,
#     c2pr_moon=cfg.complex2peakring,
#     g_moon=cfg.grav_moon,
#     rho_moon=cfg.bulk_density,
# ):
#     """
#     Return complex-to-peak ring basin transition diameter given gravity of
#     body [m s^-2] and density of target [kg m^-3] (Melosh 1989).

#     Parameters
#     ----------
#     gravity [num] : Gravity of a planetary body [m s^-2]
#     density [num] : Density of the target material [kg m^-3]
#     c2pr_moon [num] : Lunar complex crater to peak ring transition diamter [m]
#     g_moon [num] : Gravity of the Moon [m s^-2]
#     rho_moon [num] : Bulk density of the Moon [kg m^-3]

#     Return
#     ------
#     complex2peakring (num): Complex-to-peak ring basin transition diameter [m]
#     """
#     return g_moon * rho_moon * c2pr_moon / (gravity * density)


# def get_grd_ind(x, y, grdx, grdy):
#     """Return index of val in monotonic grid_axis."""
#     xidx = np.searchsorted(grdx[0, :], x) - 1
#     yidx = grdy.shape[0] - (np.searchsorted(-grdy[:, 0], y) - 1)
#     return (xidx, yidx)
# def grid_interp(x, y, grdvalues, grdx, grdy):
#     """Return interpolated value x, y in 2D grid."""
#     ix, iy = get_grd_ind(x, y, grdx, grdy)
#     gx, gy = grdx.flatten(), grdy.flatten()

#     dy = (y - gy[iy]) / (gy[iy + 1] - gy[iy])
#     interp_y = (1 - dy) * grdvalues[iy] + dy * grdvalues[iy + 1]

#     dx = (x - gx[ix]) / (gx[ix + 1] - gx[ix])
#     interp = (1 - dx) * interp_y[ix] + dx * interp_y[ix + 1]
#     return interp

# # Sublimation module
# def sublimate_ice_column(ice_column, sublimation_rate):
#     """
#     Return ice column with thickness of ice lost from top according to
#     sublimation rate
#     """
#     # TODO
#     return ice_column

# def ballistic_planar(theta, d, g=1.62):
#     """
#     Return ballistic speed (v) given ballistic range (d) and gravity of planet (g).
#     Assumes planar surface (d << R_planet).

#     Parameters
#     ----------
#     d (num or array): ballistic range [m]
#     g (num): gravitational force of the target body [m s^-2]
#     theta (num): angle of impaact [radians]

#     Returns
#     -------
#     v (num or array): ballistic speed [m s^-1]

#     """
#     return np.sqrt((d * g) / np.sin(2 * theta))


# def ballistic_spherical(theta, d, g=1.62, rp=1737e3):
#     """
#     Return ballistic speed (v) given ballistic range (d) and gravity of planet (g).
#     Assumes perfectly spherical planet (Vickery, 1986).

#     Parameters
#     ----------
#     d (num or array): ballistic range [m]
#     g (num): gravitational force of the target body [m s^-2]
#     theta (num): angle of impaact [radians]
#     rp (num): radius of the target body [m]

#     Returns
#     -------
#     v (num or array): ballistic speed [m s^-1]

#     """
#     tan_phi = np.tan(d / (2 * rp))
#     return np.sqrt(
#         (g * rp * tan_phi)
#         / ((np.sin(theta) * np.cos(theta)) + (np.cos(theta) ** 2 * tan_phi))
#     )


# def mps2kmph(v):
#     """
#     Return v in km/hr, given v in m/s

#     Parameters
#     ----------
#     v (num or array): velocity [m s^-1]

#     Returns
#     -------
#     v (num or array): velocity [km hr^-1]
#     """

#     return 3600 * v / 1000


# def thick2mass(thick, density):
#     """
#     Convert an ejecta blanket thickness to kg per meter squared, default
#     density of the ejecta blanket from Carrier et al. 1991.
#     Density should NOT be the bulk density of the Moon!

#     Parameters
#     ----------
#     thick (num or array): ejecta blanket thickness [m]
#     density (num): ejecta blanket density [kg m^-3]

#     Returns
#     -------
#     mass (num or array): mass of the ejecta blanket [kg]
#     """
#     return thick * density


# def mps2KE(speed, mass):
#     """
#     Return kinetic energy [J m^-2] given mass [kg], speed [m s^-1].

#     Parameters
#     ----------
#     speed (num or array): ballistic speed [m s^-1]
#     mass (num or array): mass of the ejecta blanket [kg]

#     Returns
#     -------
#     KE (num or array): Kinetic energy [J m^-2]
#     """
#     return 0.5 * mass * speed ** 2.0


# def ice_melted(
#     ke,
#     t_surf=cfg.coldtrap_max_temp,
#     cp=cfg.regolith_cp,
#     ice_rho=cfg.ice_density,
#     ice_frac=cfg.ice_frac,
#     heat_frac=cfg.heat_frac,
#     heat_ret=cfg.heat_retained,
#     lat_heat=cfg.ice_latent_heat,
#     t_melt=cfg.ice_melt_temp,
# ):
#     """
#     Return mass of ice [kg] melted by input kinetic energy.

#     Parameters
#     ----------
#     ke (num or array): kinetic energy of ejecta blanket [J m^-2]
#     t_surf (num): surface temperature [K]
#     cp (num): heat capacity for regolith [J kg^-1 K^-1]
#     ice_rho (num): ice density [kg m^-3]
#     ice_frac (num): fraction ice vs regolith (default: 5.6%; Colaprete 2010)
#     heat_frac (num): fraction KE used in heating vs mixing (default 50%)
#     heat_ret (num): fraction of heat retained (10-30%; Stopar 2018)
#     lat_heat (num): latent heat of ice [J/kg]
#     t_melt (num): melting point of ice [K]

#     Returns
#     -------
#     ice_mass (num or array): mass of ice melted due to ejecta [kg]
#     ice_depth (num or array): depth of ice melted due to ejecta [m]
#     """
#     heat = ice_frac * heat_ret * heat_frac * ke  # [J m^-2]
#     delta_t = t_melt - t_surf  # heat to go from t_surf to melting point
#     ice_mass = heat / (lat_heat + cp * delta_t)  # [kg]
#     # ice_depth = ice_mass / (ice_frac * ice_rho)  # [m]
#     return ice_mass


# def get_sublimation_rate(timestep=cfg.timestep, temp=cfg.coldtrap_max_temp):
#     """
#     Return ice lost due to sublimation at temp each timestep.

#     Compute surface residence time (Langmuir 1916, Kloos 2019), invert to
#     num H2O molecules lost in timestep, then convert num H2O to ice thickness.
#     """
#     vu0 = 2e12  # [s^-1] (vibrational frequency of water)
#     Ea = 0.456  # [eV] (activation energy)
#     kb = 8.6e-5  # [ev K^-1] (Boltzmann constant, in eV units)
#     tau = (1 / vu0) * np.exp(Ea / (kb * temp))  # [s], surface residence time
#     # TODO: covert tau to our units, get num H2O out/s, multiply by timestep
#     # convert num H2O to thickness of ice
#     return 0


