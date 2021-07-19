def final2transient(diams, g=G_MOON, rho_t=TARGET_DENSITY):
    """
    Return the transient crater diameter from final craters diams.    
    
    Parameters
    ----------
    diams (num or array): final crater diameters [m]
    g (num): gravitational force of the target body [m s^-2]
    rho_t (num): target density (kg m^-3)
    
    Returns
    -------
    transient_diams (num or array): transient crater diameters [m]
    """
    gama = 1.25
    eta = 0.13
    Dstar=(1.62*2700.*1.8e4)/(g*rho_t) #transition crater diameter 
    Dpr=(1.62*2700.*1.4e5)/(g*rho_t) #peak crater diameter
    if (Dfinal <= Dstar):
        transient_diams=Dfinal/gama
    else:
        transient_diams=(1./gama)*(Dfinal*Dstar**eta)**(1./(1.+eta))
    return transient_diams 
    
def diam2len_holsapple(diam, speeds, rho_i=IMPACTOR_DENSITY, rho_t=TARGET_DENSITY, 
                     g=G_MOON, v=IMPACT_SPEED, theta=IMPACT_ANGLE):
    """
    Return impactor length from input diam using Holsapple et al. (1993) method.
    
    Parameters
    ----------
    diam (num or array): transient crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)
    theta (num): impact angle (degrees)
    
    Returns
    -------
    impactor_length (num): impactor diameter (m)
    """
    i_diam_array = np.linspace(1, 100000, 1000) #1000.
    K1 = 0.132
    K2 = 0.26
    mu = 0.41
    nu = 0.33
    Y = 1e5 #dynes per cm^2 10000. #newton per m^2 = #
    pi2 = (grav*i_d) / velocity**2
    pi3 = Y / (t_rho*(velocity**2))
    m = (4/3)*np.pi*i_rho*(i_d**3)
    piV = (K1 * (pi2 * ((t_rho/i_rho)**((6*nu - 2 - mu)/(3*mu))) + K2 * pi3 * (t_rho/i_rho)**((6*nu - 2)/(3*mu)) ))**((-3*mu)/(2+mu))
    V = m/t_rho * piV
    t_diam_array = 2.*((3/(4*np.pi)) * (m/t_rho) * piV)**0.33
    
    Holsapple_int = interp(t_diam_array, i_diam_array, fill_value="extrapolate")
    impactor_length = Holsapple_int(diam)  
    return impactor_length
    
   
def diam2len_prieur(diam, speeds, rho_i=IMPACTOR_DENSITY, rho_t=TARGET_DENSITY, 
                     g=G_MOON, v=IMPACT_SPEED, theta=IMPACT_ANGLE):
    """
    Return impactor length from input diam using Prieur et al. (2017) method.
    
    Parameters
    ----------
    diam (num or array): transient crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)
    theta (num): impact angle (degrees)
    
    Returns
    -------
    impactor_length (num): impactor diameter (m)
    """
    i_diam_array = np.linspace(1, 100000, 1000) #1000.
    pi2 = 1.61*(g*i_diam_array)/(v**2)
    piD = 1.6*pi2**-0.22
    t_diam_array = piD / ((rho_t)/(rho_i*(4./3.)*np.pi*((i_diam_array/2.)**(3))))**0.33
    
    Prieur_int = interp(t_diam_array, i_diam_array, fill_value="extrapolate")
    impactor_length = Prieur_int(diam)  
    return impactor_length
    
def diam2len_collins(diam, speeds, rho_i=IMPACTOR_DENSITY, rho_t=TARGET_DENSITY, 
                     g=G_MOON, v=IMPACT_SPEED, theta=IMPACT_ANGLE):
    """
    Return impactor length from input diam using Collins et al. (2005) method.
    
    Parameters
    ----------
    diam (num or array): transient crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)
    theta (num): impact angle (degrees)
    
    Returns
    -------
    impactor_length (num): impactor diameter (m)
    """    
    impactor_length = (diam / (1.161*((rho_i/rho_t)**0.33)*(v**0.44)*(g**-0.22)*(np.sin(np.deg2rad(theta)))**0.33) )**(1/0.78)
    return impactor_length

def diam2len_johnson(diam, rho_i=IMPACTOR_DENSITY, rho_t=TARGET_DENSITY, 
                     g=G_MOON, v=IMPACT_SPEED, theta=IMPACT_ANGLE):
    """
    Return impactor length from input diam using Johnson et al. (2016) method.        
    
    Parameters
    ----------
    diam (num or array): transient crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)
    theta (num): impact angle (degrees)
    
    Returns
    -------
    impactor_length (num): impactor diameter (m)
    """
    Dstar=(1.62 * 2700 * 1.8e4) / (g * rho_t)
    denom = (1.52 * (rho_i / rho_t)**0.38 * v**0.5 * g**-0.25 * 
             Dstar**-0.13 * np.sin(theta)**0.38)
    impactor_length = (diam / denom)**(1 / 0.88)
    return impactor_length

def Potter(diam, rho_i=IMPACTOR_DENSITY, rho_t=TARGET_DENSITY, 
                     g=G_MOON, v=IMPACT_SPEED):
    """
    Return impactor length from input diam using Johnson et al. (2016) method.        
    
    Parameters
    ----------
    diam (num or array): transient crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)
   
    Returns
    -------
    impactor_length (num): impactor diameter (m)
    """
    i_diam_array = np.linspace(1, 100000, 1000) #1000.
    pi2 = 3.22*(g*(diam/2.0))/(v**2)
    piD=1.6*pi2**-0.22
    t_diam_array=piD *((rho_i*(4./3.)*np.pi*((diam/2.)**(3)))/(rho_t))**0.33

    Potter_int = interp(t_diam_array, i_diam_array, fill_value="extrapolate")
    impactor_length = Potter_int(diam)  
    return impactor_length