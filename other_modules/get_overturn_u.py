
import sys
sys.path.insert(0,'D:/2021-internship/code/code/mixing_model/')
from mixing_210628 import get_random_impactor_speeds
from numpy import diag_indices_from
from numpy.lib.function_base import _angle_dispatcher
from numpy.lib.index_tricks import fill_diagonal
import numpy as np



IMPACTOR_DENSITY = 2780#1300  # [kg m^-3], Cannon 2020 (costello uses 2700)
IMPACT_SPEED = 507#20e3  # [m/s] mean impact speed (Cannon 2020) (Costello's numbers are m/yr)
IMPACT_SD = 6e3  # [m/s] standard deviation impact speed (Cannon 2020)
ESCAPE_VEL = 2.38e3  # [m/s] lunar escape velocity
IMPACT_ANGLE = 45  # [deg]  average impact velocity
TARGET_DENSITY = 1500  # [kg m^-3] (Cannon 2020)
BULK_DENSITY = 2700  # [kg m^-3] simple to complex (Melosh)
TARGET_KR = 0.6  # [] Costello 2018, for lunar regolith
TARGET_K1 = 0.132  # [] Costello 2018, for lunar regolith
TARGET_K2 = 0.26  # [] Costello 2018, for lunar regolith
TARGET_MU = 0.41  # [] Costello 2018, for lunar regolith
TARGET_YIELD_STR = 0.01*1e6  # [Pa] Costello 2018, for lunar regolith
GRAV_MOON = 1.62  # [m s^-2], gravitational acceleration



def get_overturn_u(regime, m_type, rho_t=TARGET_DENSITY, rho_i=IMPACTOR_DENSITY, 
                kr=TARGET_KR, k1=TARGET_K1, k2=TARGET_K2, mu=TARGET_MU,
                y=TARGET_YIELD_STR, vf=IMPACT_SPEED, g=GRAV_MOON, theta_i = IMPACT_ANGLE):
    """
    Return size-frequecy factor u for overturn (eqn 13, Costello 2020).
    """
    alpha = k2 * (( (y/(rho_t*vf**2)) )**(0.5*(2+mu)))
    beta = (-3*mu)/(2+mu)
    di = 2 * kr
    gamma = (k1*np.pi*rho_i)/(6*rho_t)
    eps = (g/(2*vf**2))*(rho_i/rho_t)**(1/3)

    if m_type == 'primary':
        a = 6.3E-11 
        #a = 7.25E-14 #brown et al 2002
        b = -2.7 
    elif m_type == 'secondary':
        a = 7.25E-9
        b = -4
    elif m_type == 'micrometeorite':
        a = 1.53E-12
        b = -2.64
    else:
        print("primary, secondary, or micrometeorite?")

    if regime == 'strength':
        t1 = (np.sin(np.deg2rad(theta_i)))**(2/3)
        t2_denom = (di*(gamma*alpha**beta)**(1/3))
        u =  t1 * a * (1/t2_denom)**b 

    elif regime == 'gravity':
        t1 = (np.sin(np.deg2rad(theta_i)))**(1/3)
        t2_denom = (di*(gamma*eps**beta)**(1/3))
        exp = (3*b)/(3 + beta)
        u = t1 * a * (1/t2_denom)**exp
    else:
        print("choose a regime!")
    return u

print(get_overturn_u('strength', 'secondary'))