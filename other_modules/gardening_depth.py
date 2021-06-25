#Katelyn Frizzell 6/22/21
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Below are the lambda/probablility values, we can use or discard them"""
# d = {
#    'n': [1, 2, 3, 4, 6, 8, 10, 20, 30, 40, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6],
#    'L10': [0.105, 0.530, 1.102, 1.742, 3.150, 4.655, 6.221, 14.53, 23.33, 32.11, 87.42, 2.78e2, 9.596e2, 2.93e3, 9.872e3, 2.978e4, #9.959e4, 2.993e5, 9.9687e5],
#    'L50': [0.693, 1.678, 2.674, 3.672, 5.67, 7.67, 9.67, 19.67, 29.67, 39.67, 99.67, 2.997e2, 9.997e2, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6],
#    'L99': [4.605, 6.638, 8.406, 10.05, 13.11, 16, 18.87, 31.85, 44.19, 56.16, 1.247e2, 3.418e2, 1.075e3, 3.129e3, 1.023e4, 3.041e4, #1.007e5, 3.013e5, 1.002e6]
#}
#df = pd.DataFrame(d)

def overturn_depth(lam, u, v, t = 1e7): 
    """
    Impact Gardening module
    Goal: Take an input of crater size distribution and spit out overturn depth
    Equations from Costello et al. 2018 and 2020
    
    Notes: Secondaries have stronger impact on depth (1e5 secondary for every primary)

    Parameters
    ----------
    lam = probability [dimensionless], can plot all probabilities/events or pick one
    u = sfd scaling factor [dimensionless] (u*x^v)
    v = sfd slope/exponent [dimensionless] (u*x^v)
        ***For this module to describe physical reality v must be < -2
    t = timestep [years] (e.g. 1e7)
    c = proximity scaling parameter for overlapping craters [dimensionless?]
    h = depth fraction of crater overturned [dimensionless?] 
        (c,h from page 6 of Costello 2020 (c = 1/2*sqrt(1-h) = 0.41))
    
    Return
    ----------
    L = gardening depth [meters] 
    TODO: double check units (m?)
    """
    c = 0.41
    h = 1-(4*(c**2)) 
    p1 = ((v+2)/(v*u))
    p2 = (4*lam)/((c**2)*np.pi)
    B = (1/(v+2))
    A = abs(h*((p1*p2))**B)
    L = A*(t**(-B))
    return L

#Values Costello used for u, v to run tests
#1st entry is primaries, 2nd is secondaries, and 3rd is micrometeorites
u_test = [6.3e-11, 7.25e-9, 1.53e-12]
v_test = [-2.7, -4, -2.64]
#test lambda value until we choose one or more
lamb = (2.674)
Depth_m = [0,0,0]
for i in range(0,3):
    Depth_m[i] = overturn_depth(lamb, u_test[i],v_test[i])

print(Depth_m) #array of depths from primaries, secondaries, and micrometeorites