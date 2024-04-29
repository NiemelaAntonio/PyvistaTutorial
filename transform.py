
"""Module for computing coordinate transformations.
"""

import numpy as np
import shock_tracer.constants as constants

def spherical_coordinate_to_cartesian(crd):

    r = crd[0]; t = crd[1]; p = crd[2];

    x = r*np.sin(t)*np.cos(p)
    y = r*np.sin(t)*np.sin(p)
    z = r*np.cos(t)

    return [x,y,z]


def cartesian_coordinate_to_spherical(crd):

    x = crd[0]; y = crd[1]; z = crd[2];

    r = np.sqrt(x*x + y*y + z*z)
    t = np.arccos(z/r)
    p = np.arctan2(y, x)

    return [r,t,p]


def spherical_vector_to_cartesian(vec, crd):

    t  = crd[1]; p  = crd[2];
    Br = vec[0]; Bt = vec[1]; Bp = vec[2];

    Bx = Br*np.sin(t)*np.cos(p) + Bt*np.cos(t)*np.cos(p) - Bp*np.sin(p)
    By = Br*np.sin(t)*np.sin(p) + Bt*np.cos(t)*np.sin(p) + Bp*np.cos(p)
    Bz = Br*np.cos(t) - Bt*np.sin(t)

    return [Bx, By, Bz]


def cartesian_vector_to_spherical(vec, crd):

    sph_crd = cartesian_coordinate_to_spherical(crd)

    t  = sph_crd[1]; p = sph_crd[2];
    Bx = vec[0]; By = vec[1]; Bz = vec[2];

    Br =  Bx*np.sin(t)*np.cos(p) + By*np.sin(t)*np.sin(p) + Bz*np.cos(t)
    Bt =  Bx*np.cos(t)*np.cos(p) + By*np.cos(t)*np.sin(p) - Bz*np.sin(t)
    Bp = -Bx*np.sin(p) + By*np.cos(p)

    return [Br, Bt, Bp]

def kinetic_energy_to_momentum(KE,mass):

    return np.sqrt((KE/constants.c)**2 + 2 * mass * KE)
    
    
def momentum_to_kinetic_energy(p, mass):

    rest_energy = mass*constants.c*constants.c
    return np.sqrt((p*constants.c)**2 + rest_energy**2) - rest_energy


def momentum_to_speed(p, mass):

    return np.sqrt((p/mass)**2 / (1 + ( p/(constants.c * mass))**2))


