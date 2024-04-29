
"""Module that defines some  constants.
All units are in SI.
"""

import numpy as np
import scipy.constants

#
# Astronomical constants
#


# Solar radus, m
solar_radius		= 695800.0e3
R_sun				= solar_radius

# Astronomical unit, m
astronomical_unit	= scipy.constants.astronomical_unit
au					= astronomical_unit
AU					= astronomical_unit

# Carrington synodic rotation period, s
solar_synodic_rotation_period  = 27.2753*24.*60.*60.
solar_sidereal_rotation_period = 25.4*24*60*60

# Solar rotation rate, rad/s
solar_synodic_rotation_rate      = 2.0*np.pi/solar_synodic_rotation_period
solar_sidereal_rotation_rate     = 2.0*np.pi/solar_sidereal_rotation_period 

#charge of proton, C

e = scipy.constants.elementary_charge     
q = e

# Mass of proton, kg
proton_mass = scipy.constants.proton_mass

# Boltzmann constant, J/K
kB = scipy.constants.k

# Gravitational constant
G = scipy.constants.G

# Magnetic constant
mu0 = scipy.constants.mu_0

#speed of light in vacuum,  m/s
c = scipy.constants.speed_of_light         

# mass electron, kg

me = scipy.constants.electron_mass
electron_mass = me

# mass proton, kg 

mp = scipy.constants.proton_mass
proton_mass = mp
     
# atomic mass unit,kg

amu = scipy.constants.physical_constants["atomic mass constant"][0]

# Mass helium aplha particle, kg

mHe = 4.001506179127 * amu

#  Kilo electron volts [J]

keV = 1e3 * e

#  Mega electron volts [J]

MeV = 1e6 * e  

#  Giga electron volts s

GeV = 1e9 * e

