# This file is part of EUHFORIA.
#
# Copyright 2016, 2017 Jens Pomoell
#
# EUHFORIA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# EUHFORIA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EUHFORIA. If not, see <http://www.gnu.org/licenses/>.


"""Cone Coronal Mass Ejection Model
"""

import datetime

import numpy as np
import pandas as pd

import constants as constants


class ConeModel(object):
    """Cone CME model.

    The cone model is a cone-shaped hydrodynamical CME model. The CME is
    assumed to propagate radially outwards from the source location with a
    given speed. The density and temperature inside the CME are homogenuous.

    Attributes:
        params.lat        :  Latitude of the eruption, HEEQ rad
        params.lon        :  Longitued of the eruption, HEEQ rad
        params.speed      :  Speed of the CME, m/s
        params.half_width :  Half width of the cone, rad
        params.start_time :  Time at which the cone intersects the r=0.1 AU sphere
    """

    def __init__(self, start_time, lat, lon, half_width, speed, mass_density, temperature):
        class Parameters(object):
            pass

        self.params = Parameters()

        self.params.lat = float(lat) * np.pi/180.0
        self.params.clt = -self.params.lat + 0.5*np.pi
        self.params.lon = float(lon) * np.pi/180.0

        self.params.start_time = start_time.to_pydatetime() if isinstance(start_time, pd.Timestamp) else start_time
        self.params.half_width = float(half_width)*np.pi/180.0
        self.params.speed = float(speed)*1e3
        self.params.mass_density = float(mass_density)
        self.params.temperature = float(temperature)

        # Compute derived parameters

        # Time at which half of the CME has passed the boundary
        # NOTE: The boundary is assumed to be 0.1 AU here!
        self.params.t_half = 0.1*constants.astronomical_unit*np.tan(self.params.half_width)/self.params.speed

        # Time at which CME has been fully inserted
        self.params.end_time = self.params.start_time + datetime.timedelta(seconds=2.0*self.params.t_half)

    def opening_angle(self, t):
        """Compute the opening angle of the cone CME as a function of time

        Args:
            t (datetime) : Current date and time
        """

        angle = 0.0

        if t >= self.params.start_time and t <= self.params.end_time:

            time_from_start = (t - self.params.start_time).total_seconds()

            # angle = self.params.half_width*np.sqrt(np.sin(0.5*np.pi*time_from_start/self.params.t_half))
            angle = self.params.half_width * np.sin(0.5*np.pi*time_from_start/self.params.t_half)

        return angle

    def mask(self, clt, lon, t):
        """Determines indices of grid points on the boundary sphere where the
        CME is inserted
        """

        idx = ()

        opening_angle = self.opening_angle(t)

        if opening_angle > 0.0:

            # Construct grid of coordinates
            cltmesh, lonmesh = np.meshgrid(clt, lon, indexing="ij")

            # Compute array indices where to insert values
            mask = ((cltmesh - self.params.clt)**2 + (lonmesh - self.params.lon)**2) < (opening_angle**2)
            idx = np.where(mask > 0.0)

            # Transform to flat index
            # TODO: some better way??
            idx = np.ravel_multi_index(idx, cltmesh.shape)

        return idx

    def mass_density(self, clt, lon, t):
        """Computes CME mass density
        """
        return self.mask(clt, lon, t), self.params.mass_density

    def temperature(self, clt, lon, t):
        """Computes CME temperature
        """
        return self.mask(clt, lon, t), self.params.temperature

    def vr(self, clt, lon, t):
        """Computes CME radial speed
        """
        return self.mask(clt, lon, t), self.params.speed

    def vt(self, clt, lon, t):
        """Computes CME colatitudinal speed
        """
        return (), 0.0

    def vp(self, clt, lon, t):
        """Computes CME longitudinal speed
        """
        return (), 0.0

    def Br(self, clt, lon, t):
        """Computes CME radial magnetic field
        """
        return (), 0.0

    def Bt(self, clt, lon, t):
        """Computes CME co-lat magnetic field
        """
        return (), 0.0

    def Bp(self, clt, lon, t):
        """Computes CME lon magnetic field
        """
        return (), 0.0

    def __str__(self):

        message = "Cone CME at {} with {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
            self.params.start_time,
            self.params.speed/1e3,
            self.params.lat*180.0/np.pi,
            self.params.lon*180.0/np.pi,
            self.params.half_width*180.0/np.pi,
            self.params.mass_density/1e-18,
            self.params.temperature/1e6,
        )

        return message
