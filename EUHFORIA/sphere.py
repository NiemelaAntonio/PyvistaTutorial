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


"""Sphere Coronal Mass Ejection Model
"""

import datetime

import numpy as np
import pandas as pd

import constants as constants


class SphereModel(object):
    """Sphere CME model.

    The sphere model is the simplest possible model of a CME: a spherical
    uniformly filled hydrodynamical structure. The CME is assumed to propagate
    as a rigid body in the direction of the source location with a
    given speed. The density and temperature inside the CME are homogenuous.

    Attributes:
        params.lat        :  Latitude of the eruption, HEEQ rad
        params.lon        :  Longitude of the eruption, HEEQ rad
        params.speed      :  Speed of the CME, m/s
        params.radius     :  Radius of the sphere, m
        params.start_time :  Time at which the outer edge of the sphere is at r=0.1 AU
    """

    def __init__(self, start_time, lat, lon, radius, speed, mass_density, temperature):
        """Initializes the model.

        Args:
            start_time (datetime) : Time of CME at r = 0.1 AU
            lat (float)    : Latitude of eruption center, deg HEEQ
            lon (float)    : Longitude of eruption center, deg HEEQ
            radius (float) : Radius of the sphere, in solar radii
            speed (float)  : Speed of the CME, km/s
            mass_density (float) : Constant mass density, kg/m^3
            temperature (float)  : Constant temperature, K
        """

        class Parameters(object):
            pass

        self.params = Parameters()

        self.params.lat = float(lat) * np.pi/180.0
        self.params.clt = -self.params.lat + 0.5*np.pi
        self.params.lon = float(lon) * np.pi/180.0

        self.params.start_time = start_time.to_pydatetime() if isinstance(start_time, pd.Timestamp) else start_time
        self.params.radius = float(radius)*constants.solar_radius
        self.params.speed = float(speed)*1e3
        self.params.mass_density = float(mass_density)
        self.params.temperature = float(temperature)

        #
        # Compute derived parameters
        #

        # Time at which CME has fully passed the boundary
        self.params.end_time = self.params.start_time + datetime.timedelta(
            seconds=2.0*self.params.radius/self.params.speed
        )

        # Cartesian velocity vector
        self.velocity = self.params.speed*np.array(
            (
                np.sin(self.params.clt)*np.cos(self.params.lon),
                np.sin(self.params.clt)*np.sin(self.params.lon),
                np.cos(self.params.clt),
            )
        )

        # Distance of center of sphere from r=0.1 AU at t = start_time
        distance_from_boundary = 0.1*constants.astronomical_unit - self.params.radius

        # Cartesian position of center of sphere at t = start_time
        self.pos_of_center_at_start_time = distance_from_boundary * np.array(
            (
                np.sin(self.params.clt)*np.cos(self.params.lon),
                np.sin(self.params.clt)*np.sin(self.params.lon),
                np.cos(self.params.clt),
            )
        )

    def center(self, t):
        """Computes the Cartesian coordinates of the position of the center of
        the sphere at the given time

        Args:
            t (datetime) : Current date and time
        Returns:
            numpy array containing (x,y,z) coordinates
        """

        time_from_start = (t - self.params.start_time).total_seconds()

        return self.pos_of_center_at_start_time + self.velocity*time_from_start

    def mask(self, clt, lon, t):
        """Computes a mask of the grid points on the boundary at r = 0.1 AU
        where the CME is inserted.
        """

        # Construct grid of coordinates
        cltmesh, lonmesh = np.meshgrid(clt, lon, indexing="ij")

        # Get Cartesian coordinates of center of sphere
        center = self.center(t)

        # Compute squared distance to center of sphere

        r = 0.1*constants.astronomical_unit

        dx = r*np.sin(cltmesh)*np.cos(lonmesh) - center[0]
        dy = r*np.sin(cltmesh)*np.sin(lonmesh) - center[1]
        dz = r*np.cos(cltmesh) - center[2]

        dsqr = dx**2 + dy**2 + dz**2

        # If distance < radius => inside sphere
        mask = dsqr <= (self.params.radius**2)

        return mask

    def indices_to_insert(self, clt, lon, t):
        """Indices of grid points on the boundary at r = 0.1 AU where the
        CME is inserted.
        """

        idx = ()

        if t >= self.params.start_time and t <= self.params.end_time:

            mask = self.mask(clt, lon, t)
            shape = (len(clt), len(lon))
            idx = np.ravel_multi_index(np.where(mask > 0.0), shape)

        return idx

    def get_coordinates_on_sphere(self, clt, lon, idx):
        """Get (colat, lon) coordinates on the given flattened indices
        """

        cltmesh, lonmesh = np.meshgrid(clt, lon, indexing="ij")

        clts = cltmesh.flatten()[idx]  # Use ravel instead?
        lons = lonmesh.flatten()[idx]

        return clts, lons

    def mass_density(self, clt, lon, t):
        """Computes CME mass density
        """
        return self.indices_to_insert(clt, lon, t), self.params.mass_density

    def temperature(self, clt, lon, t):
        """Computes CME temperature
        """
        return self.indices_to_insert(clt, lon, t), self.params.temperature

    def vr(self, clt, lon, t):
        """Computes CME radial speed
        """

        vr = 0.0
        idx = self.indices_to_insert(clt, lon, t)

        # Compute on the indices where the CME is to be inserted
        if len(idx) > 0:

            clts, lons = self.get_coordinates_on_sphere(clt, lon, idx)

            vr = (
                self.velocity[0]*np.sin(clts)*np.cos(lons)
                + self.velocity[1]*np.sin(clts)*np.sin(lons)
                + self.velocity[2]*np.cos(clts)
            )

        return idx, vr

    def vt(self, clt, lon, t):
        """Computes CME colat speed
        """

        vt = 0.0
        idx = self.indices_to_insert(clt, lon, t)

        # Compute on the indices where the CME is to be inserted
        if len(idx) > 0:

            clts, lons = self.get_coordinates_on_sphere(clt, lon, idx)

            vt = (
                self.velocity[0]*np.cos(clts)*np.cos(lons)
                + self.velocity[1]*np.cos(clts)*np.sin(lons)
                - self.velocity[2]*np.sin(clts)
            )

        return idx, vt

    def vp(self, clt, lon, t):
        """Computes CME lon speed
        """

        vp = 0.0
        idx = self.indices_to_insert(clt, lon, t)

        # Compute on the indices where the CME is to be inserted
        if len(idx) > 0:

            clts, lons = self.get_coordinates_on_sphere(clt, lon, idx)

            vp = -self.velocity[0]*np.sin(lons) + self.velocity[1]*np.cos(lons)

        return idx, vp

    def Br(self, clt, lon, t):
        """Computes CME radial magnetic field
        """
        return (), 0.0

    def Bt(self, clt, lon, t):
        """Computes CME colat magnetic field
        """
        return (), 0.0

    def Bp(self, clt, lon, t):
        """Computes CME lon magnetic field
        """
        return (), 0.0

    def __str__(self):

        message = "Sphere CME at {} with {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
            self.params.start_time,
            self.params.speed/1e3,
            self.params.lat*180.0/np.pi,
            self.params.lon*180.0/np.pi,
            self.params.radius/constants.solar_radius,
            self.params.mass_density/1e-18,
            self.params.temperature/1e6,
        )

        return message
