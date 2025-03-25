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


"""Coronal Mass Ejection launcher
"""

import datetime
import os
import string

import pandas as pd

# Import CME models
import EUHFORIA.cone_model
import EUHFORIA.sphere

class CMELauncher(object):
    def __init__(self, start_time=None, end_time=None):

        self.launch_window_start = start_time
        self.launch_window_end = end_time

        self.list_of_cmes = []

    def set_launch_window(self, start_time, end_time):

        self.launch_window_start = start_time
        self.launch_window_end = end_time

        if self.launch_window_start > self.launch_window_end:
            raise ValueError("Launch window start efter end of window")

    def spawn(self, solar_wind_boundary=None):
        """Creates CME instances based on list of CMEs
        """

        # Loop through selected CMEs and create instances
        for index, cme in self.df.iterrows():

            cme_type = cme["type"].lower()

            if cme_type == "cone":
                self.list_of_cmes.append(euhforia.cme.cone_model.ConeModel(*cme[1:8]))
            elif cme_type == "sphere":
                self.list_of_cmes.append(euhforia.cme.sphere.SphereModel(*cme[1:8]))
            elif cme_type == "lffspheromak":
                self.list_of_cmes.append(euhforia.cme.spheromak.LFFSpheromakModel(*cme[1:11]))
            elif cme_type == "gl":
                self.list_of_cmes.append(gl.GibsonLowModel(*cme[1:12]))
            elif cme_type == "fri3d":
                self.list_of_cmes.append(
                    euhforia.cme.fri3d_model.FRi3DModel(*cme[1:17], solar_wind_boundary=solar_wind_boundary)
                )
            else:
                raise ValueError("Unknown CME type: " + cme["type"])

    def read_from_list(self, file_name):
        """Reads CME parameters from given file and stores CMEs
        that are within the launch window in a pandas.DataFrame

        Args:
            file_name (str) : Name of file of CME list
        """

        #
        # Read CME list into pandas.DataFrame
        #
        max_num_columns = 20

        try:
            df = pd.read_csv(
                file_name, comment="#", sep=r"\s+", header=None, keep_default_na=True, names=range(max_num_columns)
            )
        except IOError:
            raise IOError("CME list not found: " + os.path.abspath(file_name))

        # Name the columns
        # The first two columns MUST be the type of the CME followed by the date&time
        # at which it enters the simulation
        df.columns = ["type"] + ["date"] + list(string.ascii_lowercase[: max_num_columns - 2])

        df["date"] = pd.to_datetime(df["date"])

        #
        # Select CMEs that enter within the launch window
        #
        mask = (df["date"] >= self.launch_window_start) & (df["date"] <= self.launch_window_end)

        # Store dataFrame
        self.df = df.loc[mask].copy()

        # Sort according to date
        self.df.sort_values("date", inplace=True)
