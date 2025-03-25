# This file is part of EUHFORIA.
#
# Copyright 2016, 2017 Jens Pomoell
# Copyright 2018 Jens Pomoell, Alexey Isavnin, Christine Verbeke
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


"""EUHFORIA heliospheric MHD model.

This is the main module implementing the time-dependent, 3D heliosphere MHD
model of EUHFORIA.
"""

import configparser
import datetime
import errno
import os
import re
import warnings

import numpy as np
import scipy

#import coco
import constants as constants
#import coco.io.npy
#import coco.io.point
#import coco.sim
#import euhforia
#import euhforia.core.io
import EUHFORIA.cme_launcher as cme_launcher
#import euhforia.heliosphere.mhd.electric_field_solver
#import euhforia.heliosphere.mhd.solar_wind_boundary_data as sw
#import euhforia.orbit


class InnerHeliosphereSimulation(coco.sim.IdealMagnetohydrodynamicsSimulation):
    """EUHFORIA inner heliosphere MHD model.
    """

    def __init__(self, config, **kwargs):

        coco.sim.IdealMagnetohydrodynamicsSimulation.__init__(self, **kwargs)

        #
        # Parse configuration data
        #
        self.config = configparser.ConfigParser()

        # Make sure the configuration file exists
        if os.path.isfile(config):
            self.config.read(config)
        else:
            raise IOError("Configuration file not found: " + os.path.abspath(config))

        #
        # Solar wind boundary data
        #
        self.solar_wind_boundary = sw.SingleSynopticSolarWindBoundaryModel()

        #
        # Coronal mass ejection models
        #
        self.CMEs = cme_launcher.CMELauncher()

        #
        # Virtual spacecraft to track and extract data at
        #
        self.virtual_spacecraft = []

        #
        # Set default model parameters
        #

        # Mean molecular mass, kg
        self.params.mean_molecular_mass = 0.5*coco.constants.proton_mass

        # Adiabatic index
        self.params.adiabatic_index = 1.5

        # Heliocentric radius of the inner radial boundary of the computational
        # domain
        self.params.r0 = 0.1*coco.constants.astronomical_unit

    def set_grid(self):
        """Construct the computational grid
        """

        #
        # Select spherical geometry
        #
        self.grid.geometry = "spherical"

        #
        # Construct grid coordinates from config
        #

        # Compute the number of cells in each direction

        num_radial_cells = self.config.getint("Grid", "num_radial")
        angular_resolution = self.config.getfloat("Grid", "angular_resolution")

        num_clt_cells = int(120.0/angular_resolution)
        num_lon_cells = int(360.0/angular_resolution)

        AU = coco.constants.astronomical_unit

        #
        # Uniform radial grid
        #
        r = coco.CoordinateAxis(name="r", unit=AU)
        r.construct_uniform(0.1, self.config.getfloat("Grid.radial", "outer_edge", fallback=2.0), num_radial_cells + 1)

        #
        # Uniform co-latitude grid
        # Ensure the number of cells is odd so that clt = pi/2 is a cell-center
        # location. Note that this means that the given angular resolution
        # is not exactly used
        #
        num_clt_cells = num_clt_cells + 1 if num_clt_cells % 2 == 0 else num_clt_cells
        colat = coco.CoordinateAxis(name="clt")
        colat.construct_uniform(30.0, 150.0, num_clt_cells + 1)
        colat.coordinates *= np.pi/180.0

        #
        # Uniform longitudinal grid
        # The longitude is shifted by dlon/2 so that lon = 0 is located on
        # a cell-center
        #
        lon = coco.CoordinateAxis(name="lon")
        lon.construct_uniform(-180.0 + 0.5*angular_resolution, 180.0 + 0.5*angular_resolution, num_lon_cells + 1)
        lon.coordinates *= np.pi/180.0

        #
        # Set the axes
        #
        self.grid.set_coordinate_axes(r, colat, lon)

    def set_solver_parameters(self):
        """Set parameters intrinsic to the physical model
        """
        self.solver.kernel.adiabatic_index = self.params.adiabatic_index
        self.solver.kernel.gravity = True
        self.solver.courant_number = 0.3

    def initialize_solar_wind_boundary_data(self):

        # Load solar wind data
        # Perform smoothing of the boundary data proportional to the
        # angular grid size
        angular_resolution = self.config.getfloat("Grid", "angular_resolution")

        smoothing = self.config.getfloat(
            "Misc", "solar_wind_boundary_smoothing", fallback=max(0.0, 0.4 + 0.37*(angular_resolution - 1.0))
        )

        self.solar_wind_boundary.load(self.config.get("Data", "solar_wind"), smooth=smoothing)

        # Check that the boundaries agree
        # if not np.isclose(self.solar_wind_boundary.radius, self.params.r0):
        #    raise ValueError("Solar wind boundary data radius different from " \
        #                     "simulation inner boundary radius")

    def initialize_electric_field_solver(self):

        self.boundary_electric_field_solver = euhforia.heliosphere.mhd.electric_field_solver.ElectricFieldSolver()

        self.boundary_electric_field_solver.set_grid(
            self.global_grid.indomain_edge_coords.clt, self.global_grid.indomain_edge_coords.lon, self.params.r0
        )

    def initialize_CMEs(self):

        # Specify time frame in which CMEs are inserted
        self.CMEs.set_launch_window(self.cme_insertion_start_time, self.forecast_end_time)

        # Read CME list
        if self.config.has_option("Data", "cmes"):

            self.CMEs.read_from_list(self.config.get("Data", "cmes"))

            # Create CME objects
            self.CMEs.spawn(solar_wind_boundary=self.solar_wind_boundary)

    def update_simulation_time(self):
        """Update the date and time in the simulation.
        """
        self.datetime = self.forecast_start_time + datetime.timedelta(seconds=self.t)

    def set_simulation_duration(self, skip_cme_insertion=False):
        """Set the time-related parameters of the simulation.

        There are two variables related to time that the simulation keeps
        track of:
            1. self.t        : Time of the simulation in seconds counting from
                               the forecast time. Any time before the forecast
                               time is negative.
            2. self.datetime : datetime object storing the current actual date
                               and time in the simulation.
        """

        #
        # Get simulation duration from configuration
        #
        one_hour = 60.0*60.0
        one_day = 24.0*one_hour

        relaxation_duration = self.config.getfloat("Duration", "relaxation", fallback=10.0) * one_day

        cme_insertion_duration = self.config.getfloat("Duration", "cme_insertion", fallback=5.0) * one_day

        if skip_cme_insertion:
            cme_insertion_duration = 0.0

        forecast_duration = self.config.getfloat("Duration", "forecast", fallback=5.0) * one_day

        #
        # Set starting time for simulation. Time t=0 defines the start of the
        # forecast, and is preceded by the relaxation and CME insertion.
        #
        self.t = -(relaxation_duration + cme_insertion_duration)

        #
        # Store date and times of simulation phases
        #
        self.forecast_start_time = self.solar_wind_boundary.time_of_chart

        self.forecast_end_time = self.forecast_start_time + datetime.timedelta(seconds=forecast_duration)

        self.cme_insertion_start_time = self.forecast_start_time - datetime.timedelta(seconds=cme_insertion_duration)

        # Update time in the simulation
        self.update_simulation_time()

        # Set arguments for run
        self.run_kwargs = {"t_stop": forecast_duration}

    def set_initial_condition(self):
        """Set initial state of simulation.
        """

        if not self.solar_wind_boundary.initialized:
            raise ValueError("Solar wind boundary data has not been initalized")

        #
        # Restart
        #
        if self.config.has_option("Data", "restart_from"):

            restart_from_file = self.config.get("Data", "restart_from")

            #
            # Load restart data
            #
            restart = euhforia.core.io.load_heliospheric_data(restart_from_file)

            #
            # Check that the grid is conformant with that in the input
            #
            for dim in (0, 1, 2):
                if not np.allclose(self.global_grid.edge_coords[dim], restart.grid.edge_coords[dim]):
                    raise ValueError("Coordinates of restart data not consistent with requested grid")

            #
            # Set local data
            #

            # Indices where domain overlap starts
            i = abs(restart.grid.center_coords[0] - self.grid.center_coords[0][0]).argmin()
            j = abs(restart.grid.center_coords[1] - self.grid.center_coords[1][0]).argmin()
            k = abs(restart.grid.center_coords[2] - self.grid.center_coords[2][0]).argmin()

            selection = (
                slice(i, i + self.grid.num_cells[0]),
                slice(j, j + self.grid.num_cells[1]),
                slice(k, k + self.grid.num_cells[2]),
            )

            rho = 1e6*self.params.mean_molecular_mass*restart.data["n"][selection]

            # Mass density
            self.solver.kernel.mass_density()[:, :, :] = rho

            # Pressure
            self.solver.kernel.pressure()[:, :, :] = restart.data["P"][selection]

            # Momentum density
            for dim in (0, 1, 2):
                self.solver.kernel.momentum_density(dim)[:, :, :] = (
                    1e3*rho*restart.data[("vr", "vclt", "vlon")[dim]][selection]
                )

            # Magnetic field
            for dim in (0, 1, 2):

                i = abs(restart.grid.face_center_coords(dim)[0] - self.grid.face_center_coords(dim)[0][0]).argmin()
                j = abs(restart.grid.face_center_coords(dim)[1] - self.grid.face_center_coords(dim)[1][0]).argmin()
                k = abs(restart.grid.face_center_coords(dim)[2] - self.grid.face_center_coords(dim)[2][0]).argmin()

                selection = (
                    slice(i, i + self.grid.num_cells[0] + (1 if dim == 0 else 0)),
                    slice(j, j + self.grid.num_cells[1] + (1 if dim == 1 else 0)),
                    slice(k, k + self.grid.num_cells[2] + (1 if dim == 2 else 0)),
                )

                self.solver.kernel.magnetic_field(dim)[:, :, :] = (
                    1e-9*restart.data[("Br", "Bclt", "Blon")[dim]][selection]
                )

            # Set simulation time to time of restart
            self.t = (restart.datetime - self.forecast_start_time).total_seconds()

            #
            # Update simulation state to time of restart
            #
            self.update_simulation_time()

            return

        #
        # Initialize cell-centered quantities
        #

        # Mass density at r=r0
        rho0 = self.solar_wind_boundary.number_density(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.t)
        rho0 *= self.params.mean_molecular_mass

        # Radial velocity at r=r0
        vr0 = self.solar_wind_boundary.vr(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.t)

        # Temperature at r=r0
        T0 = self.solar_wind_boundary.temperature(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.t)

        # Pressure at r=r0
        P0 = rho0*constants.kB*T0/self.params.mean_molecular_mass

        # Set cell centered quantities as a function of r
        for i, r in enumerate(self.grid.axis.r.centers):

            inv_square_fallof = (self.params.r0/r)**2

            # Mass density so that radial mass flux is constant along ray
            #
            #   rho(r) = rho(r_0) (r_0/r)^2
            #
            self.solver.kernel.mass_density()[i, :, :] = rho0*inv_square_fallof

            # Thermal pressure set so that entropy is constant along ray
            #
            # P \rho^{-\gamma} = const ==> P(r) = P(r_0) [(r_0/r)^2]^\gamma
            #
            self.solver.kernel.pressure()[i, :, :] = P0*(inv_square_fallof)**(self.params.adiabatic_index)

            # Radial velocity constant along ray
            #
            self.solver.kernel.momentum_density(0)[i, :, :] = self.solver.kernel.mass_density()[i, :, :]*vr0

        #
        # Initialize purely radial magnetic field
        #

        # Radial magnetic field at r=r0
        Br0 = self.solar_wind_boundary.Br(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.t)

        for i, r in enumerate(self.grid.axis.r.edges):
            self.solver.kernel.magnetic_field(0)[i, :, :] = Br0*(self.params.r0 / r)**2

    def add_output_events(self):
        """Add all output events
        """

        # Add VTK output
        if self.config.has_section("Output.VTK"):
            self.add_vtk_output_event()

        # Add NPY output
        if self.config.has_section("Output.NPY"):
            self.add_npy_output_event()

        # Add output events for the virtual spacecraft
        if self.config.has_section("VirtualSpacecraft"):
            self.add_virtual_spacecraft_output()

    def get_base_of_output_file_name(self):
        """Returns base part of output file names
        """

        # Path + base name of output file
        output_directory = self.config.get("Output", "directory", fallback="./")
        base_of_file_name = output_directory + self.config.get("Output", "base_name")

        # Create the directory if it does not exist
        # Avoids race condition: http://stackoverflow.com/a/5032238/720077
        try:
            os.makedirs(output_directory)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        return base_of_file_name

    def get_output_start_time(self, output_type, default="forecast"):
        """Returns start time of given output event
        """

        # Start of output
        start = self.config.get(output_type, "start", fallback=default)

        start_of_output = 0.0
        if start == "insertion":
            start_of_output = (self.cme_insertion_start_time - self.forecast_start_time).total_seconds()
        elif start == "relaxation":
            start_of_output = self.t
        elif start == "forecast":
            start_of_output = 0.0
        elif start == "backgroundwind":
            start_of_output = (self.forecast_start_time - self.forecast_end_time).total_seconds()

        return start_of_output

    def add_vtk_output_event(self):

        if self.grid.rank == 0:
            warnings.warn("Direct VTK output is deprecated. Use at own peril!")

        import coco.io.vtk

        #
        # Define data to output
        #
        def vtk_output_data():

            rho = self.solver.kernel.mass_density()

            # convert to nanoTesla
            B1 = 1e9*self.solver.kernel.magnetic_field(0)
            B2 = 1e9*self.solver.kernel.magnetic_field(1)
            B3 = 1e9*self.solver.kernel.magnetic_field(2)
            B = np.array((B1, B2, B3))

            return {
                "n": 1e-6*rho/self.params.mean_molecular_mass,
                "vr": 1e-3*self.solver.kernel.momentum_density(0)/rho,
                "vclt": 1e-3*self.solver.kernel.momentum_density(1)/rho,
                "vlon": 1e-3*self.solver.kernel.momentum_density(2)/rho,
                "P": self.solver.kernel.pressure(),
                "B": B,
            }

        #
        # Add output event
        #
        one_hour = 60.0*60.0

        # Output frequency
        interval = self.config.getfloat("Output.VTK", "interval") * one_hour

        vtk_output_event = coco.io.vtk.VTKWriteEvent(
            self,
            vtk_output_data,
            interval=interval,
            base_name=self.get_base_of_output_file_name(),
            centering="nodal",
            start=self.get_output_start_time("Output.VTK"),
        )

        vtk_output_event.output_file_name = (
            lambda: self.get_base_of_output_file_name() + "_" + self.datetime.strftime("%Y-%m-%dT%H-%M-%S")
        )

        self.events.add(vtk_output_event)

    def add_npy_output_event(self):
        """NumPy array output
        """

        def output_data():

            rho = self.solver.kernel.mass_density()

            return {
                "n": 1e-6*rho/self.params.mean_molecular_mass,
                "vr": 1e-3*self.solver.kernel.momentum_density(0)/rho,
                "vclt": 1e-3*self.solver.kernel.momentum_density(1)/rho,
                "vlon": 1e-3*self.solver.kernel.momentum_density(2)/rho,
                "P": self.solver.kernel.pressure(),
                "Br": 1e9*self.solver.kernel.magnetic_field(0),
                "Bclt": 1e9*self.solver.kernel.magnetic_field(1),
                "Blon": 1e9*self.solver.kernel.magnetic_field(2),
                "datetime": self.datetime,
            }

        # Output frequency
        one_hour = 60.0*60.0
        interval = self.config.getfloat("Output.NPY", "interval") * one_hour

        event = coco.io.npy.NPYDataOutputEvent(
            self,
            output_data,
            interval=interval,
            base_name=self.get_base_of_output_file_name(),
            start=self.get_output_start_time("Output.NPY"),
        )

        event.output_file_name = (
            lambda: self.get_base_of_output_file_name() + "_" + self.datetime.strftime("%Y-%m-%dT%H-%M-%S")
        )

        self.events.add(event)

    def add_virtual_spacecraft_output(self):
        """Add output events for the virtual spacecraft
        """

        #
        # Define interpolators
        #
        def cell_centered_interpolation(q, cell_idx, p):
            """Interpolates cell-centered quantity q to point p using simple
               dimension-by-dimension linear interpolation.
            """
            axis = self.grid.axis

            i, j, k = cell_idx

            cell_center = (
                0.5*(axis[0].coordinates[i + 1] + axis[0].coordinates[i]),
                0.5*(axis[1].coordinates[j + 1] + axis[1].coordinates[j]),
                0.5*(axis[2].coordinates[k + 1] + axis[2].coordinates[k]),
            )

            dqdx1 = 0.5*(q[i + 1, j, k] - q[i - 1, j, k]) / (axis[0].coordinates[i + 1] - axis[0].coordinates[i])
            dqdx2 = 0.5*(q[i, j + 1, k] - q[i, j - 1, k]) / (axis[1].coordinates[j + 1] - axis[1].coordinates[j])
            dqdx3 = 0.5*(q[i, j, k + 1] - q[i, j, k - 1]) / (axis[2].coordinates[k + 1] - axis[2].coordinates[k])

            return (
                q[i, j, k]
                + dqdx1 * (p[0] - cell_center[0])
                + dqdx2 * (p[1] - cell_center[1])
                + dqdx3 * (p[2] - cell_center[2])
            )

        def magnetic_field_interpolator(cell_idx, p):
            """Interpolates staggered magnetic field to point p
            """
            axis = self.grid.axis

            i, j, k = cell_idx

            dx = (
                axis[0].coordinates[i + 1] - axis[0].coordinates[i],
                axis[1].coordinates[j + 1] - axis[1].coordinates[j],
                axis[2].coordinates[k + 1] - axis[2].coordinates[k],
            )

            B1 = self.solver.kernel.magnetic_field(0)
            B2 = self.solver.kernel.magnetic_field(1)
            B3 = self.solver.kernel.magnetic_field(2)

            return (
                B1[i, j, k] + (B1[i + 1, j, k] - B1[i, j, k])*(p[0] - axis[0].coordinates[i])/dx[0],
                B2[i, j, k] + (B2[i, j + 1, k] - B2[i, j, k])*(p[1] - axis[1].coordinates[j])/dx[1],
                B3[i, j, k] + (B3[i, j, k + 1] - B3[i, j, k])*(p[2] - axis[2].coordinates[k])/dx[2],
            )

        #
        # Define data to output at each virtual spacecraft
        #
        def output_point_data(p, idx):

            rho = cell_centered_interpolation(self.solver.kernel.mass_density(), idx, p)
            n = 1e-6*rho/self.params.mean_molecular_mass

            mr = cell_centered_interpolation(self.solver.kernel.momentum_density(0), idx, p)
            mclt = cell_centered_interpolation(self.solver.kernel.momentum_density(1), idx, p)
            mlon = cell_centered_interpolation(self.solver.kernel.momentum_density(2), idx, p)

            vr = 1e-3*mr/rho
            vclt = 1e-3*mclt/rho
            vlon = 1e-3*mlon/rho

            P = cell_centered_interpolation(self.solver.kernel.pressure(), idx, p)

            Br, Bclt, Blon = magnetic_field_interpolator(idx, p)

            return (
                self.datetime.strftime("%Y-%m-%dT%H:%M:%S"),
                p[0]/constants.astronomical_unit,
                p[1],
                p[2],
                n,
                P,
                vr,
                vclt,
                vlon,
                Br*1e9,
                Bclt*1e9,
                Blon*1e9,
            )

        # Header of output file
        header = (
            "date r[AU] clt[rad] lon[rad] n[1/cm^3] P[Pa] vr[km/s] vclt[km/s] vlon[km/s] Br[nT] Bclt[nT] Blon[nT]"
            + "\n"
        )

        #
        # Instantiate planets/spacecraft/etc based on inputs in the config
        #
        for section in self.config.sections():
            if "VirtualSpacecraft." in section:

                # Name of virtual spacecraft
                vs_name = re.sub("VirtualSpacecraft.", "", section)

                # Get object that it is relative to.
                # If not given, assumed to be one of the standard objects
                relative_to = self.config.get("VirtualSpacecraft." + vs_name, "relative_to", fallback=vs_name)

                # Instantiate the base object
                obj_relative_to = getattr(euhforia.orbit, relative_to)()

                # Get shift from object that it is relative to.
                shift = self.config.get("VirtualSpacecraft." + vs_name, "shift", fallback="{0.0, 0.0, 0.0}")

                # Parse shift
                if shift is not None:
                    shift = np.fromstring(shift.strip("{}"), sep=",")
                    shift[0] *= constants.astronomical_unit
                    shift[1] *= np.pi/180.0
                    shift[2] *= np.pi/180.0

                # Instantiate the virtual spacecraft
                vs = euhforia.orbit.orbit.RelativeOrbit(name=vs_name, obj_relative_to=obj_relative_to, shift=shift)

                # Add vs to list of virtual spacecraft if its orbit is known
                # for the entire duration of the simulation
                if vs.exists(self.datetime, self.forecast_end_time):
                    self.virtual_spacecraft.append(vs)

        #
        # Add output event for each vs
        #
        for vs in self.virtual_spacecraft:

            # Initial position
            pos_r, pos_lat, pos_lon = vs.position(self.datetime)

            # Output frequency
            one_hour = 60.0*60.0
            interval = self.config.getfloat("VirtualSpacecraft." + vs.name, "interval", fallback=1.0/6.0) * one_hour

            # Add event
            event = euhforia.core.io.VirtualSpacecraftPointOutput(
                self.get_base_of_output_file_name() + "_" + vs.name + ".dsv",
                (pos_r, 0.5*np.pi - pos_lat, pos_lon),
                self,
                output_point_data,
                vs=vs,
                output_format="{} " + (3 + 8) * "{:.5e} " + "\n",
                header=header,
                interval=interval,
                start=self.get_output_start_time("VirtualSpacecraft." + vs.name, default="insertion"),
            )

            # Add the event to list of events
            self.events.add(event)

    def set_inner_radial_ghost_cells(self):
        """Sets the values of the ghost cells at the inner radial boundary
        """

        # Shorthand to kernel
        mhd = self.solver.kernel

        #
        # Compute solar wind boundary data
        #

        # Solar wind number density at r=r0
        number_density = self.solar_wind_boundary.number_density(
            self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.t
        )
        # Convert to mass density
        mass_density = self.params.mean_molecular_mass * number_density

        # Solar wind speed components at r=r0
        # The solar wind speed is assumed to be purely radial
        vr = self.solar_wind_boundary.vr(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.t)

        vt = np.zeros(mhd.momentum_density(1)[1, :, :].shape)

        vp = np.zeros(mhd.momentum_density(2)[1, :, :].shape)

        # Solar wind temperature at r=r0
        T = self.solar_wind_boundary.temperature(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.t)

        # Solar wind magnetic field components at r=r0
        Br = self.solar_wind_boundary.Br(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.t)

        Bt = np.zeros(mhd.magnetic_field(1)[1, :, :].shape)

        Bp = self.solar_wind_boundary.Bp(self.grid.axis.clt.centers, self.grid.axis.lon.edges, self.t)

        #
        # Replace solar wind boundary data with CMEs
        #

        for cme in self.CMEs.list_of_cmes:

            mass_density.put(*cme.mass_density(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.datetime))

            T.put(*cme.temperature(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.datetime))

            vr.put(*cme.vr(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.datetime))
            vt.put(*cme.vt(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.datetime))

            vp.put(*cme.vp(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.datetime))

            Br.put(*cme.Br(self.grid.axis.clt.centers, self.grid.axis.lon.centers, self.datetime))

            Bt.put(*cme.Bt(self.grid.axis.clt.edges, self.grid.axis.lon.centers, self.datetime))

            Bp.put(*cme.Bp(self.grid.axis.clt.centers, self.grid.axis.lon.edges, self.datetime))

        # Thermal pressure at r=r0
        P = mass_density*coco.constants.kB*T/self.params.mean_molecular_mass

        #
        # Set ghost cells so that the interface values are enforced linearly
        #

        # Mass density
        mhd.mass_density()[1, :, :] = 2.0*mass_density - mhd.mass_density()[2, :, :]
        mhd.mass_density()[0, :, :] = 4.0*mass_density - 3.0*mhd.mass_density()[2, :, :]

        # Momentum density
        #
        # The flux is computed from the reconstructed interface states. The
        # reconstructur on the other hand uses the velocity. So if we want to
        # enforce the velocity, then we should set
        #
        #     vr[1] = 2*vr[interface] - vr[2]
        #           = 2*vr[interface] - mr[2]/rho[2]
        #  => mr[1] = 2*vr[interface]*rho[1] - mr[2]*rho[1]/rho[2]

        mhd.momentum_density(0)[1, :, :] \
            = 2.0*vr*mhd.mass_density()[1, :, :] \
            - mhd.momentum_density(0)[2, :, :]*(mhd.mass_density()[1, :, :]/mhd.mass_density()[2, :, :])

        mhd.momentum_density(0)[0, :, :] \
            = 4.0*vr*mhd.mass_density()[0, :, :] \
            - 3.0*mhd.momentum_density(0)[2, :, :]*(mhd.mass_density()[0, :, :]/mhd.mass_density()[2, :, :])

        mhd.momentum_density(1)[1, :, :] \
            = 2.0*vt*mhd.mass_density()[1, :, :] \
            - mhd.momentum_density(1)[2, :, :]*(mhd.mass_density()[1, :, :]/mhd.mass_density()[2, :, :])

        mhd.momentum_density(1)[0, :, :] \
            = 4.0*vt*mhd.mass_density()[0, :, :] \
            - 3.0*mhd.momentum_density(1)[2, :, :]*(mhd.mass_density()[0, :, :]/mhd.mass_density()[2, :, :])

        mhd.momentum_density(2)[1, :, :] \
            = 2.0*vp*mhd.mass_density()[1, :, :] \
            - mhd.momentum_density(2)[2, :, :]*(mhd.mass_density()[1, :, :]/mhd.mass_density()[2, :, :])

        mhd.momentum_density(2)[0, :, :] \
            = 4.0*vp*mhd.mass_density()[0, :, :] \
            - 3.0*mhd.momentum_density(2)[2, :, :]*(mhd.mass_density()[0, :, :]/mhd.mass_density()[2, :, :])

        # Pressure
        mhd.pressure()[1, :, :] = 2.0*P - mhd.pressure()[2, :, :]
        mhd.pressure()[0, :, :] = 4.0*P - 3.0*mhd.pressure()[2, :, :]

        # Magnetic field

        # The radial component is to some degree arbitrary
        mhd.magnetic_field(0)[0, :, :] = Br*(self.grid.axis.r.edges[2]/self.grid.axis.r.edges[0])**2
        mhd.magnetic_field(0)[1, :, :] = Br*(self.grid.axis.r.edges[2]/self.grid.axis.r.edges[1])**2

        mhd.magnetic_field(1)[1, :, :] = 2.0*Bt - mhd.magnetic_field(1)[2, :, :]
        mhd.magnetic_field(1)[0, :, :] = 4.0*Bt - 3.0*mhd.magnetic_field(1)[2, :, :]

        mhd.magnetic_field(2)[1, :, :] = 2.0*Bp - mhd.magnetic_field(2)[2, :, :]
        mhd.magnetic_field(2)[0, :, :] = 4.0*Bp - 3.0*mhd.magnetic_field(2)[2, :, :]

    def set_inner_radial_boundary_electric_field(self):
        """Specify the horizontal electric field components at the r = Rb inner
        boundary.

        NOTE: this function is currently under developement
        """

        # Shorthand to kernel
        mhd = self.solver.kernel

        if self.dt == 0.0:
            return

        #
        # Coordinates at which to compute dBrdt
        #
        # clt = self.global_grid.axis.clt.centers[1:-1]
        # lon = self.global_grid.axis.lon.centers[2:-2]
        clt = self.global_grid.axis.clt.centers
        lon = self.global_grid.axis.lon.centers

        #
        # Compute Br(t) and Br(t+dt) for solar wind
        #

        # TODO: Optimization: store next sw time step so that it is not needed
        # to be computed again

        # Solar wind speed components at r=r0
        # The solar wind speed is assumed to be purely radial
        vr = self.solar_wind_boundary.vr(clt, lon, self.t)
        vt = np.zeros(vr.shape)
        vp = np.zeros(vr.shape)

        # Solar wind magnetic field components at r=r0
        Br = self.solar_wind_boundary.Br(clt, lon, self.t)
        Bt = np.zeros(Br.shape)
        Bp = self.solar_wind_boundary.Bp(clt, lon, self.t)

        #
        # Replace solar wind boundary data with CMEs
        #
        for cme in self.CMEs.list_of_cmes:

            vr.put(*cme.vr(clt, lon, self.datetime))
            vt.put(*cme.vt(clt, lon, self.datetime))
            vp.put(*cme.vp(clt, lon, self.datetime))

            Br.put(*cme.Br(clt, lon, self.datetime))
            Bt.put(*cme.Bt(clt, lon, self.datetime))
            Bp.put(*cme.Bp(clt, lon, self.datetime))

        # SW Br at time = t
        # Br      = self.solar_wind_boundary.Br(clt, lon, self.t)

        # SW Br at time = t + dt
        # Br_next = self.solar_wind_boundary.Br(clt, lon, self.t + self.dt)

        #
        # Replace Br(t) and Br(t+dt) due to contributions from CME magnetic fields
        #
        for cme in self.CMEs.list_of_cmes:
            pass
            # Replace Br(t) pixels due to CMEs
            # Br.put(*cme.Br(clt, lon, self.datetime))

            # Replace Br(t+dt) pixels due to CMEs
            # Br_next.put(*cme.Br(clt, lon, self.datetime + datetime.timedelta(seconds=self.dt)))

            # # Previous implementation left here for now. Is the 4 pixel
            # # cap really necessary??
            # idx, Br_cme = cme.Br(clt, lon, self.datetime)
            #
            # # Insert CME only when/if it is larger than 4 pixels
            # if len(idx) > 4:
            #
            #     # Replace Br(t) of the solar wind with that of the CME
            #     Br.put(idx, Br_cme)
            #
            #     # Replace Br(t+dt) of the solar wind with that of the CME.
            #     # NOTE: This is done only for pixels that contain a CME
            #     # at time t which are listed in idx, and not all the pixels
            #     # for which there is a contribution at time t + dt.
            #     Br_next.put(idx, cme.Br_at_indices(clt, lon, self.datetime + datetime.timedelta(seconds=self.dt), idx))

        # dBrdt = (Br_next-Br)/self.dt

        #
        # Compute horizontal electric field from dBr/dt
        #
        # Eclt, Elon \
        #    = self.boundary_electric_field_solver.compute_electric_field(dBrdt)

        #
        # Place in grid on in-domain edges
        #
        # TODO: Optimization: global to local index mapping can be determined
        # only once and then stored
        # NOTE: The interpolator functionality is here used only out of
        # convenience. In fact, as Eclt and Elon are computed precisely on the
        # locations where they are needed, the interpolation just acts to pick
        # out the values of the electric field components at the correct
        # grid points. So, no interpolation is actually performed.
        # TODO: Check that this is actually the case.

        # Eclt = self.solar_wind_boundary.Bp(self.global_grid.indomain_center_coords.clt, self.global_grid.indomain_edge_coords.lon, self.t)*self.solar_wind_boundary.vr(self.global_grid.indomain_center_coords.clt, self.global_grid.indomain_edge_coords.lon, self.t)
        # Elon = np.zeros(Elon.shape)

        # E = - v x B
        Eclt = vr*Bp - vp*Br
        Elon = -(vr*Bt - vt*Br)

        #
        # Co-latitudinal electric field component
        #
        # Eclt_interp \
        #    = scipy.interpolate.RegularGridInterpolator((self.global_grid.indomain_center_coords.clt,
        #                                                 self.global_grid.indomain_edge_coords.lon),
        #                                                 Eclt)

        Eclt_interp = scipy.interpolate.RegularGridInterpolator(
            (self.global_grid.axis.clt.centers, self.global_grid.axis.lon.centers), Eclt
        )

        colatmesh, lonmesh = np.meshgrid(
            self.grid.indomain_center_coords.clt, self.grid.indomain_edge_coords.lon, indexing="ij"
        )

        crds = zip(colatmesh.flatten(), lonmesh.flatten())

        mhd.electric_field(1)[2, 2:-2, 2:-2] = np.reshape(Eclt_interp(list(crds)), colatmesh.shape)

        #
        # Longitudinal electric field component
        #
        # Elon_interp \
        #    = scipy.interpolate.RegularGridInterpolator((self.global_grid.indomain_edge_coords.clt,
        #                                                 self.global_grid.indomain_center_coords.lon),
        #                                                 Elon)

        Elon_interp = scipy.interpolate.RegularGridInterpolator(
            (self.global_grid.axis.clt.centers, self.global_grid.axis.lon.centers), Elon
        )

        colatmesh, lonmesh = np.meshgrid(
            self.grid.indomain_edge_coords.clt, self.grid.indomain_center_coords.lon, indexing="ij"
        )

        crds = zip(colatmesh.flatten(), lonmesh.flatten())

        mhd.electric_field(2)[2, 2:-2, 2:-2] = np.reshape(
            Elon_interp(list(crds)), mhd.electric_field(2)[2, 2:-2, 2:-2].shape
        )

    def set_outer_radial_ghost_cells(self):
        """Set ghost cells at outer radial boundary.
        """

        # Shorthand to kernel
        mhd = self.solver.kernel

        # Radial coordinates of grid cell centers
        rc = self.grid.axis.r.centers

        # Radial coordinates of grid edges
        re = self.grid.axis.r.edges

        # Compute speed at last in-domain radial cells
        #
        rho = mhd.mass_density()[-3, :, :]

        vr = mhd.momentum_density(0)[-3, :, :]/rho
        vt = mhd.momentum_density(1)[-3, :, :]/rho
        vp = mhd.momentum_density(2)[-3, :, :]/rho

        #
        # Extrapolate mass density by assuming that r^2 rho is constant. This
        # is appropriate for a radial flow of constant speed. The constant is
        # computed at the last in-domain cell:
        #
        #  r^2 rho = const = r^2 rho at last indomain cell
        #
        mhd.mass_density()[-2, :, :] = mhd.mass_density()[-3, :, :]*(rc[-3]/rc[-2])**2
        mhd.mass_density()[-1, :, :] = mhd.mass_density()[-3, :, :]*(rc[-3]/rc[-1])**2

        #
        # Extrapolate momentum density by assuming all components of the
        # velocity to be constant
        #

        # vr = const = vr at last in-domain radial cell
        mhd.momentum_density(0)[-2, :, :] = mhd.mass_density()[-2, :, :]*vr
        mhd.momentum_density(0)[-1, :, :] = mhd.mass_density()[-1, :, :]*vr

        # vt = constant = vt at last in-domain radial cell
        mhd.momentum_density(1)[-2, :, :] = mhd.mass_density()[-2, :, :]*vt
        mhd.momentum_density(1)[-1, :, :] = mhd.mass_density()[-1, :, :]*vt

        # vp = constant = vp at last in-domain radial cell
        mhd.momentum_density(2)[-2, :, :] = mhd.mass_density()[-2, :, :]*vp
        mhd.momentum_density(2)[-1, :, :] = mhd.mass_density()[-1, :, :]*vp

        # Extrapolate E ??

        # Extrapolate radial component of magnetic field using
        # r^2 Br = constant
        mhd.magnetic_field(0)[-2, :, :] = mhd.magnetic_field(0)[-3, :, :]*(re[-3]/re[-2])**2
        mhd.magnetic_field(0)[-1, :, :] = mhd.magnetic_field(0)[-3, :, :]*(re[-3]/re[-1])**2

        # Bt = constant
        mhd.magnetic_field(1)[-2, :, :] = mhd.magnetic_field(1)[-3, :, :]
        mhd.magnetic_field(1)[-1, :, :] = mhd.magnetic_field(1)[-3, :, :]

        # Bp = constant
        mhd.magnetic_field(2)[-2, :, :] = mhd.magnetic_field(2)[-3, :, :]
        mhd.magnetic_field(2)[-1, :, :] = mhd.magnetic_field(2)[-3, :, :]

        # Extrapolate pressure by assuming T = const
        #
        # P = (k/m) rho T ==> T = const =  P / rho
        mhd.pressure()[-2, :, :] = (mhd.pressure()[-3, :, :]*mhd.mass_density()[-2, :, :]/mhd.mass_density()[-3, :, :])
        mhd.pressure()[-1, :, :] = (mhd.pressure()[-3, :, :]*mhd.mass_density()[-1, :, :]/mhd.mass_density()[-3, :, :])

    def set_lower_latitudinal_ghost_cells(self):

        mhd = self.solver.kernel

        for q in (
            mhd.mass_density(),
            mhd.pressure(),
            mhd.momentum_density(0),
            mhd.momentum_density(1),
            mhd.momentum_density(2),
            # mhd.magnetic_field(0),
            mhd.magnetic_field(2),
        ):

            q[2:-2, 1, :] = q[2:-2, 2, :]
            q[2:-2, 0, :] = q[2:-2, 3, :]

        mhd.magnetic_field(0)[3::, 1, :] = 2.0*mhd.magnetic_field(0)[3::, 2, :] - mhd.magnetic_field(0)[3::, 3, :]
        mhd.magnetic_field(0)[3::, 0, :] = (3.0*mhd.magnetic_field(0)[3::, 2, :] - 2.0*mhd.magnetic_field(0)[3::, 3, :])

        mhd.magnetic_field(1)[:, 1, :] = mhd.magnetic_field(1)[:, 3, :]
        mhd.magnetic_field(1)[:, 0, :] = mhd.magnetic_field(1)[:, 4, :]

    def set_upper_latitudinal_ghost_cells(self):

        mhd = self.solver.kernel

        for q in (mhd.mass_density(),
                  mhd.pressure(),
                  mhd.momentum_density(0),
                  mhd.momentum_density(1),
                  mhd.momentum_density(2),
                  # mhd.magnetic_field(0),
                  mhd.magnetic_field(2),
                  ):

            q[2:-2, -2, :] = q[2:-2, -3, :]
            q[2:-2, -1, :] = q[2:-2, -4, :]

        mhd.magnetic_field(0)[3::, -2, :] = 2.0*mhd.magnetic_field(0)[3::, -3, :] - mhd.magnetic_field(0)[3::, -4, :]
        mhd.magnetic_field(0)[3::, -1, :] \
            = (3.0*mhd.magnetic_field(0)[3::, -3, :] - 2.0*mhd.magnetic_field(0)[3::, -4, :])

        mhd.magnetic_field(1)[:, -2, :] = mhd.magnetic_field(1)[:, -4, :]
        mhd.magnetic_field(1)[:, -1, :] = mhd.magnetic_field(1)[:, -5, :]

    def update_metadata(self):
        self.update_simulation_time()

    def set_boundary_conditions(self):

        self.boundary.r.lower.ghost_cells = self.set_inner_radial_ghost_cells
        self.boundary.r.lower.flux = self.set_inner_radial_boundary_electric_field

        self.boundary.r.upper.ghost_cells = self.set_outer_radial_ghost_cells

        self.boundary.clt.lower.ghost_cells = self.set_lower_latitudinal_ghost_cells
        self.boundary.clt.upper.ghost_cells = self.set_upper_latitudinal_ghost_cells

        self.boundary.lon.lower = "periodic"
        self.boundary.lon.upper = "periodic"

    def print_info_to_screen_at_startup(self):

        if self.grid.rank == 0:

            # Compute run duration rounded to nearest whole second
            run_duration = self.forecast_end_time - self.datetime
            run_duration = datetime.timedelta(seconds=np.round(run_duration.total_seconds()))

            print("EUHFORIA " + euhforia.__version__)
            print("----------------------------------------------------")
            print("Forecast starts at:      ", self.forecast_start_time.strftime("%Y-%m-%d %H:%M:%S"))
            print("Forecast ends at:        ", self.forecast_end_time.strftime("%Y-%m-%d %H:%M:%S"))
            print("CME insertion starts at: ", self.cme_insertion_start_time.strftime("%Y-%m-%d %H:%M:%S"))
            print("Run starts at:           ", self.datetime.strftime("%Y-%m-%d %H:%M:%S"))
            print("Run duration:            ", run_duration)
            print("----------------------------------------------------")

            # Print CME information
            for cme in self.CMEs.list_of_cmes:
                print(cme)
