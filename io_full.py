"""
Light-weight version of IO module of EUHFORIA. Can also handle hdf5 format used by PARADISE. 
"""

import numpy as np
import pyvista as pv
from datetime import datetime, timedelta
import pathlib,h5py,warnings 
import grid as grid
import constants as constants
import transform as transform
import pandas as pd
from perturbed_solar_wind import rotate_W0
import fd_4th_order_uniform as fd_kernel
import vector_calculus as vector_calculus

class DataContainer():

    def __init__(self, corotating=True):
        self.data   = {}
        self.grid   = grid.Grid()
        self.rotate = corotating
 
    def add_variable(self, data, name,**kwargs):

        self.data[name] = data
        setattr(self, name, self.data[name])

    def convert_to_pv_mesh(self,variables,delete=True,background=None,**kwargs):

        lims = kwargs.get('lims', [self.grid.center_coords.r[0],   self.grid.center_coords.r[-1],
                                   self.grid.center_coords.clt[0], self.grid.center_coords.clt[-1],
                                   self.grid.center_coords.lon[0], self.grid.center_coords.lon[-1]]
                                   )
        self.pv_idx_lims = self.grid.get_index_voi(lims)

        lon_ghosts = self.grid.num_ghost_cells[2]
        
        if lon_ghosts  != 0: 
            dgi = self.pv_idx_lims[4]    - lon_ghosts
            dgf = self.grid.num_cells[2] - self.pv_idx_lims[5] - lon_ghosts

            if dgi >= 0 and dgf >= 0: 
                # no ghost-cells at either side of the sub-axis
                lon_ghosts = 0
            elif (dgi <0 and dgf >=0) :
                # remove ghost-cells when only present in lower side of the axis 
                self.pv_idx_lims[4] = lon_ghosts
                lon_ghosts = 0
            elif  (dgi >= 0 and dgf <0) :
                # remove ghost-cells when only present in upper side of the axis 
                self.pv_idx_lims[5] = self.grid.num_cells[2]-lon_ghosts 
                lon_ghosts = 0              
            else:
                # make sure the same number of ghost cells is present in both sides of the axis  
                lon_ghosts          = - max(dgi,dgf)           
                self.pv_idx_lims[4] = self.grid.num_ghost_cells[2] - lon_ghosts 
                self.pv_idx_lims[5] = self.grid.num_cells[2] - self.pv_idx_lims[4]


        r,t,p = np.meshgrid(self.grid.edge_coords.r  [self.pv_idx_lims[0]:self.pv_idx_lims[1]+1],
                            self.grid.edge_coords.clt[self.pv_idx_lims[2]:self.pv_idx_lims[3]+1], 
                            self.grid.edge_coords.lon[self.pv_idx_lims[4]:self.pv_idx_lims[5]+1],
                            indexing="ij" )
   
        [x,y,z] = transform.spherical_coordinate_to_cartesian([r,t,p])

        self.pvgrid = pv.StructuredGrid(x, y, z)

        for var in variables:
            if var in self.data.keys():
                self.pvgrid.cell_data[var] = self.slice3D(self.data[var],self.pv_idx_lims).T.ravel("C")
                if delete:
                    delattr(self, var)
                    del self.data[var]

        #
        # Also store the background wind on the pv mesh if requested
        #

        if background is not None:

            dt =(kwargs.get('datetime',self.datetime)-background.datetime).total_seconds() if background.rotate else 0
            dt =(kwargs.get('datetime',self.datetime)-background.datetime).total_seconds()
            print(dt) ##ADDED THIS

            if 'n' in background.data:
                self.pvgrid.cell_data['n'] = rotate_W0(self.slice3D(background.data['n'],self.pv_idx_lims),
                                             dt,self.grid.dlon[0], lon_ghosts ).T.ravel("C")
            if 'P' in background.data:
                self.pvgrid.cell_data['P'] = rotate_W0(self.slice3D(background.data['P'],self.pv_idx_lims),
                                             dt,self.grid.dlon[0],lon_ghosts ).T.ravel("C")


            r,t,p = np.meshgrid(self.grid.center_coords.r  [self.pv_idx_lims[0]:self.pv_idx_lims[1]],
                                self.grid.center_coords.clt[self.pv_idx_lims[2]:self.pv_idx_lims[3]], 
                                self.grid.center_coords.lon[self.pv_idx_lims[4]:self.pv_idx_lims[5]],
                                indexing="ij" )

            if 'Br' in background.data:
                self.pvgrid.cell_data['B_pol'] = np.sign(rotate_W0(self.slice3D(background.data['Br'],self.pv_idx_lims),
                                                      dt,self.grid.dlon[0],
                                                      lon_ghosts)).T.ravel("C")


                B = transform.spherical_vector_to_cartesian(
                            [rotate_W0(self.slice3D(background.Br,self.pv_idx_lims),dt,  self.grid.dlon[0],lon_ghosts ),
                             rotate_W0(self.slice3D(background.Bclt,self.pv_idx_lims),dt,self.grid.dlon[0],lon_ghosts ),
                             rotate_W0(self.slice3D(background.Blon,self.pv_idx_lims),dt,self.grid.dlon[0],lon_ghosts )], 
                            [r,t,p])
                self.pvgrid.cell_data['B'] = np.column_stack (
                (B[0].T.ravel("C"),B[1].T.ravel("C"),B[2].T.ravel("C")))


            if 'vr' in background.data:
                V = transform.spherical_vector_to_cartesian(
                            [rotate_W0(self.slice3D(background.vr,self.pv_idx_lims),dt,  self.grid.dlon[0],lon_ghosts ),
                             rotate_W0(self.slice3D(background.vclt,self.pv_idx_lims),dt,self.grid.dlon[0],lon_ghosts ),
                             rotate_W0(self.slice3D(background.vlon,self.pv_idx_lims),dt,self.grid.dlon[0],lon_ghosts )], 
                            [r,t,p])
                V_sph = [rotate_W0(self.slice3D(background.vr,self.pv_idx_lims),dt,  self.grid.dlon[0],lon_ghosts ),## I ADDED THIS LINE
                             rotate_W0(self.slice3D(background.vclt,self.pv_idx_lims),dt,self.grid.dlon[0],lon_ghosts ),## I ADDED THIS LINE
                             rotate_W0(self.slice3D(background.vlon,self.pv_idx_lims),dt,self.grid.dlon[0],lon_ghosts )]## I ADDED THIS LINE

                self.pvgrid.cell_data['V_bg_cart'] = np.column_stack (
                    (V[0].T.ravel("C"),V[1].T.ravel("C"),V[2].T.ravel("C")))
                self.pvgrid.cell_data['V_bg_sph'] = np.column_stack (                                                   ## I ADDED THIS LINE
                    (V_sph[0].T.ravel("C"),V_sph[1].T.ravel("C"),V_sph[2].T.ravel("C")))                             ## I ADDED THIS LINE

    def slice3D(self,data,lims):
        return np.copy(data[lims[0]:lims[1],lims[2]:lims[3],lims[4]:lims[5]])

def load_heliospheric_data(fname, variables_and_scale_factors,**kwargs):


    # check if file exists
    fname = pathlib.Path(fname)
    if not fname.is_file():
        raise IOError("Solar wind datafile not found")

    # check if file is hdf5 file (PARADISE format) or 
    # npz file (EUHFORIA format)
    if h5py.is_hdf5(str(fname)):
        ftype = "hdf5"
    elif fname.suffix == ".npz":
        ftype = "npz"
    else:
        raise IOError("Solar wind datafile is not a hdf5 or npz file")

    # datacontainer in which all the data will be stored
    data = DataContainer(corotating=kwargs.get('corotating', True))

    if ftype == "npz":
        with np.load(str(fname),allow_pickle=True,encoding="latin1") as f:
            
            if "grid" in kwargs.keys():
                data.grid = kwargs["grid"]
            else:
                data.grid.initialize(r  =f["r"]/kwargs.get("r_unit",1),
                                     clt=f["clt"], lon=f["lon"],
                                     axis_type="edges",
                                     num_ghost_points=(2,2,2),
                                     r_unit=kwargs.get("r_unit",1))

            data.datetime = f["datetime"].item()
            for var, sf in variables_and_scale_factors.items():

                if var == "Br" and kwargs.get('B2CellCenters', False):
                    data.add_variable(0.5*(f[var][1:,:,:]+f[var][:-1,:,:])* sf, var)
                elif var == "Bclt" and kwargs.get('B2CellCenters', False):
                    data.add_variable(0.5*(f[var][:,1:,:]+f[var][:,:-1,:])* sf, var)
                elif var == "Blon" and kwargs.get('B2CellCenters', False):
                    data.add_variable(0.5*(f[var][:,:,1:]+f[var][:,:,:-1])* sf, var)
                elif var == "speed":
                    data.add_variable(vector_calculus.norm([f['vr'],f['vclt'], f['vlon']]), var)
                elif var == "divV":
                    vectorCalc  = vector_calculus.VectorDifferentialOperators(data.grid, fd_kernel)
                    divV        = vectorCalc.div( np.array([f['vr'],f['vclt'],f['vlon']]) )
                    data.add_variable(divV*sf,var)
                elif var == 'entropy_density':
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="invalid value encountered in power")
                        data.add_variable(f['P']*( 0.5 * f['n'] * 1e6 * constants.mp )**(1.0 - 1.5) * sf,var)
                elif var in f.keys():
                    data.add_variable(f[var] * sf, var)
                else:
                    raise KeyError(var + " not found!")

    elif ftype == "hdf5":
        with h5py.File(str(fname), "r") as f:

            if "grid" in kwargs.keys():
                data.grid = kwargs["grid"]
            else:
                num_ghosts = [0,0,0]
                if "num_ghosts_r" in f.attrs:   num_ghosts[0] = f.attrs["num_ghosts_r"]
                if "num_ghosts_clt" in f.attrs: num_ghosts[1] = f.attrs["num_ghosts_clt"]
                if "num_ghosts_lon" in f.attrs: num_ghosts[2] = f.attrs["num_ghosts_lon"]

                data.grid.initialize(r=f["r"][:]/kwargs.get("r_unit",1),
                                    clt=f["clt"][:],lon=f["lon"][:],
                                    axis_type="centers",
                                    num_ghost_points=num_ghosts,
                                    r_unit=kwargs.get("r_unit",1))

            data.datetime = datetime.strptime(f.attrs["datetime"].replace(':', '-'),
                "%Y-%m-%dT%H-%M-%S")

            for var, sf in variables_and_scale_factors.items():
                
                if var == "U" or var.lower() == "v":
                    data.add_variable(f["U"][0] * sf, 'vr')
                    data.add_variable(f["U"][1] * sf, 'vclt')
                    data.add_variable(f["U"][2] * sf, 'vlon')
                elif var == "B":
                    data.add_variable(f["B"][0] * sf, 'Br')
                    data.add_variable(f["B"][1] * sf, 'Bclt')
                    data.add_variable(f["B"][2] * sf, 'Blon')
                elif var == "speed":
                    data.add_variable(vector_calculus.norm(f["U"][:]) * sf, var)
                elif var == "divV" or var == "divU":
                    vectorCalc  = vector_calculus.VectorDifferentialOperators(data.grid, fd_kernel)
                    data.add_variable(vectorCalc.div(f["U"][:])*sf,var)
                elif var in f.keys():
                    data.add_variable(f[var][:] * sf,var)
                else:
                    raise KeyError(var + " not found!")

    return data


def get_cme_injection_paramters(cme_file):

    try:
        cme_df = pd.read_csv(cme_file, comment="#", sep='\s+', header=None, 
                         keep_default_na=True)
    except IOError:
        raise IOError("CME list not found: " + os.path.abspath(cme_file))

    if cme_df.shape[0]>1:
        raise RuntimeError(f'More than one CME present in {cme_file}')

    cme_inj_pars =\
        { "name": cme_df.iloc[0,0],
          "injection time":datetime.strptime(cme_df.iloc[0,1],"%Y-%m-%dT%H:%M:%S"),
          "clt": (90 - cme_df.iloc[0,2])*np.pi/180. ,
          "lon": cme_df.iloc[0,3]*np.pi/180.,
          "halfwidth":cme_df.iloc[0,4]*np.pi/180. ,
          "injection speed":cme_df.iloc[0,5] * 1e3 / constants.au}

    cme_inj_pars['radius']    = 0.1*np.sin(cme_inj_pars["halfwidth"])

    if "Spheromak" in cme_df.iloc[0,0]:
        cme_inj_pars['radius']    = cme_df.iloc[0,4] * constants.R_sun/constants.au
        cme_inj_pars['halfwidth'] = np.arcsin(cme_inj_pars['radius']/0.1)


    cme_inj_pars['end time'] = cme_inj_pars["injection time"] \
    + timedelta(seconds=2 * cme_inj_pars['radius'] / cme_inj_pars['injection speed'])
     
    cme_inj_pars['half time'] = cme_inj_pars["injection time"] \
    + timedelta(seconds=cme_inj_pars['radius'] / cme_inj_pars['injection speed'])

    cme_inj_pars['quarter time'] = cme_inj_pars["injection time"] \
    + timedelta(seconds=0.5 * cme_inj_pars['radius'] / cme_inj_pars['injection speed'])

    return cme_inj_pars

def Clipping_close_CME(limits_to_plot):
    # Generate all combinations of limits
    r_values = [limits_to_plot[0], limits_to_plot[1]]
    clt_values = [limits_to_plot[2], limits_to_plot[3]]
    lon_values = [limits_to_plot[4], limits_to_plot[5]]

    # Create Cartesian coordinates for all combinations
    cart_list = [
        np.array(transform.spherical_coordinate_to_cartesian([r, clt, lon]))
        for r in r_values
        for clt in clt_values
        for lon in lon_values
    ]

    # Compute min and max for each axis
    x_min, x_max = min(arr[0] for arr in cart_list), max(arr[0] for arr in cart_list)
    y_min, y_max = min(arr[1] for arr in cart_list), max(arr[1] for arr in cart_list)
    z_min, z_max = min(arr[2] for arr in cart_list), max(arr[2] for arr in cart_list)

    return [x_min, x_max, y_min, y_max, z_min, z_max]



