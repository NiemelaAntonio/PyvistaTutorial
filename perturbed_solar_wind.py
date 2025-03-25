
import numpy as np
import constants as constants
import fd_4th_order_uniform as fd_kernel
import vector_calculus as vector_calculus
import grid as grid

def get_W_rel(data, background, variables=["speed"], abs_value=True):

    dlon = background.grid.center_coords.lon[3] - background.grid.center_coords.lon[2]
    dt   = (data.datetime - background.datetime).total_seconds()

 #   assert dt>0, "dt is smaller than zero."

    for var in variables:

        W0     = rotate_W0(np.copy(background.data[var]),dt,dlon,background.grid.num_ghost_cells[2])
        W_rel  = np.where(data.data[var] == W0, 0, (data.data[var]-W0)/np.abs(W0))
        fill_lon_ghost_cells(W_rel, data.grid.num_ghost_cells[2])
        if abs_value: W_rel = np.abs(W_rel)
        var_name = var+"_rel"
        data.add_variable(W_rel,var_name)


def rotate_W0(data, dt, lon_res, num_lon_ghosts):

    if dt == 0: return data
    
    imin = num_lon_ghosts
    imax = data.shape[2] - num_lon_ghosts

    data_rot = data[:,:,imin:imax]
    dlon_rot = dt * constants.solar_synodic_rotation_rate

    if dlon_rot / lon_res > 1:
        data_rot =  np.roll(data[:,:,imin:imax], int(dlon_rot / lon_res), axis = 2)
    
    a =  dlon_rot / lon_res - int(dlon_rot / lon_res)

    data_rot[:,:,1:] =   (a * data_rot[:,:,:-1] + (1-a) * data_rot[:,:,1:])
    data_rot[:,:, 0] =   (a * data_rot[:,:,-1]  + (1-a) * data_rot[:,:,0])

    if num_lon_ghosts == 0:
        return data_rot
    else:
        data_rot_out = np.zeros_like(data)
        data_rot_out[:,:,imin:imax] = data_rot
        fill_lon_ghost_cells(data_rot_out,num_lon_ghosts)
        return data_rot_out

def fill_lon_ghost_cells(data,num_lon_ghosts):

    for i in range(-num_lon_ghosts,num_lon_ghosts):
        data[:,:, i]  = data[:,:,int(-np.sign(i+1e-13)*2*num_lon_ghosts + i)]

def empty_ghost_cells(data,num_ghost_cells):

    for i in range(-num_ghost_cells[0],num_ghost_cells[0]):
        data[i,:,:]  = 0
    for i in range(-num_ghost_cells[1],num_ghost_cells[1]):
        data[:,i,:]  = 0
    for i in range(-num_ghost_cells[2],num_ghost_cells[2]):
        data[:,:,i]  = 0

def compute_divV(data):

    vectorCalc  = vector_calculus.VectorDifferentialOperators(data.grid, 
                                                              fd_kernel)

    DivV = vectorCalc.div( np.array([data.vr,data.vclt,data.vlon]) )

    if data.grid.num_ghost_cells != (0,0,0):
        empty_ghost_cells(DivV, data.grid.num_ghost_cells)
        fill_lon_ghost_cells(DivV, data.grid.num_ghost_cells[2])
    data.add_variable(DivV / data.grid.r_unit, "DivV")
