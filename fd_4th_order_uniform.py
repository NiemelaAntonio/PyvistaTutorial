"""Module for computing derivatives of a scalar field.
"""
import numpy as np

num_required_ghost_points = (2,2,2)

def partial_1(V, delta, num_ghost_points):
    """Partial derivative of scalar V with respect to x1.

    Args:
        V       : 3D scalar field defined at grid points
        delta   : Constant spacing between grid points in x1-direction
    """
    dVdx1     = np.zeros_like(V)

    dVdx1[2:-2,:,:] = ( V[ :-4,:,:] - 8.0 *  V[1:-3,:,:] \
                     -  V[4:  ,:,:] + 8.0 *  V[3:-1,:,:] )  

    if num_ghost_points == 2:
        #fill ghostcells
        dVdx1[  :2,:,:] = dVdx1[-4:-2,:,:]
        dVdx1[-2: ,:,:] = dVdx1[ 2: 4,:,:]

    elif num_ghost_points == 0:
        # alternating 
        dVdx1[ 0,:,:] =  -25.0*V[ 0,:,:] + 48.0*V[ 1,:,:]\
                         -36.0*V[ 2,:,:] + 16.0*V[ 3,:,:]\
                         - 3.0*V[ 4,:,:]

        dVdx1[-1,:,:] = -25.0*dVdx1[ -1,:,:] + 48.0*V[ -2,:,:]\
                        -36.0*dVdx1[ -3,:,:] + 16.0*V[ -4,:,:]\
                        - 3.0*dVdx1[ -5,:,:]


        dVdx1[ 1,:,:] =  - 3.0*V[ 0,:,:] - 10.0*V[ 1,:,:]\
                         +18.0*V[ 2,:,:] -  6.0*V[ 3,:,:]\
                             + V[ 4,:,:]
         
        dVdx1[-2,:,:] = - 3.0*V[ -1,:,:] - 10.0*V[ -2,:,:]\
                        +18.0*V[ -3,:,:] -  6.0*V[ -4,:,:]\
                             + V[ -5,:,:]
    else:
        raise RuntimeError('Only zero or two ghostcells supported')

    return dVdx1/(12.0*delta)


def partial_2(V, delta, num_ghost_points):
    """Partial derivative of scalar V with respect to x2.

    Args:
        V       : 3D scalar field defined at grid points
        delta   : Constant spacing between grid points in x2-direction
    """
    dVdx2     = np.zeros_like(V)

    dVdx2[:,2:-2,:] = ( V[:, :-4,:] - 8.0 *  V[:,1:-3,:] \
                     -  V[:,4:  ,:] + 8.0 *  V[:,3:-1,:] )  

    if num_ghost_points == 2:
        #fill ghostcells
        dVdx2[:,  :2,:] = dVdx2[:,-4:-2,:]
        dVdx2[:,-2: ,:] = dVdx2[:, 2: 4,:]

    elif num_ghost_points == 0:
        # alternating 
        dVdx2[:, 0,:] =  -25.0*V[:, 0,:] + 48.0*V[:, 1,:]\
                         -36.0*V[:, 2,:] + 16.0*V[:, 3,:]\
                         - 3.0*V[:, 4,:]

        dVdx2[:,-1,:] = -25.0*V[:, -1,:] + 48.0*V[:, -2,:]\
                        -36.0*V[:, -3,:] + 16.0*V[:, -4,:]\
                        - 3.0*V[:, -5,:]


        dVdx2[:, 1,:] =  - 3.0*V[:, 0,:] - 10.0*V[:, 1,:]\
                         +18.0*V[:, 2,:] -  6.0*V[:, 3,:]\
                             + V[:, 4,:]
         
        dVdx2[:,-2,:] = - 3.0*V[:, -1,:] - 10.0*V[:, -2,:]\
                        +18.0*V[:, -3,:] -  6.0*V[:, -4,:]\
                             + V[:, -5,:]
    else:
        raise RuntimeError('Only zero or two ghostcells supported')

    return dVdx2/(12.0*delta)




def partial_3(V, delta, num_ghost_points):
    """Partial derivative of scalar V with respect to x1.

    Args:
        V       : 3D scalar field defined at grid points
        delta   : Constant spacing between grid points in x3-direction
    """
    dVdx3     = np.zeros_like(V)

    dVdx3[:,:,  2:-2] =( V[:,:, :-4] - 8.0 *  V[:,:,1:-3] \
                       - V[:,:,4:  ] + 8.0 *  V[:,:,3:-1] )  

    if num_ghost_points == 2:
        #fill ghostcells
        dVdx3[:,:,  :2] = dVdx3[:,:,-4:-2]
        dVdx3[:,:,-2: ] = dVdx3[:,:, 2: 4]

    elif num_ghost_points == 0:
        # alternating 
        dVdx3[:,:, 0] =  -25.0*V[:,:, 0] + 48.0*V[:,:, 1]\
                         -36.0*V[:,:, 2] + 16.0*V[:,:, 3]\
                         - 3.0*V[:,:, 4]

        dVdx3[:,:,-1] = -25.0*V[:,:, -1] + 48.0*V[:,:, -2]\
                        -36.0*V[:,:, -3]+ 16.0*V[:,:, -4]\
                        - 3.0*V[:,:, -5]


        dVdx3[:,:, 1] =  - 3.0*V[:,:, 0] - 10.0*V[:,:, 1]\
                         +18.0*V[:,:, 2]  -  6.0*V[:,:, 3]\
                             + V[:,:, 4]
         
        dVdx3[:,:,-2] = - 3.0*V[:,:, -1] - 10.0*V[:,:, -2]\
                         +18.0*V[:,:, -3]  -  6.0*V[:,:, -4]\
                             + V[:,:, -5]
    else:
        raise RuntimeError('Only zero or two ghostcells supported')

    return dVdx3/(12.0*delta)



