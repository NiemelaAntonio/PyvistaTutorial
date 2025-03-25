
"""Module defining vector calculus operators
"""

import numpy as np

def dot_product(U,V):
    return U[0]*V[0] + U[1]*V[1] + U[2]*V[2]

def norm(U):
    return np.sqrt(dot_product(U,U))

def distance_between_two_cart_pts(p1,p2):
    diff = np.asarray(p1) - np.asarray(p2)
    return np.sqrt(dot_product(diff,diff))

def cross_product(b,c):
    
    
    ax = b[1]*c[2] - b[2]*c[1]
    ay = b[2]*c[0] - b[0]*c[2]
    az = b[0]*c[1] - b[1]*c[0]
        
    return np.array([ax,ay,az])


def _scale_for_curl_in_spherical_geometry(V1, V2, V3, grid):

    r,clt,lon = np.meshgrid(grid.x1, grid.x2, grid.x3,  indexing='ij')
    sin_clt = np.sin(clt)
    
    V1s      = np.copy(V1)
    V2s      = V2*r
    V3s      = V3*r*sin_clt
    
    return V1s, V2s, V3s


def _scale_curl_differences_in_spherical_geometry(dV1sdx2, dV1sdx3,
                                                  dV2sdx1, dV2sdx3,
                                                  dV3sdx1, dV3sdx2,
                                                  grid):
                                                  

    r,clt,lon = np.meshgrid(grid.x1, grid.x2, grid.x3,  indexing='ij')
    sin_clt = np.sin(clt)
    
    curlV_1 = ( dV3sdx2 - dV2sdx3)/(r*r*sin_clt)
    curlV_2 = ( dV1sdx3 - dV3sdx1)/(r*sin_clt)
    curlV_3 = ( dV2sdx1 - dV1sdx2)/(r)

    return curlV_1, curlV_2, curlV_3

def _scale_for_div_in_spherical_geometry(V1, V2, V3, grid):

    r,clt,lon = np.meshgrid(grid.x1, grid.x2, grid.x3,  indexing='ij')
    sin_clt = np.sin(clt)

    V1s = V1*r*r*sin_clt
    V2s = V2*r*sin_clt
    V3s = V3*r

    return V1s, V2s, V3s
    
def _scale_for_tensor_div_in_spherical_geometry(T,grid):

    Ts = np.copy(T)
    
    r,clt,lon = np.meshgrid(grid.x1, grid.x2, grid.x3,  indexing='ij')
    sin_clt = np.sin(clt)
    
    
    Ts[0,0] = r*r*sin_clt * T[0,0]
    Ts[0,1] = r*sin_clt * T[0,1]
    Ts[0,2] = r * T[0,2]
    
    Ts[1,0] = r*r*sin_clt * T[1,0]
    Ts[1,1] = r*sin_clt * T[1,1]
    Ts[1,2] = r* T[1,2]
    
    Ts[2,0] = r*r*sin_clt * T[2,0]
    Ts[2,1] = r*sin_clt * T[2,1]
    Ts[2,2] = r* T[2,2]
    
    return Ts
    


def _grad_in_spherical_geometry(dVdx1, dVdx2, dVdx3, grid):

    r,clt,lon = np.meshgrid(grid.x1, grid.x2, grid.x3,  indexing='ij')
    sin_clt = np.sin(clt)
    
    gradV_x1 = dVdx1
    gradV_x2 = dVdx2/r
    gradV_x3 = dVdx3/(r*sin_clt)

    return gradV_x1, gradV_x2, gradV_x3


def _scale_div_differences_in_spherical_geometry(divsum, grid):

    r,clt,lon = np.meshgrid(grid.x1, grid.x2, grid.x3,  indexing='ij')
    sin_clt = np.sin(clt)
    
    rsqr_sin_th = r*r*sin_clt
    divV        = divsum/(rsqr_sin_th)

    return divV



class VectorDifferentialOperators():

    def __init__(self, grid, fdkernel, **kwargs):

        # Grid
        self.grid = grid

        
        # Finite-difference operator module
        self.fd   = fdkernel

        # Size of ghost zone. Default is to use the number of grid points
        # required by the compute differencing method
        self.num_ghost_cells = grid.num_ghost_cells

        # Check that the number of ghosts in grid and are compatible with
        # the chosen differencing method
        for dim in range(0, 3):
            if self.num_ghost_cells[dim] < fdkernel.num_required_ghost_points[dim]:
                raise ValueError("Incompatible number of ghost points assigned")



        self.curl = self._curl_spherical_uniform
        self.div  = self._div_spherical_uniform
        self.grad = self._grad_spherical_uniform
        self.conv = self._convective_derivative_spherical_uniform
        self.dot  = dot_product
        self.cross= cross_product
        
        self.div_tensor  = self._div_tensor_spherical_uniform
    
        
    def _convective_derivative_spherical_uniform(self,U,V,dV):
        
        r,clt,lon = np.meshgrid(self.grid.x1, self.grid.x2, self.grid.x3,  indexing='ij')
    
        h_inv = np.array([np.ones(r.shape),1./r, 1./r/np.sin(clt)])
        
        UdV = [0,0,0]
        
        for i in range(3):
            for j in range(3):
                UdV[i] += U[j]* dV[i,j] * h_inv[j]
                

        cot_clt = np.cos(clt) /  np.sin(clt) 
        
      
        UdV[0] -= (U[1]*V[1] + U[2] * V[2]           ) * h_inv[1]
        UdV[1] += (U[1]*V[0] - U[2] * V[2] * cot_clt ) * h_inv[1]
        UdV[2] += (U[2]*V[0] + U[2] * V[1] * cot_clt ) * h_inv[1]
        
        return np.array(UdV)
    


    def _curl_spherical_uniform(self, V):

            #
            # Rescale components
            #
            Vs = _scale_for_curl_in_spherical_geometry(V[0], V[1], V[2],
                                                       self.grid)

            #
            # Compute differences of scaled vector components
            #
            dV1sdx2 = self.fd.partial_2(Vs[0], self.grid.dx2[0],self.num_ghost_cells[1])
            dV1sdx3 = self.fd.partial_3(Vs[0], self.grid.dx3[0],self.num_ghost_cells[2])

            dV2sdx1 = self.fd.partial_1(Vs[1], self.grid.dx1[0],self.num_ghost_cells[0])
            dV2sdx3 = self.fd.partial_3(Vs[1], self.grid.dx3[0],self.num_ghost_cells[2])

            dV3sdx1 = self.fd.partial_1(Vs[2], self.grid.dx1[0],self.num_ghost_cells[0])
            dV3sdx2 = self.fd.partial_2(Vs[2], self.grid.dx2[0],self.num_ghost_cells[1])

            #
            # Assemble result
            #
            curlV = _scale_curl_differences_in_spherical_geometry(dV1sdx2, dV1sdx3,
                                                                 dV2sdx1, dV2sdx3,
                                                                 dV3sdx1, dV3sdx2,
                                                                 self.grid)

            return np.array(curlV)


    def _div_spherical_uniform(self, V):

        #
        # Rescale components
        #
        Vs = _scale_for_div_in_spherical_geometry(V[0], V[1], V[2],
                                                  self.grid)

        #
        # Compute differences of scaled vector components
        #
        dV1sdx1  = self.fd.partial_1(Vs[0], self.grid.dx1[0],self.num_ghost_cells[0])
        dV2sdx2  = self.fd.partial_2(Vs[1], self.grid.dx2[0],self.num_ghost_cells[1])
        dV3sdx3  = self.fd.partial_3(Vs[2], self.grid.dx3[0],self.num_ghost_cells[2])

        #
        # Assemble result
        #
        dVsum    = dV1sdx1 + dV2sdx2 + dV3sdx3

        return _scale_div_differences_in_spherical_geometry(dVsum, self.grid)

    def _div_tensor_spherical_uniform(self, T):
    
        r,clt,lon = np.meshgrid(self.grid.x1, self.grid.x2, self.grid.x3,  indexing='ij')
        sin_clt = np.sin(clt)

        Ts = _scale_for_tensor_div_in_spherical_geometry(T,self.grid)
        
        scale = 1./(r**2 * np.sin(clt)) 
        cot_clt = np.cos(clt) /  np.sin(clt) 
        
        #
        # Compute differences of scaled vector components
        #
        dT11dx1  = self.fd.partial_1(Ts[0,0], self.grid.dx1[0],self.num_ghost_cells[0])
        dT12dx2  = self.fd.partial_2(Ts[0,1], self.grid.dx2[0],self.num_ghost_cells[1])
        dT13dx3  = self.fd.partial_3(Ts[0,2], self.grid.dx3[0],self.num_ghost_cells[2])
        
        dT21dx1  = self.fd.partial_1(Ts[1,0], self.grid.dx1[0],self.num_ghost_cells[0])
        dT22dx2  = self.fd.partial_2(Ts[1,1], self.grid.dx2[0],self.num_ghost_cells[1])
        dT23dx3  = self.fd.partial_3(Ts[1,2], self.grid.dx3[0],self.num_ghost_cells[2])
        
        dT31dx1  = self.fd.partial_1(Ts[2,0], self.grid.dx1[0],self.num_ghost_cells[0])
        dT32dx2  = self.fd.partial_2(Ts[2,1], self.grid.dx2[0],self.num_ghost_cells[1])
        dT33dx3  = self.fd.partial_3(Ts[2,2], self.grid.dx3[0],self.num_ghost_cells[2])
        
        
        dT1  = (dT11dx1 + dT12dx2 + dT13dx3)*scale
        dT1 -= (T[1,1] + T[2,2])/r
        
        dT2  = (dT21dx1 + dT22dx2 + dT23dx3)*scale
        dT2 += (T[0,1] - T[2,2]*cot_clt)/r
        
        dT3  = (dT31dx1 + dT32dx2 + dT33dx3)*scale
        dT3 += (T[0,2] + T[1,2]*cot_clt)/r


        return np.array([dT1,dT2,dT3])



    def _grad_spherical_uniform(self, V):


        #
        # Compute differences of scaled vector components
        #
        dVdx1  = self.fd.partial_1(V, self.grid.dx1[0],self.num_ghost_cells[0])
        dVdx2  = self.fd.partial_2(V, self.grid.dx2[0],self.num_ghost_cells[1])
        dVdx3  = self.fd.partial_3(V, self.grid.dx3[0],self.num_ghost_cells[2])

        gradV_x1, gradV_x2, gradV_x3 \
            = _grad_in_spherical_geometry(dVdx1,
                                          dVdx2,
                                          dVdx3,
                                          self.grid)

        return np.array( (gradV_x1, gradV_x2, gradV_x3) )
        

    
