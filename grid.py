
"""Module implementing computational grid
"""

import numpy as np
from sys import exit

class EmptyContainer():
    pass


class Grid(object):

    def __init__(self, **kwargs):
        pass 

    def initialize(self, **kwargs):

        #
        # Construct global grid coordinates
        #

        coordinate_axis = list()

        # Accepted axis input labels

        accepted_axis_label = ( ["r"],
                                ["t", "clt", "theta"],
                                ["p", "lon", "phi"] )

        self.r_unit = kwargs.get("r_unit",1)

        for dim in range(3):

            #
            # Construct in-domain coordinates in this dim
            #

            for key in kwargs:
                if key in accepted_axis_label[dim]:
                    axis_input = kwargs.get(key, None)

            if axis_input is None:
                print("no valid axis_label found")
                exit()
            else:
                x = np.array(axis_input)

            coordinate_axis.append(x)

        self.num_ghost_cells = kwargs.get("num_ghost_points", (2,2,2))

        axis_type = kwargs.get("axis_type", "centers")

        self.center_coords        = EmptyContainer()
        self.edge_coords          = EmptyContainer()
        self.indomain_edge_coords = EmptyContainer()

        self.dr   = coordinate_axis[0][1:] - coordinate_axis[0][:-1]
        self.dclt = coordinate_axis[1][1:] - coordinate_axis[1][:-1]
        self.dlon = coordinate_axis[2][1:] - coordinate_axis[2][:-1]


        if axis_type == "edges":

            self.edge_coords.r   = coordinate_axis[0]
            self.edge_coords.clt = coordinate_axis[1]
            self.edge_coords.lon = coordinate_axis[2]

        elif axis_type == "centers":

            self.edge_coords.r   = coordinate_axis[0][:-1] + 0.5 * self.dr
            self.edge_coords.clt = coordinate_axis[1][:-1] + 0.5 * self.dclt
            self.edge_coords.lon = coordinate_axis[2][:-1] + 0.5 * self.dlon

            self.edge_coords.r   = np.concatenate([[self.edge_coords.r[0] - self.dr[0]], 
                                                    self.edge_coords.r,   
                                                   [self.edge_coords.r[-1] + self.dr[-1]]])

            self.edge_coords.clt   = np.concatenate([[self.edge_coords.clt[0] - self.dclt[0]], 
                                                      self.edge_coords.clt,   
                                                     [self.edge_coords.clt[-1] + self.dclt[-1]]])

            self.edge_coords.lon   = np.concatenate([[self.edge_coords.lon[0] - self.dlon[0]], 
                                                      self.edge_coords.lon,   
                                                     [self.edge_coords.lon[-1] + self.dlon[-1]]])

        else:
            print("no valid axis_type found")
            exit()
      
        self.center_coords.r   = 0.5*(self.edge_coords.r[1:]   + self.edge_coords.r[:-1])
        self.center_coords.clt = 0.5*(self.edge_coords.clt[1:] + self.edge_coords.clt[:-1])
        self.center_coords.lon = 0.5*(self.edge_coords.lon[1:] + self.edge_coords.lon[:-1])

        self.indomain_edge_coords.r   = self.edge_coords.r  [self.num_ghost_cells[0]:-self.num_ghost_cells[0]+len(self.edge_coords.r)]
        self.indomain_edge_coords.clt = self.edge_coords.clt[self.num_ghost_cells[1]:-self.num_ghost_cells[1]+len(self.edge_coords.clt)]
        self.indomain_edge_coords.lon = self.edge_coords.lon[self.num_ghost_cells[2]:-self.num_ghost_cells[2]+len(self.edge_coords.lon)]

        self.x1  = self.center_coords.r
        self.x2  = self.center_coords.clt
        self.x3  = self.center_coords.lon

        self.dx1 = self.dr
        self.dx2 = self.dclt
        self.dx3 = self.dlon

        self.num_cells = (len(self.x1),len(self.x2),len(self.x3))


    def get_index_voi(self, lims):

        get_idx_xi = lambda x, pt : (np.abs( x - pt)).argmin()

        return [get_idx_xi(self.center_coords.r,lims[0]) ,
                get_idx_xi(self.center_coords.r,lims[1])+1,
                get_idx_xi(self.center_coords.clt,lims[2]),
                get_idx_xi(self.center_coords.clt,lims[3])+1,
                get_idx_xi(self.center_coords.lon,lims[4] ),
                get_idx_xi(self.center_coords.lon,lims[5])+1
                ]
        


        





