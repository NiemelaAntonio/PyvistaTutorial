import logging,traceback
import pyvista as pv 
import numpy as np
import rh_solver as rh
import constants as constants
import transform as transform
from vector_calculus import dot_product
try:
    import pymeshfix as mf
except: 
    pass
try:
    # try to silence some annoying vtk error messages
    import vtk 
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
except:
    pass

def foreshock_analyzer(shock, mesh,eps=4e-2):

    mesh.compute_implicit_distance(shock,inplace=True)
    # point ids upstream of the shock
    fs_id_pts = np.nonzero(mesh['implicit_distance']> 0)[0]
    #point ids downstream of the shock
    bs_id_pts = np.nonzero(mesh['implicit_distance']<=0)[0]

    # if no upstream point are found, simply return the mesh
    if fs_id_pts.shape[0] == 0: 
        mesh              = mesh.cell_data_to_point_data()
        mesh['distance']  = np.copy(mesh['implicit_distance'][:])
        return mesh

    # b is a vector pointing outward
    mesh['b'] = np.transpose(np.transpose(mesh['B'])*mesh['B_pol'])
    mesh      = mesh.cell_data_to_point_data()

    # Get upstream points
    fs_pts = pv.wrap(mesh.points[fs_id_pts])

    # Put b vector downstream to zero, so that fieldlines are not traced downstream
    mesh.point_data['b'][bs_id_pts] = 0

    # For upstream points, compute fieldlines backward since b is pointing outward
    streamlines = mesh.streamlines_from_source(fs_pts, 'b', 
                              max_step_length=0.15, 
                              max_steps=10000, 
                              compute_vorticity=False,
                              integration_direction='backward')                      
    streamlines = streamlines.compute_arc_length()

    # Make scalar field storing the distance to the shock.
    # Upstream the distance is positive and measured along the magnetic field 
    # Downstream it is the shortest distance to the shock
    mesh['distance'] = np.copy(mesh['implicit_distance'][:])
    mesh['distance'][fs_id_pts] = np.nan

    # Each cell of mesh correpsonds to one fieldline. Each cell has multiple points
    # where arc_length jumps back to zero, a new fieldline starts. 
    cell_transition_points =np.nonzero(np.diff(streamlines['arc_length'])<0)[0]
    # the last cell is skipped by this method; Include it if the cell contains a point
    if streamlines['arc_length'][-1] > 0:  
        cell_transition_points = np.append(cell_transition_points,streamlines.n_points -1)

    # fieldines that reach the shock should have stagnated ('ReasonForTermination' == 6), since mesh.point_data['b'][bs_id_pts] = 0
    # That is, in the downstream region the b field is set to zero.

    cells_that_reach_shock = np.logical_and(streamlines['ReasonForTermination']>1, 
                                            np.abs(streamlines['implicit_distance'][cell_transition_points])<eps)
    # cells_that_reach_shock = streamlines['ReasonForTermination']>1


    cell_transition_points = cell_transition_points[cells_that_reach_shock]
    mesh['distance'][fs_id_pts[streamlines['SeedIds'][cells_that_reach_shock ]]] =\
    streamlines['arc_length'][cell_transition_points]

    # for i in range(streamlines.n_cells):
    #     temp = streamlines.extract_cells(i)
    #     idxs = np.nonzero(temp.point_data['impl_dist_diff'][:-1])[0]  
    #     if idxs.shape[0] > 0:
    #         mesh['distance'][fs_id_pts[temp['SeedIds'][0]]] = temp['arc_length'][idxs[0]] + temp['implicit_distance'][idxs[0]]

    return mesh

def get_cobpoint_data(sc_pos, mesh, shock):

    #compute  the fieldline and determine it's distance from the shock
    mesh      = mesh.cell_data_to_point_data()
    streamline = mesh.streamlines('B',start_position=sc_pos, 
                                   max_step_length=0.5, #in cell_length unit
                                   max_steps=10000, #Default: 2000.
                                   compute_vorticity=False)
    if streamline.n_points == 0: 
        logging.debug('streamline could not be computed!')
        return None
    else:
        streamline.compute_implicit_distance(shock,inplace=True)

    #Not sure which method is best
    try:
        return _get_cobpoint_from_intersection(sc_pos, streamline, shock)
    except:
        pass
    try:
        return  _get_cobpoint_from_collision(sc_pos, streamline, shock)
    except:
        return None
    # try:
    #     return  _get_cobpoint_from_implicit_distance(sc_pos, streamline, shock)
    # except:
    #     return None


def _get_cobpoint_from_intersection(sc_pos, streamline, shock):

    #############
    # Option 1: #
    #############

    streamtubes = streamline.tube(radius=0.0001).triangulate()
    
    # change log level to avoid annoying VTK messages 
    log_lvl = logging.root.level
    logger  = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    try:
        intersection, _,_ = shock.intersection(streamtubes,split_first=False, split_second=False)
    except Exception as e:
        logger.setLevel(log_lvl)
        raise e
    else:
        logger.setLevel(log_lvl)

    if intersection.n_points == 0:
        logging.debug('no intersection found between shock and streamline!')
        return None

    intersection_blocks = intersection.split_bodies()
    d_closest = np.infty

    for block in intersection_blocks:
        d = np.linalg.norm(np.asarray(sc_pos) - np.asarray(block.center))
        if d < d_closest:
            d_closest = d
            block_closest = block

    _,cobpoint  = shock.find_closest_cell(block_closest.center,return_closest_point=True)
    cobpoint_pv = pv.PolyData(cobpoint)
    streamline  = streamline.compute_arc_length()
    idx         = streamline.find_closest_point(cobpoint)
    cobpoint_pv.point_data['dist_from_shock'] = streamline['arc_length'][idx]\
                                      + np.sign(streamline['implicit_distance'][idx])\
                                      * np.linalg.norm(np.asarray(streamline.points[idx])\
                                                     - np.asarray(cobpoint))
    
    cobpoint_pv = cobpoint_pv.sample(shock)

    return cobpoint_pv

def _get_cobpoint_from_collision(sc_pos, streamline, shock):

    #############
    # Option 2: #
    #############
    streamtubes = streamline.tube(radius=0.0001).triangulate()

    log_lvl = logging.root.level
    logger  = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    try:
        collisions, ncol = shock.collision(streamtubes, generate_scalars=True)
    except Exception as e:
        logger.setLevel(log_lvl)
        raise e
    else:
        logger.setLevel(log_lvl)

    collisions       = shock.extract_cells(collisions['ContactCells'])

    if ncol == 0:
        logging.debug('no collisions found between shock and streamline!')
        return None

    collision_blocks = collisions.split_bodies()
    d_closest = np.infty

    for block in collision_blocks:
        d = np.linalg.norm(np.asarray(sc_pos) - np.asarray(block.center))
        if d < d_closest:
            d_closest = d
            block_closest = block

    cobpoint_pv = block.integrate_data()
    for key in cobpoint_pv.cell_data:
        cobpoint_pv.cell_data[key] /=  cobpoint_pv.cell_data['Area'] if key != 'Area' else 1

    streamline = streamline.compute_arc_length()
    idx        = streamline.find_closest_point(cobpoint_pv.points[0])
    cobpoint_pv.point_data['dist_from_shock'] = streamline['arc_length'][idx]\
                                      + np.sign(streamline['implicit_distance'][idx])\
                                      * np.linalg.norm(np.asarray(streamline.points[idx])\
                                                     - np.asarray(cobpoint_pv.points[0]))
    return cobpoint_pv


def _get_cobpoint_from_implicit_distance(sc_pos, streamline, shock):

    #############
    # Option 3: #
    #############

    # first sign switch in implicit distance indicates the shock crossing
    idxs = np.where(np.diff(np.sign(streamline['implicit_distance'])) != 0)[0]
    if len(idxs) == 0: 
        logging.debug('No sign switch in implicit distance from streamline to shock!')
        return None
    elif len(idxs) == 1:
        idx = idxs[0]
    else: 
        # in case multiple shock crossings are found, take the one furthest away of the sun
        candidates = np.asarray(streamline.points[idxs])
        idx = idxs[np.argmax(candidates[:,0]*candidates[:,0]+\
                             candidates[:,1]*candidates[:,1]+\
                             candidates[:,2]*candidates[:,2])]

    # ensure this sign crossing is indeed at the shock surface
    # if (np.abs(streamline['implicit_distance'][idx])>0.05): return None
    # sign switch is from idx to idx+1, take the 
    idx += np.argmin([streamline['implicit_distance'][idx], streamline['implicit_distance'][idx+1]])

    # find the shock surface point that is closest to the fieldline. In theory, this should
    # be the intersection point of the fieldline with the shock
    _,cobpoint  = shock.find_closest_cell(streamline.points[idx],return_closest_point=True)

    # check if the cobpoint is indeed close to the streamline

    MAX_DISTANCE_COB2FIELDLINE = 1e-4
    if np.linalg.norm(np.asarray(cobpoint) - np.asarray(streamline.points[idx]))\
      > MAX_DISTANCE_COB2FIELDLINE:
        logging.debug('Cobpoint candidate is too far from streamline!')
        return None

    # store the point as a PolyData mesh and sample the shock parameters at the point
    cobpoint_pv = pv.PolyData(cobpoint)

    #find distance to shock (along streamline)

    streamline = streamline.compute_arc_length()
    cobpoint_pv.point_data['dist_from_shock'] = streamline['arc_length'][idx]\
                                    + np.sign(streamline['implicit_distance'][idx])\
                                      * np.linalg.norm(np.asarray(streamline.points[idx])\
                                                   - np.asarray(cobpoint))
    cobpoint_pv = cobpoint_pv.sample(shock)

    return cobpoint_pv

def calculate_shock_speed(shock_next,shock_prev, dt,
                           use_implicit_distance=False,
                           unit=constants.au):
    """
    Calculate shock speeds for a shock front represented by two consecutive shock surfaces.

    This function computes the shock speed based on the difference between two shock surfaces,
    `shock_next` and `shock_prev`, over a given time step `dt`. The shock speed can be calculated
    in two ways, depending on the value of `use_implicit_distance`:

    Parameters:
    -----------
    shock_next : pv.PolyData
        The shock surface at the next time step.

    shock_prev : pv.PolyData
        The shock surface at the previous time step.

    dt : float
        The time step for which the shock speed is calculated.

    use_implicit_distance : bool, optional (default=False)
        If True, the shock speed is calculated based on implicit distance between
        `shock_next` and `shock_prev`. This relies on the vtkImplicitPolyDataDistance
        method of the VTK lib, which sometimes produces segmentation faults

    unit : float, optional (default=constants.au)
        The unit for the shock speed. Typically, it's expressed in astronomical units (AU).

    Returns:
    --------
    shock_next : pv.PolyData
        The input `shock_next` PolyData object with an additional 'shock_speed' array containing
        the calculated shock speeds for each point on the shock surface.
    """
    
    if "Normals" not in shock_next.point_data:
        shock_next.compute_normals(inplace=True)
    
    if use_implicit_distance:
        shock_next.compute_implicit_distance(shock_prev, inplace=True)
        shock_next['shock_speed'] = np.abs(shock_next['implicit_distance'])/dt * unit

    else:
        _,closest_points  = shock_prev.find_closest_cell(shock_next.points, 
                                                 return_closest_point=True)

        shock_next['shock_speed'] = np.abs(np.linalg.norm(shock_next.points - closest_points, 
                                                   axis=1))/dt * unit
        
    return shock_next

def get_spatial_shock_extent(shock):

    crt = np.asarray(shock.points).T
    sph = transform.cartesian_coordinate_to_spherical(crt)
    return [np.nanmin(sph[0]),np.nanmax(sph[0]),
            np.nanmin(sph[1]),np.nanmax(sph[1]),
            np.nanmin(sph[2]),np.nanmax(sph[2])]

def estimate_spatial_shock_extent_from_CME_inj_pars(datetime,cme_inj_pars,grid,**kwargs):

    fudge = 1.25
    dt      = (datetime - cme_inj_pars["injection time"]).total_seconds()

    clt_min = max(0,    cme_inj_pars["clt"] - kwargs.get('clt_fudge', fudge)*cme_inj_pars["halfwidth"]) 
    clt_max = min(np.pi,cme_inj_pars["clt"] + kwargs.get('clt_fudge', fudge)*cme_inj_pars["halfwidth"]) 

    lon_min = max(-np.pi,cme_inj_pars["lon"] - kwargs.get('lon_fudge',fudge)*cme_inj_pars["halfwidth"]) 
    lon_max = min( np.pi,cme_inj_pars["lon"] + kwargs.get('lon_fudge',fudge)*cme_inj_pars["halfwidth"]) 
    
    dt_end  = (datetime - cme_inj_pars["end time"]).total_seconds()  
    
    r_min   = max(grid.x1[0],grid.x1[0] + cme_inj_pars["injection speed"]* dt_end / fudge )
    r_max   = grid.x1[0]*kwargs.get('r_fudge',fudge) + cme_inj_pars["injection speed"]*dt\
    *kwargs.get('V_fudge',(1.2+2.5*("spheromak" in cme_inj_pars["name"].lower())))

    return [r_min,r_max, clt_min,clt_max, lon_min,lon_max]

def estimate_spatial_shock_extent_from_prev_shock(dt,shock_prev,cme_inj_pars,**kwargs):

    lims = get_spatial_shock_extent(shock_prev)

    lims[2] = min(lims[2],cme_inj_pars["clt"] - cme_inj_pars["halfwidth"]) 
    lims[3] = max(lims[3],cme_inj_pars["clt"] + cme_inj_pars["halfwidth"]) 
    lims[4] = min(lims[4],cme_inj_pars["lon"] - cme_inj_pars["halfwidth"]) 
    lims[5] = max(lims[5],cme_inj_pars["lon"] + cme_inj_pars["halfwidth"]) 

    # The CME might not be expanding self-similarly, so it's halwidth might
    # increase over time. How to account for this?
    # Just add 5 degrees safety margin?

    fudge = kwargs.get('angle_fudge',np.radians(5))
    lims[2] = max(0,      lims[2] - fudge )
    lims[3] = min( np.pi, lims[3] + fudge )
    lims[4] = max(-np.pi, lims[4] - fudge )
    lims[5] = min( np.pi, lims[5] + fudge )

    # get an estimate of the maximum speed of the shock in au/s
    speed_max  = cme_inj_pars["injection speed"]
    speed_max *= ( 1 + ("spheromak" in cme_inj_pars['name'].lower()) )

    if 'shock_speed' in shock_prev.cell_data:
        speed_prev = np.nanmax(shock_prev['shock_speed'])/constants.au 
        # skip suspiciously small speeds
        if speed_prev > 0.25 * speed_max:
            speed_max  = min(speed_max, speed_prev)
    
    # increase the outer radial limit; speed_fudge is needed 
    # because shock might have accelerated

    lims[1] += speed_max*kwargs.get('speed_fudge',2)*dt

    return lims

def reasonable_shock_limits(shock_lims, shock_lims_prev, cme_inj_pars):
    
    
    if not np.all(np.isfinite(shock_lims_prev)):
        return shock_lims

    dlim = [shock_lims_prev[1] - shock_lims_prev[0],
            shock_lims_prev[3] - shock_lims_prev[2],
            shock_lims_prev[5] - shock_lims_prev[4]]

    if np.any(np.array(dlim) <= 0):
        return shock_lims

    #condition1 =  dlim[0] >  shock_lims[1] * np.sin(cme_inj_pars["halfwidth"])\
    #                       / (1 + np.sin(cme_inj_pars["halfwidth"]))
    

    if shock_lims_prev[0] > 1.25 * shock_lims[0]:
        shock_lims_prev[0] =  shock_lims[0]    
        
    if shock_lims_prev[1] < 0.75 * shock_lims[1]:
        shock_lims_prev[1] =  shock_lims[1]      

    # These conditions should already be satisfied 
    if shock_lims_prev[2] > cme_inj_pars["clt"] - cme_inj_pars["halfwidth"]:
        shock_lims_prev[2] =  shock_lims[2]
        
    if shock_lims_prev[3] < cme_inj_pars["clt"] + cme_inj_pars["halfwidth"]:
        shock_lims_prev[3] =  shock_lims[3]

    if shock_lims_prev[4] > cme_inj_pars["lon"] - cme_inj_pars["halfwidth"]:
        shock_lims_prev[4] =  shock_lims[4]
    
    if shock_lims_prev[5] < cme_inj_pars["lon"] + cme_inj_pars["halfwidth"]:
        shock_lims_prev[5] =  shock_lims[5]
    
    return shock_lims_prev


def get_iso_surface(pv_mesh,var,iso_lvl,smooth_iter = 20, **kwargs):

    pv_mesh.set_active_scalars(var)
    surfaces = pv_mesh.cell_data_to_point_data().contour(isosurfaces=[iso_lvl])
    # remove noise; Assume shock is the largest surface among the detected 
    # isosurfaces. 
    surface  = surfaces.extract_largest()

    # only retain the points that have an outward normal: n_r dot r > 0
    surface.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    [r,t,p]=transform.cartesian_coordinate_to_spherical(surface.points.T)
    surface['nr'] =  surface['Normals'][:, 0]*np.sin(t)*np.cos(p)\
                    + surface['Normals'][:, 1]*np.sin(t)*np.sin(p)\
                    + surface['Normals'][:, 2]*np.cos(t)
    
    surface.point_data['r'] = r
    surface.clip_scalar(scalars='nr', value=0, inplace=True, invert=False)
    
    # Look for the surface that is furthest away from the origin
    bodies  = surface.split_bodies()
    ibody   = np.argmax([np.nanmax(block['r']) for block in bodies])
    surface = bodies[ibody].extract_surface()
    #problematic line:
    #surface = (surface.extract_surface()).extract_largest() 

    if smooth_iter > 0:
        surface.smooth_taubin(inplace=True,n_iter=smooth_iter,
                              boundary_smoothing=False)

    if kwargs.get('meshfix', False):
        try:
            meshfix = mf.MeshFix(surface)
            meshfix.repair(verbose=False)
            surface =  meshfix.mesh
            surface.smooth(inplace=True,n_iter=smooth_iter)
            surface = surface.sample(pv_mesh)
            surface.compute_normals(cell_normals=False, point_normals=True, inplace=True)
            [r,t,p]=transform.cartesian_coordinate_to_spherical(surface.points.T)
            nr =  surface['Normals'][:, 0]*np.sin(t)*np.cos(p)\
                + surface['Normals'][:, 1]*np.sin(t)*np.sin(p)\
                + surface['Normals'][:, 2]*np.cos(t) 
            surface = surface.extract_points(np.arange(surface.n_points)[nr > 0.0])
            surface = (surface.extract_surface()).extract_largest() 
        except:
            pass

    return surface
   
def calculate_upstream_shock_conditions(shock):

    np.seterr(invalid='ignore')
    # store upstream magnetic field magnitude 
    shock.point_data['Bmag']  = np.linalg.norm(shock.point_data['B'], axis=1)

    # store and normalize shock normals
    shock.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    shock.point_data["Normals"] = np.transpose(shock.point_data["Normals"].T /\
                                  np.linalg.norm(shock.point_data["Normals"], 
                                  axis=1))

    if "shock_vel_dir" in shock.point_data.keys():
        n = shock.point_data["shock_vel_dir"].T 
    else:
        n = shock.point_data["Normals"].T

    shock.point_data['plasma_beta'] = 2 * constants.mu0 * shock.point_data["P"]\
                                    / (shock.point_data['Bmag']*shock.point_data['Bmag'] )

    shock.point_data['theta_Bn'] = np.arccos(dot_product(shock.point_data["B"].T,n)/shock.point_data['Bmag'])
    shock.point_data['V_u']      = dot_product(shock.point_data["V"].T,n)
    
    #the 0.5 is to convert the plasma number density to proton number density
    shock.point_data['V_A'] = shock.point_data['Bmag']\
                            / np.sqrt(constants.mu0 * 0.5 * constants.mp * shock.point_data['n'])

    gamma =  5./3. 
    Vs_u  = np.sqrt(gamma * shock.point_data["P"] / (0.5 * constants.mp * shock.point_data['n'])) 
    Vsqr  = shock.point_data['V_A']**2+ Vs_u**2

    shock.point_data['V_fms'] = np.sqrt(0.5 * (Vsqr + np.sqrt(Vsqr**2\
                              -(2*shock.point_data['V_A']*Vs_u\
                                 *np.cos(shock.point_data['theta_Bn']))**2)))

    return shock

def solve_rankine_hugoniot(shock):
    np.seterr(invalid='ignore')
    rh_solver   = rh.RH_Solver()
    
    shock.point_data['r_g']   = np.full([shock.n_points], np.nan)
    shock.point_data['r_b']   = np.full([shock.n_points], np.nan)
    shock.point_data['r_p']   = np.full([shock.n_points], np.nan)
    shock.point_data['V_d']   = np.full([shock.n_points], np.nan)
    shock.point_data['M_A']   = np.full([shock.n_points], np.nan)
    shock.point_data['M_fms'] = np.full([shock.n_points], np.nan)
    shock.point_data['VR']    = np.full([shock.n_points], np.nan) 
    
    for i in range(shock.n_points):
        try:
            rh_solver.initialize(shock.point_data["shock_speed"][i],
                             shock.point_data["theta_Bn"][i],
                             shock.point_data["V_u"][i],
                             shock.point_data["V_A"][i],
                             shock.point_data["plasma_beta"][i],
                             shock.point_data["P"][i], 
                             0.5 * constants.mp * shock.point_data['n'][i])
        
            shock.point_data['M_A'][i]   = np.abs(rh_solver.M_A)
            if shock.point_data['M_A'][i] < 1: continue
            if rh_solver.rg_sh < 1: continue

            shock.point_data['M_fms'][i]= np.abs(rh_solver.M_fms)
            shock.point_data['r_g'][i]  = rh_solver.rg_sh
            shock.point_data['r_b'][i]  = rh_solver.rb_sh
            shock.point_data['r_p'][i]  = rh_solver.rp_sh
            shock.point_data['V_d'][i]  = rh_solver.Vd_n
            shock.point_data['VR'][i]   = (shock.point_data['V_d'][i]\
                - shock.point_data["V_u"][i])/shock.point_data["V_u"][i]
        except Exception as e:
            #logging.error(traceback.format_exc())
            continue

    shock.point_data['r_c']  = shock.point_data["r_g"]*(1 - 1/shock.point_data["M_A"])
    shock.point_data['mu_i'] = np.sqrt(1-1/shock.point_data['r_b'])  

    return shock
