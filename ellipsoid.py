
"""Module implementing computational grid
"""

import numpy as np
import h5py,json
import pathlib
from datetime import datetime, timedelta
import shock_tracer.transform as transform
import shock_tracer.constants as constants
import pyvista as pv
from scipy import optimize

class Ellipsoid(object):

    def __init__(self, fname, unit=constants.au, rmax_fit=None):
        file_extension = pathlib.Path(fname).suffix
        if(file_extension == '.hdf5'):
    
            with h5py.File(fname,'r') as data:
            
                self.r    = data['r_c'][:]/unit
                self.clt  = data['clt_c'][:]
                self.lon  = data['lon_c'][:] 
                self.a    = data['a'][:]/unit
                self.b    = data['b'][:]/unit
                self.c    = data['c'][:]/unit
                self.tilt = data['tilt'][:] 
                
                dt  = datetime.strptime(data.attrs['datetime'], "%Y-%m-%dT%H-%M-%S") 

                self.datetimes = [dt + timedelta(seconds=t) for t in data['timeline'][:]]

                
        elif(file_extension == '.json'):
        
            with open(fname) as f:
                data = json.load(f)
                data = data["geometrical_model"]["parameters_uniform"]
                self.r    = np.asarray(data['rcenter'])  * constants.R_sun/unit
                self.clt  = (90-np.asarray(data['hglt'])) * np.pi/180
                self.lon  = np.asarray(data['hgln']) * np.pi/180
                self.a    = np.asarray(data['radaxis'])    * constants.R_sun/unit
                self.b    = np.asarray(data['orthoaxis1']) * constants.R_sun/unit
                self.c    = np.asarray(data['orthoaxis2']) * constants.R_sun/unit
                self.tilt = np.asarray(data['tilt'])  * np.pi/180            
                datetimes = data['time']  
            try:
                self.datetimes = [datetime.strptime(date_string , "%Y-%m-%dT%H:%M:%S.%f")\
                for date_string in datetimes]
            except:
                self.datetimes = [datetime.strptime(date_string.decode("utf-8") , "%Y-%m-%dT%H:%M:%S")\
                         for date_string in datetimes]
            
        else:
            raise IOError('Unknow file extension for ellipsoid file: '+ file_extension)                 
        
        if rmax_fit is not None:           
            nose  = self.a+self.r
            i_max = np.argmin(np.abs(nose - rmax_fit))
        
            self.a    = self.a[:i_max]
            self.b    = self.b[:i_max]
            self.c    = self.c[:i_max]
            self.r    = self.r[:i_max]
            self.lon  = self.lon[:i_max]
            self.clt  = self.clt[:i_max]
            self.tilt = self.tilt[:i_max]
            self.datetimes = self.datetimes[:i_max]
              
        
    def at_datetime(self, datetime,sample=None,**kwargs):
    
        for it, dt in enumerate(self.datetimes):
            if dt > datetime: break
            
        par = (datetime-self.datetimes[it-1]).total_seconds()\
        / (self.datetimes[it]-self.datetimes[it-1]).total_seconds()
        
        r    = self._interpolate(self.r,it, par)
        clt  = self._interpolate(self.clt,it, par)
        lon  = self._interpolate(self.lon,it, par)
        
        a    = self._interpolate(self.a,it, par)
        b    = self._interpolate(self.b,it, par)
        c    = self._interpolate(self.c,it, par)
        tilt = self._interpolate(self.tilt,it, par)
        


        center_sphe = np.array([r,clt,lon])
        center_cart = transform.spherical_coordinate_to_cartesian(center_sphe)
        er = transform.spherical_vector_to_cartesian([1,0,0],center_sphe)
        
        ellipsoid = pv.ParametricEllipsoid(a,b,c,
                                          min_u=kwargs.get('min_u',-np.pi),
                                          max_u=kwargs.get('max_u', np.pi),
                                          u_res=kwargs.get('u_res',180), 
                                          v_res=kwargs.get('v_res',90), 
                                          )

        ellipsoid.translate(center_cart, inplace=True)

        ellipsoid.rotate_vector(vector=(0, -1, 0), angle=90-np.degrees(center_sphe[1]),
                                point=ellipsoid.center, inplace=True)
        ellipsoid.rotate_vector(vector=(0, 0, 1), angle=np.degrees(center_sphe[2]),
                                point=ellipsoid.center, inplace=True)

        ellipsoid.rotate_vector(vector=er, angle=tilt,
                                point=ellipsoid.center, inplace=True)

        #reduce number of cells to avoid extremely high resolution at poles
        # ellipsoid.decimate(0.25,volume_preservation=True,inplace=True)

        if sample is not None:
            ellipsoid = ellipsoid.sample(sample)
        
        # auto_orient_normals=True to get outward normals
        ellipsoid.compute_normals(auto_orient_normals=True, inplace=True, cell_normals=False)

        return ellipsoid

    def extend_until_shock_arrival(self, arrival_datetime, r_sc,clt_sc,lon_sc):


        def ellipsoid_IP(r_c, t_c, p_c, X_sc, r0,tilt0,a0,b0,c0):
    
            a = a0 * r_c / r0
            b = b0 * r_c / r0
            c = c0 * r_c / r0    

            center_sphe = [r_c,t_c,p_c]

            # rotate to a coordinate system in which e_x = e_rc, e_y = e_pc, and e_z = -e_tc   
            

            e1 = transform.spherical_vector_to_cartesian([1,0,0],center_sphe)
            e2 = transform.spherical_vector_to_cartesian([0,0,1],center_sphe)
            e3 = transform.spherical_vector_to_cartesian([0,-1,0],center_sphe)
            [x,y,z]  = [np.dot(X_sc,e1),  np.dot(X_sc,e2),  np.dot(X_sc,e3)]
    
            # take into account the tilt of the ellipse
    
            cos_rot = np.cos(tilt0);
            sin_rot = np.sin(tilt0);       
            y_rot   =  cos_rot*y +sin_rot*z;
            z_rot   = -sin_rot*y +cos_rot*z;
                
            return (x - r_c) * (x - r_c)  / (a*a) + y_rot*y_rot / (b*b) + z_rot*z_rot / (c*c) - 1;

  
        X_sc = transform.spherical_coordinate_to_cartesian([r_sc, clt_sc, lon_sc])


        args = (self.clt[-1], self.lon[-1],X_sc,self.r[-1],
                self.tilt[-1], self.a[-1],self.b[-1],self.c[-1]) 

        r_c = np.linspace(self.r[-1],r_sc, 1000)
        for i, r in enumerate(r_c):
            if ellipsoid_IP(r,*args)<=0: 
                break
    
        sol = optimize.root_scalar(ellipsoid_IP,args=args, bracket=[r_c[i-1], r_c[i]], 
                                   method='brentq',x1=1e-1,xtol=1e-27,rtol=8.89e-16)
      

        dt = (self.datetimes[-1] - self.datetimes[-2]).total_seconds()
        # time to travel from last fitted ellipsoid to arrival at earth
        T  = (arrival_datetime - self.datetimes[-1]).total_seconds()
        t  = np.arange(dt,T+dt, dt)
        # shock nose of last 2 fitted ellipsoids
        R   = self.r[-2:] + self.a[-2:]
        # shock nose speed of last fitted ellipsoid
        Vsh = (R[1] - R[0]) /dt 
        # speed of fitted ellipsoid centre
        V0  = Vsh / (1+self.a[-1]/self.r[-1])
        # distance between center last fitted ellipsoid and center of the ellipsoid that arrives at Earth
        Dr = sol.root-self.r[-1]
        # fast deceleration time during the first Tdec
        Tdec  = T
        # remainder has a slower deceleration
        alpha = 0.01
        DT     = T - Tdec 

        # A  = 2. * (Dr/(T*T) - V0/T)

        A = (Dr - V0*T) / (Tdec*(T-0.5*Tdec) + 0.5 * alpha * DT**2)

        r  = np.zeros_like(t)
        r[t<=Tdec] =  self.r[-1] + V0 * t[t<=Tdec] + 0.5 * A * t[t<=Tdec]*t[t<=Tdec]

        r[t>Tdec]  =  self.r[-1] + V0 * Tdec + 0.5 * A * Tdec*Tdec\
        + (V0 + A*Tdec)* ( t[t>Tdec]-Tdec) + 0.5 * alpha*A * ( t[t>Tdec]-Tdec)**2


        self.a    = np.concatenate((self.a, self.a[-1] * r / self.r[-1]))
        self.b    = np.concatenate((self.b, self.b[-1] * r / self.r[-1]))
        self.c    = np.concatenate((self.c, self.c[-1] * r / self.r[-1]))
        self.r    = np.concatenate((self.r, r))
        self.clt  = np.concatenate((self.clt,self.clt[-1]   * np.ones_like(r)))
        self.lon  = np.concatenate((self.lon,self.lon[-1]   * np.ones_like(r)))
        self.tilt = np.concatenate((self.tilt,self.tilt[-1] * np.ones_like(r)))
        self.datetimes = self.datetimes + [self.datetimes[-1] + timedelta(seconds=i) for i in t ]


    def _interpolate(self, arr,it, par):  
        return (1-par) * arr[it-1]  + par * arr[it]
            
            
            
        

        
                
                






