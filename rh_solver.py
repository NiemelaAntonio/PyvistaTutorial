import numpy as np
import numba
from numba.experimental import jitclass
from vector_calculus import dot_product, cross_product

spec = [("rg_sh_calculated", numba.boolean),
        ("rb_sh_calculated", numba.boolean),
        ("rp_sh_calculated", numba.boolean),
        ("gamma", numba.float64),
        ("V_shock", numba.float64),
        ("theta_u", numba.float64),
        ("cos_theta_u", numba.float64),
        ("Va_u", numba.float64),
        ("Vs_u", numba.float64),
        ("vn_u", numba.float64),
        ("Vn_u", numba.float64),
        ("plasma_beta_u", numba.float64),
        ("rg_sh_value", numba.float64),
        ("rb_sh_value", numba.float64),
        ("rp_sh_value", numba.float64),
        ("cross_shock_potential_coef", numba.float64)
        ]

@jitclass(spec)
class RH_Solver():
    #
    # Rankine Hugoniot solver for a moving shock
    # It is assumed that the shock velocity is along the shock normal
    # Based on Eq. 13 in Vainio & Schlickeiser 1999

    def __init__(self):
        self.gamma   = 5./3. # polytropic index gamma 

    def initialize(self, V_sh, theta_u, vn_u, Va_u, plasma_beta_u, p_u, rho_u):

        self.V_shock     = V_sh
        self.theta_u     = theta_u
        self.cos_theta_u = np.cos(self.theta_u)
        
        # Upstream alfven speed
        self.Va_u = Va_u
        # Vn_u is the upstream normal velocity in the inertial frame.
        self.Vn_u = vn_u 
        # vn_u is the upstream normal velocity in a shock-rest frame.
        # note that the assumed normal points toward the downstream
        self.vn_u = (self.V_shock - self.Vn_u) 

        # upstream speed of sound
        self.Vs_u = np.sqrt(self.gamma * p_u / rho_u)  

        self.plasma_beta_u = plasma_beta_u

        self.cross_shock_potential_coef= 0
        self.rg_sh_value  = 0 
        self.rb_sh_value = 0

        self.rg_sh_calculated = False
        self.rb_sh_calculated = False
        self.rp_sh_calculated = False

    @property
    def csp_coef(self):
    # Provide cross-shock potential coefficient

        return  self.cross_shock_potential_coef


    #@property
    #def V_HT(self,n,V,B):

    #   v = self.V_shock * n - V 
    #  return -cross_product(n,cross_product(B,v)) / (dot_product(n,B))   


    @property
    def Vd_n(self):

        # u refers to the speed in the HT frame. 
        # However, u_1n = v_1n and u_2n = v_2n
        #u_1n = self.vn_u
        un_d = self.vn_u / self.rg_sh

        return self.V_shock - un_d

    @property
    def M_A(self):
    # Compute Alfvenic Mach number along magnetic field line
    # u_u is the upstream speed in the HT frame 
    # M_A = un_u / Van_u = u_u / Va_u

        u_u = self.vn_u/self.cos_theta_u
        M_A = u_u/self.Va_u
 
        return  M_A

    @property
    def M_fms(self):
    # Compute Fast Magnetosonic Mach number (NOT in HT frame)

        Vsqr = self.Va_u**2+self.Vs_u**2

        V_fms = np.sqrt(0.5 * (Vsqr + np.sqrt(Vsqr**2\
            -(2*self.Va_u*self.Vs_u*np.cos(self.theta_u))**2)))

        return   self.vn_u/V_fms

    @property
    def rg_sh(self):

        # Wrapping function to compute gas
        # compression ratio   

        if (not self.rg_sh_calculated):
            self.rg_sh_value      = self.rg_sh_calc()
            self.rg_sh_calculated = True

        return self.rg_sh_value

    @property
    def rb_sh(self):

        # Wrapping function to compute magnetic
        # compression ratio      

        if ( not self.rb_sh_calculated):
           self.rb_sh_value      = self.rb_sh_calc()
           self.rb_sh_calculated = True

        return self.rb_sh_value 

    @property
    def rp_sh(self):

        # Wrapping function to compute thermal pressure
        # compression ratio      

        if ( not self.rp_sh_calculated):
           self.rp_sh_value      = self.rp_sh_calc()
           self.rp_sh_calculated = True

        return self.rp_sh_value 

    
    def alternative_init_guess(self, half_gb):

            # Compute an alternative initial guess value for
            # the gas compresion solver.

            cos_theta_sqr = self.cos_theta_u**2
            sin_theta_sqr = 1.0 - cos_theta_sqr

            z = (1.0 + half_gb + np.sqrt( (1.0 + half_gb)**2 \
                - 4 * half_gb * cos_theta_sqr) )/(2 * cos_theta_sqr) - 1.0
            return z

    def func_vals(self,z, g, half_gb):


        # Compute value of the function f on LHS of 
        # the equation f(z) = 0 solved by Newton's 
        # method, and its derivative at argument z.

        cos_theta_sqr = self.cos_theta_u*self.cos_theta_u
        sin_theta_sqr = 1.0 - cos_theta_sqr

        a1 = z*z * (g + 1.0) * cos_theta_sqr \
                + (1.0 - g * z) * sin_theta_sqr
        a1_prime = 2 * z * (g + 1.0) * cos_theta_sqr - g * sin_theta_sqr

        A       = (1 + z) * a1 - 2 * half_gb * z*z
        A_prime = a1 + (1.0 + z) * a1_prime - 4 * half_gb * z

        B = z*z * (g - 1.0) * cos_theta_sqr \
            + (1.0 + (2.0 - g) * z) * sin_theta_sqr
        B_prime = 2 * z * (g - 1.0) * cos_theta_sqr \
                  + (2.0 - g) * sin_theta_sqr

        f = A/B - self.M_A**2
        f_prime = (A_prime * B - A * B_prime)/(B*B)

        return f, f_prime

    def rg_sh_calc(self):

        #Solves gas compression ratio of the shock from
        #the cubic equation using Newton's method for two
        #two different initial guesses.

        dr_max  = 1.e-4
        half_gb = 0.5 * self.gamma * self.plasma_beta_u

        # Initial guess 1
        z_init_1 = self.M_A**2 - 1.0

        # Initial guess 2
        z_init_2 = self.alternative_init_guess(half_gb)

        for i in [1, 2]:
            z = z_init_1 if i==1 else z_init_2

            r_old = 4.0
            dr_abs = 1.0
            n_step = 0

            # Newton's method  
            while (dr_abs > dr_max):
                r = self.M_A**2/(1.0 + z)
                dr_abs = abs(r - r_old)
                r_old = r
                f, f_prime = self.func_vals(z,self.gamma, half_gb)
                n_step = n_step + 1
                if (n_step > 100):
                    raise  RuntimeError("Calculation of gas compression ratio: Number of iteration exceeded 100.")

                z = z - f/f_prime

            if (i == 1):
               r_1 = r
            else:
               r_2 = r

        if (abs(r_1 - r_2) < dr_max):
            r = r_1
        else:
            raise RuntimeError("Error in calculation of gas compression ratio.")
        return r

    def rb_sh_calc(self):

    # Compute magnetic compression ratio of the shock.

        cos_theta_sqr = self.cos_theta_u*self.cos_theta_u
        sin_theta_sqr = 1.0 - cos_theta_sqr
        M_A_sqr       = self.M_A*self.M_A

        rb = np.sqrt(cos_theta_sqr + sin_theta_sqr \
            * ((M_A_sqr - 1.0)/(M_A_sqr - self.rg_sh) * self.rg_sh)**2)

        return rb

    def rp_sh_calc(self):

    # Compute pressure compression ratio of the shock.

        r         = self.rg_sh
        Van_u_sqr = self.cos_theta_u * self.Va_u
        vn_u_sqr  = self.vn_u**2

        rp = 1 + self.gamma * vn_u_sqr * (r-1) / (self.Vs_u**2 * r)\
               *( 1 -  r * self.Va_u**2 * ((r+1)*vn_u_sqr - 2 * r * Van_u_sqr )\
                                         /(2*(vn_u_sqr - r * Van_u_sqr)**2))


        return rp

