"""
Ideas from:
Shadden, Shawn & Lekien, Francois & Marsden, Jerrold. (2005).
            Definition and properties of Lagrangian coherent structures from
            finit-time Lyapunov exponents in two-dimensional aperiodic flows.
            Physica D. 212. 271-304. 10.1016/j.physd.2005.10.007.
"""
import numpy as np
import xarray as xr


class FTLE:
    """
    Class to undertake Finite Time Lyapunov Exponent analysis.
    """
    def __init__(self, t_0, time):
        """
        Initialise setup, in order to extract FTLE.
        Args:
            - t_0 (int): The initial time of the simulation.
            - time (int): T (once converted into seconds), where [t0, t0 + T]
                is the domain of integration. 
        """
        self.t_0 = t_0
        self.time = time 
        


    def time_to_T(self):
        """
        Converts integration time to seconds.
        Also obtains timeindex, from which to index x and y values for Jacobian.
        
        Returns:
            - T(float): Integration time (seconds)
        """
        T = (ds.time[self.time] - ds.time[0]).dt.total_seconds().to_numpy()
        return T

    def get_ftle(self):
        """
        Calculates a 2D FTLE field in cartesian coordinates (x,y)
        """
        t_idx = self.time
        T = self.time_to_T()
        x_T = ds.x.isel(time=t_idx).squeeze().to_numpy()
        y_T = ds.y.isel(time=t_idx).squeeze().to_numpy()
        dy = ds.y0.to_numpy()
        dx = ds.x0.to_numpy()
        dxdy, dxdx = np.gradient(x_T, dy, dx)        
        dydy, dydx = np.gradient(y_T, dy, dx)
        ny, nx = dxdy.shape
        ftle = xr.zeros_like(dxdy)
        for i in range(0, ny):
            for j in range(0, nx):
                J = np.array([[dxdx[i, j], dxdy[i, j]],
                              [dydx[i, j], dydy[i, j]]])
                M = np.dot(np.transpose(J), J)
                evalues_lya, _ = np.linalg.eigvalsh(M)
                ftle[i, j] = (1./np.abs(T))*np.log(np.sqrt(evalues_lya.max()))
        return ftle

    
        
