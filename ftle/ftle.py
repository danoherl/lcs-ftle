# Class to create FTLE field.
"""
Ideas from:
Shadden, Shawn & Lekien, Francois & Marsden, Jerrold. (2005).
            Definition and properties of Lagrangian coherent structures from
            finit-time Lyapunov exponents in two-dimensional aperiodic flows.
            Physica D. 212. 271-304. 10.1016/j.physd.2005.10.007.
"""

# Functions included:
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
        
        Returns:
            - T(float): Integration time (seconds)
        """
        T = (ds.time[self.time] - ds.time[0]).values.astype('timedelta64[s]').astype('float64')
        return T

    def get_ftle(self):
        """
        Calculates a 2D FTLE field. 
        """
        


