"""
The purpose of this module is to do standard FTLE field calculations using 
only numpy code, with the purpose of using numba decorators to then compile
to machine code.
"""



import numpy as np
from numba import njit, float32, prange


        





@njit 
def compute_flowmap_gradient(phi, dx, dy, i, j):
    """
    Flow derivative, as defined in Brunton (2009). 
    Using numpy arrays only to allow use of numba.
    Just a stencil, loops occur in main function. 
    
    Args:
        - phi (np.ndarray): Flowmap, shape = (nx,ny,2), phi = (f, g)

    dx : float
    distance between grid points, x direction
    dy: float
    distance between grid points, y direction
    i: int
    """
    dfdx = (phi[i + 1, j, 0] - phi[i - 1, j, 0]) / (2 * dx)
    dfdy = (phi[i, j + 1, 0] - phi[i, j - 1, 0]) / (2 * dy)
    dgdx = (phi[i + 1, j, 1] - phi[i - 1, j, 1]) / (2 * dx)
    dgdy = (phi[i, j + 1, 1] - phi[i, j - 1, 1]) / (2 * dy)
    
    return dfdx, dfdy, dgdx, dgdy

@njit
def max_evalue(D):
    """
    Computes the max eigenvalue of the Cauchy-Green deformation tensor.
    As this is (real) symmetric, we use a Hermitian eigenvalue calculation.
    Args:
        - D (np.ndarray): CG Deformation Tensor, calculated as the flowmap (dphi)^T*(dphi)

    """
    tr = A[0,0] + A[1,1]
    discrim = np.sqrt((A[0,0] - A[1,1])**2 + 4 * (A[1,0])**2)
    return 0.5 * (tr + discrim)

@njit
def transpose_product(a,b,c,d):
    """
    Calculates (A^T)A for given 2x2 real matrix A = [[a,b],[c,d]].
    """
    C = np.zeros((2,2))
    C[0] = a**2 + c**2, a*b + c*d
    C[1] = a*b + c*d, b**2 + d**2
    return np.array(C) 


@njit(parallel=True)
def ftle_2D_interior(phi, dx, dy, T):
    """
    Calculate FTLE resulting field using the eigenvalues and derivative stencil.
    Boundaries are not included.
    """
    nx,ny = phi.shape[0], phi.shape[1]
    ftle = np.zeros((nx, ny), float32)
    for i in prange(1, nx - 1): 
        for j in range(1, ny - 1):
            dxdx, dxdy, dydx, dydy = compute_flowmap_gradient(phi, dx, dy, i, j)
            C = transpose_product(dxdx, dxdy, dydx, dydy)
            max_evalue = max_evalue(C)
            ftle[i,j] = np.log(max_evalue) /  np.abs(T)
        
    return ftle




        
