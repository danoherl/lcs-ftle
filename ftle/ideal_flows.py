import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

def vortex_quiver():
     """
     An ideal, steady vortex, without using xarray. Designed to be easily
     reproduced with a quiver plot in a Jupyter Notebook.
     """
     x = np.arange(-5,5,0.2)
     y = np.arange(-5,5,0.2)
     nx = len(x)
     ny = len(y)
     xx, yy = np.meshgrid(x, y)
     u, v = np.zeros((nx,ny)), np.zeros((nx,ny))
     for i in range(nx):
          for j in range(ny):
               u[i,j] = yy[i,j]
               v[i,j] = -xx[i,j]
    
     return x, y, u, v

               


def vortex(dlon, dlat, nt, lon_min, lon_max, lat_min, lat_max, alpha=1 ):
    """
    Idealized steady vortex flow. The flow will be (u,v) = (y, -x),
    scaled by alpha.
    Coordinates: (lon, lat)
    """
    lon_grid = np.arange(lon_min,lon_max,dlon)
    lat_grid = np.arange(lat_min,lat_max,dlat)
    nlon = lons.shape[0]
    nlat = lats.shape[0]
    velocity_shape = (nlon, nlat, nt)
    u = np.zeros(velocity_shape)
    v = np.zeros(velocity_shape)
    # Dictionary for coordinates
    
    for t in range(nt):
            for y in range(nlat):
                for x in range(nlon):
                    u[x,y,t] = y
                    v[x,y,t] = -x
    # Convert to xarray DataArray, with coords to correspond with dims
    coords = {
        'lon':lons,
        'lat':lats,
        'time': pd.date_range('2025-01-01',freq = '3H',periods=nt)
    }
    dims = list(coords.keys())
    u = xr.DataArray(u, dims=dims,coords=coords)
    v = xr.DataArray(v,dims=dims,coords=coords)
    return u,v