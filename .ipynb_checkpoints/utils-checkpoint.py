# Commonly used functions for spatial analysis
import numpy as np
import xarray as xr

def area_average(da):
    coslat = np.cos(np.deg2rad(da["lat"]))
    print(f'computed coslat')
    aa = (da * coslat).sum(dim=["lat","lon"]) / (coslat * da.lon.size).sum().item()
    print(f'did areal average')
    da_finite_fraction = np.isfinite(da).mean(dim=["lat","lon"])
    aa = xr.where(da_finite_fraction>0.5, aa, np.nan)
    return aa

def spherical2cartesian(lon,lat):
    vec = np.array([
        np.cos(np.deg2rad(lon))*np.cos(np.deg2rad(lat)),
        np.sin(np.deg2rad(lon))*np.cos(np.deg2rad(lat)),
        np.sin(np.deg2rad(lat)),
        ])
    return vec

def great_circle_distance(lon0,lat0,lon1,lat1):
    vec0 = spherical2cartesian(lon0,lat0)
    vec1 = spherical2cartesian(lon1,lat1)
    return np.sqrt(2 * (1 - np.sum(vec0*vec1)))


def rezero_lons(ds,lonmax=180):
    # Roll coordinates to run from [-180,180) about a given center
    lons_geq_lonmax = np.where(ds.lon.values >= lonmax)[0]
    if len(lons_geq_lonmax) > 0:
        lonroll = lons_geq_lonmax[0]
    else:
        lonroll = 0
    ds_rolled = (
            ds
            .assign_coords(lon=(ds.lon.values - 360*(ds.lon >= lonmax)))
            .roll(lon=lonroll,roll_coords=True))
    return ds_rolled
