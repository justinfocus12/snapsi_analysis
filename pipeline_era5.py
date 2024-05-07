# The pipeline for calculating the hazard functions for ERA5 data

import numpy as np
import xarray as xr 
from cartopy import crs as ccrs
import netCDF4
from matplotlib import pyplot as plt, rcParams, ticker
pltkwargs = dict({
    'bbox_inches': 'tight',
    'pad_inches': 0.2,
    })
rcParams.update({
    'font.family': 'monospace',
    'font.size': 15,
    })
pltkwargs = {"bbox_inches": "tight", "pad_inches": 0.2}
import datetime
import sys
from os import listdir, makedirs
from os.path import join, exists, basename
import glob
from scipy.stats import norm as spnorm, genextreme as spgex

# My own modules
import utils

def analysis_multiparams():
    # lon/lat ratios are 6/1 at the bottom and 4/1 at the top; stick to 5/1 
    cgs_levels = [(1,1),(5,1),(10,2),(20,4),(40,8),(141,16)]
    return cgs_levels

def era5_workflow(verbose=False):
    raw_data_dir = '/gws/nopw/j04/snapsi/processed/wg2/ju26596/era5'
    years = np.arange(1979,2019,dtype=int)
    year_filegroups = []
    event_time_intervals = []
    event_duration = datetime.datetime(2018,3,8) - datetime.datetime(2018,2,21)
    for year in years:
        year_filegroups.append(tuple(
            join(raw_data_dir,f't2m_nhem_{year:04}-{month:02}.nc')
            for month in [1,2,3]
            ))
        event_time_intervals.append((datetime.datetime(year,2,21),datetime.datetime(year,2,21) + event_duration))

    event_region = dict(lat=slice(50,65),lon=slice(-10,130))

    reduced_data_dir = join('/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018/results_2024-05-04',gcm)
    makedirs(reduced_data_dir,exist_ok=True)

    # spatial coarse graining (cgs)
    cgs_levels = analysis_multiparams()


    figdir = f'/home/users/ju26596/snapsi_analysis_figures/feb2018/figures_2024-05-04/era5'
    makedirs(figdir,exist_ok=True)

    workflow = (
            years,event_region,event_time_intervals,
            year_filegroups,reduced-data_dir,figdir,
            cgs_levels,
            )
    return workflow

def coarse_grain_time(year_filegroups, event_region, event_time_intervals):
    ds_ens = []
    for yfg in year_filegroups:
        # TODO






