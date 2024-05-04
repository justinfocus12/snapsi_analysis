import numpy as np
import xarray as xr
import datetime
import sys
from os import listdir, makedirs
from os.path import join, exists, basename
import glob

# My own modules
import utils

def gcm_multiparams():
    # Sets out all the options for which piece of data to ingest
    gcm2institute = all_gcms_institutes()
    gcms = list(gcm2institute.keys())
    expts = ['control','free','nudged']
    inits = ['20180125','20180208']
    return gcms, expts, inits

def all_gcms_institutes():
    gcm2institute = dict({
        'BCC-CSM2-HR': 'BCC',
        'GLOBO': 'CNR-ISAC',
        'GEM-NEMO': 'ECCC',
        'CanESM5': 'ECCC',
        'IFS': 'ECMWF',
        'SPEAR': 'NOAA-GFDL',
        'GRIMs': 'SNU',
        'GloSea6-GC32': 'KMA',
        'CNRM-CM61': 'Meteo-France',
        'CESM2-CAM6': 'NCAR',
        'NAVGEM': 'NRL',
        'GloSea6': 'UKMO',
        })
    return gcm2institute

def date2label(date):
    return r"s%s"%(date.strftime("%Y%m%d"))

def gcm_workflow(i_gcm, i_expt, i_init, verbose=False):
    # Sets out the folders necessary to ingest a chunk of data specified by the input arguments 
    gcms,expts,inits = gcm_multiparams()
    gcm = gcms[i_gcm]
    expt = expts[i_expt]
    init = inits[i_init]

    gcm2institute = all_gcms_institutes()

    # ----------- Files for each stage of analysis -------------
    # 1. Raw data
    raw_data_dir = join('/badc/snap/data/post-cmip6/SNAPSI', gcm2institute[gcm], gcm, expt, 's'+init)
    print(f'{raw_data_dir = }')
    print(f'Ensemble_dir: \n{listdir(raw_data_dir)}')
    ens_path_skeleton = join(raw_data_dir,'r*i*p*f*')
    mem_labels = [basename(p) for p in glob.glob(ens_path_skeleton)]
    print(f'{mem_labels = }')

    path_skeleton = join(ens_path_skeleton,'6hrPt','tas','g*','v*','*.nc')
    raw_mem_files = glob.glob(path_skeleton)
    # 2. Spatiotemporal sub-selection and coarse-graining (cg)
    event_region = dict(lat=slice(50,65),lon=slice(-10,130))
    event_time_interval = [datetime.datetime(2018,2,21),datetime.datetime(2018,3,8)]
    reduced_data_dir = join('/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018/results_2024-05-04',gcm)
    makedirs(reduced_data_dir,exist_ok=True)
    # 2a. Coarse-grained in time (cgt): daily minima, daily mean, and maybe other stats (all members combined into one file)
    ens_file_cgt = join(reduced_data_dir, f't2m_e{expt}_i{init}_cgt.nc') # will hold daily min and daily mean 

    # 2b. Coarse-grained in time and space (cgts): furthermore take averages over grid boxes of increasing size. List of pairs (a,b) = number of subdivisions in (lon,lat) directions  
    # The region's longitudinal extent is approximately 6 and 4 times its latitudinal extent, as measured at the lower and upper latitudinal boundaries respectively. 
    cgs_levels = [(1,1),(2,1),(4,1),(6,1),(35,8),(47,8)]
    ens_file_cgts = join(reduced_data_dir, f't2m_e{expt}_i{init}_cgts.nc')
    
    # 3. Minimize over time 
    ens_file_cgts_mint = join(reduced_data_dir, f't2m_e{expt}_i{init}_cgts_mint.nc')

    workflow = (
            gcm,expt,init,
            event_region,event_time_interval,
            raw_mem_files,mem_labels,reduced_data_dir,
            ens_file_cgt,
            ens_file_cgts,cgs_levels,
            ens_file_cgts_mint
            )
    return workflow

def coarse_grain_time(raw_mem_files, mem_labels, event_region, event_time_interval):
    timesel = dict(time=slice(event_time_interval[0],event_time_interval[1]))
    preprocess = lambda dsmem: preprocess_gcm_6hrPt(dsmem, event_time_interval[0], timesel, event_region)
    print(f'{all([exists(f) for f in raw_mem_files]) = }')
    ds_ens = xr.open_mfdataset(raw_mem_files, preprocess=preprocess, parallel=True, combine='nested', concat_dim='member').assign_coords(member=mem_labels)
    print(f'{ds_ens.coords = }')
    # Take daily mean 
    daily_mean = (
            ds_ens
            .coarsen({'time': 4}, side='left', coord_func='min')
            .mean()
            ).compute()
    daily_min = (
            ds_ens
            .coarsen({'time': 4}, side='left', coord_func='min')
            .min()
            ).compute()
    ds_ens.close()
    ds_ens_cgt = xr.concat([daily_mean,daily_min], dim='daily_stat').assign_coords(daily_stat=['daily_mean','daily_min'])
    return ds_ens_cgt

def coarse_grain_space(ds_ens_cgt, cgs_levels):
    data_vars = dict()
    Nlon,Nlat = (ds_ens_cgt[dim].size for dim in ('lon','lat'))
    for cgs_level in cgs_levels:
        level_key = r'%dx%d'%(cgs_level[0],cgs_level[1]) # denotes the level of coarsening
        data_vars[level_key] = ds_ens_cgt.coarsen({'lon': int(round(Nlon/cgs_level[0])), 'lat': int(round(Nlat/cgs_level[1]))}, boundary='pad', coord_func={'lon': 'mean', 'lat': 'mean'}).mean()
    ds_ens_cgts = xr.Dataset(data_vars=data_vars)
    return ds_ens_cgts



def preprocess_gcm_6hrPt(dsmem,fcdate,timesel,spacesel):
    dsmem = (
            utils.rezero_lons(
                dsmem['tas']
                .assign_coords(time=np.arange(fcdate,fcdate+datetime.timedelta(hours=6*dsmem.time.size),datetime.timedelta(hours=6)))
                .sel(timesel))
            )
    dsmem = (
            dsmem
            .sel(spacesel)
            .isel(time=slice(None,4*int(dsmem.time.size/4)))
            )
    print(f'Preprocessing done')
    return dsmem 

def gcm_procedure():
    tododict = dict({
        'coarse_grain_time':           0,
        'coarse_grain_space':          1,
        })
    gcm,expt,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,ens_file_cgt,ens_file_cgts,cgs_levels,ens_file_cgts_mint = gcm_workflow(4,0,0)
    if tododict['coarse_grain_time']:
        ds_ens_cgt = coarse_grain_time(raw_mem_files, mem_labels, event_region, event_time_interval)
        ds_ens_cgt.to_netcdf(ens_file_cgt)
    else:
        ds_ens_cgt = xr.open_dataarray(ens_file_cgt)
    if tododict['coarse_grain_space']:
        ds_ens_cgts = coarse_grain_space(ds_ens_cgt, cgs_levels)
        ds_ens_cgts.to_netcdf(ens_file_cgts)
    else:
        ds_ens_cgts = xr.open_dataset(ens_file_cgts)

    ds_ens_cgt.close()
    ds_ens_cgts.close()
    return 

if __name__ == "__main__":
    gcm_procedure()
