import numpy as np
import xarray as xr
import datetime
from os import listdir
from os.path import join, exists
import glob

def gcm_multiparams():
    # Sets out all the options for which piece of data to ingest
    gcm2institute = all_gcms_institutes()
    gcms = list(gcm2institute.keys())
    expts = ['control','free','nudged']
    inits = [datetime.datetime(2018,1,25),datetime.datetime(2018,2,8)]
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
    raw_data_dir = join('/badc/snap/data/post-cmip6/SNAPSI', gcm2institute[gcm], gcm, expt, date2label(init))
    print(f'{ensemble_dir = }')
    print(f'Ensemble_dir: \n{listdir(ensemble_dir)}')
    path_skeleton = join(raw_data_dir,'r*i*p*f*','6hrPt','tas','g*','v*','*.nc')
    raw_mem_files = glob.glob(path_skeleton)

    # 2. Spatiotemporal sub-selection and coarse-graining (cg)
    event_region = dict(lat=slice(50,65),lon=slice(-10,130))
    event_time_interval = [datetime.datetime(2018,2,21),datetime.datetime(2018,3,8)]
    reduced_data_dir = join('/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018/results_2024-05-04',gcm)
    # 2a. Coarse-graind in time (cgt): daily minima, daily mean, and maybe other stats (all members combined into one file)
    ens_file_cgt = join(reduced_data_dir, f't2m_init{i_init}_cgt.nc') # will hold daily min and daily mean 

    # 2b. Coarse-grained in time and space (cgts): furthermore take averages over grid boxes of increasing size 
    ens_file_cgts = join(reduced_data_dir, f't2m_init{i_init}_cgts.nc')
    
    # 3. Minimize over time 
    ens_file_cgts_mint = join(reduced_data_dir, f't2m_init{i_init}_cgts_mint.nc')

    workflow = (
            gcm,expt,init,
            event_region,event_time_interval,
            raw_mem_files,reduced_data_dir,
            ens_file_cgt,ens_file_cgts,ens_file_cgts_mint
            )
    return workflow

def coarse_grain_time(raw_mem_files, event_region, event_time_interval, ens_file_cgt):


def preprocess_gcm_6hrPt(dsmem,vbl,fcdate,timesel,spacesel):
    print(f'{spacesel = }')
    dsmem = (
            utils.rezero_lons(
                dsmem[vbl2key[vbl]]
                .assign_coords(time=np.arange(fcdate,fcdate+datetime.timedelta(hours=6*dsmem.time.size),datetime.timedelta(hours=6)))
                .sel(timesel))
            )
    print(f'{dsmem.lon = }')
    print(f'{dsmem.time = }')
    dsmem = (
            dsmem
            .sel(spacesel)
            .isel(time=slice(None,4*int(dsmem.time.size/4)))
            )
    print(f'{dsmem.time = }')
    dsmem = (
            dsmem
            .coarsen({'time': 4}, side='left', coord_func='min')
            .mean())
    return dsmem

if __name__ == "__main__":
    mem_files = gcm_workflow(4,0,0)
    print(f'{mem_files = }')
