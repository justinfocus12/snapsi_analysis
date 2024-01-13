# Convenience functions to find and process data
import numpy as np
import pandas as pd
import datetime
import xarray as xr
import dask
from os.path import join,exists,basename
import glob
import yaml

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

def date2label(date):
    return r"s%s"%(date.strftime("%Y%m%d"))

def area_average(da):
    coslat = np.cos(np.deg2rad(da["lat"]))
    print(f'computed coslat')
    aa = (da * coslat).sum(dim=["lat","lon"]) / (coslat * da.lon.size).sum().item()
    print(f'did areal average')
    da_finite_fraction = np.isfinite(da).mean(dim=["lat","lon"])
    aa = xr.where(da_finite_fraction>0.5, aa, np.nan)
    return aa

# ------------ Constant parameters for file system ------------

model2institute = dict({
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

vbl2key = dict({
    "t2m": "tas",
    })

base_dirs = dict({
    'era5': '/gws/nopw/j04/snapsi/processed/wg2/era5',
    'gcms': '/badc/snap/data/post-cmip6/SNAPSI'
    })

def get_dirinfo():
    return model2institute,vbl2key,base_dirs



def get_ensemble_member_filenames(model, vbl, expt, verbose=False):
    ensemble_dir = join(base_dirs['gcms'], model2institute[model], model, expt, date2label(sdate))
    vbl_key = dirinfo["models"]["vbl_keys"][vbl]
    path_skeleton = join(ensemble_dir,"r*i*p1f1","6hr",vbl_key,"g*","v20*","*.nc")
    if verbose:
        print(f"{path_skeleton = }")
    mem_files = glob.glob(path_skeleton)
    return mem_files

def get_era5_filename(vbl):
    if vbl == "t2m":
        fname = join(base_dirs['era5'],'2t_1979_2020_D_regrid.nc') 
    else:
        raise Exception(f"{vbl} not indexed for ERA5 retrieval")
    return fname

def get_clim_filename(vbl):
    if vbl == "t2m":
        fname = join(base_dirs['era5'],'2t_daily_clim_1deg.nc') 
    else:
        raise Exception(f"{vbl} not indexed for ERA5 retrieval")
    return fname

def get_gcm_6hrPt_filenames(model, vbl, expt, forecast_date):
    path_skeleton = join(base_dirs['gcms'],model2institute[model],model,expt,date2label(forecast_date))
    path_skeleton = join(path_skeleton,"r*i*p*f*")
    mem_labels = [basename(p) for p in glob.glob(path_skeleton)]
    if vbl in vbl2key.keys():
        path_skeleton = join(path_skeleton,"6hrPt",vbl2key[vbl])
    else:
        raise Exception("Variable {vbl} not accounted for")
    path_skeleton = join(path_skeleton,'g*','v*','*.nc')
    file_list = glob.glob(path_skeleton)
    return glob.glob(path_skeleton),mem_labels

def count_ensemble_sizes():
    counts = dict()
    for model in list(model2institute.keys()):
        counts[model] = dict()
        print(f"\n\n{model = }")
        for expt in ["control","free","nudged"]:
            counts[model][expt] = dict()
            print(f"\t{expt = }")
            for fcdate in [datetime.datetime(year=2018,month=1,day=25)]:
                print(f"\t\t{fcdate = }")
                fnames = get_t2m_6hrPt_filenames(model,"t2m",expt,fcdate)
                print(f"\t\t\t{len(fnames) = }")
                counts[model][expt][fcdate] = len(fnames)
    return counts 

def preprocess_gcm_6hrPt(dsmem,vbl,fcdate,timesel,spacesel):
    dsmem = (
            rezero_lons(
                dsmem[vbl2key[vbl]]
                #.assign_coords(time = pd.to_datetime(dsmem.indexes['time'].to_datetimeindex()))
                #.assign_coords(time = sdate + (dsmem.time.to_numpy() - dsmem.time[0].item()))
                .assign_coords(time=np.arange(fcdate,fcdate+datetime.timedelta(hours=6*dsmem.time.size),datetime.timedelta(hours=6)))
                .sel(timesel))
            .sel(spacesel))
    return dsmem






if __name__ == "__main__":
    count_ensemble_sizes()
