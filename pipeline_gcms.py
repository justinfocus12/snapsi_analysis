import numpy as np
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
pltkwargs = dict({
    'bbox_inches': 'tight',
    'pad_inches': 0.2,
    })
import datetime
import sys
from os import listdir, makedirs
from os.path import join, exists, basename
import glob
from scipy.stats import norm as spnorm, genextreme as spgex

# My own modules
import utils

def gcm_multiparams():
    # Sets out all the options for which piece of data to ingest
    gcm2institute = all_gcms_institutes()
    gcms = list(gcm2institute.keys())
    expts = ['control','free','nudged']
    inits = ['20180125','20180208']
    return gcms, expts, inits

def analysis_multiparams():
    cgs_levels = [(1,1),(2,1),(4,1),(6,1),(35,8),(47,8),(141,16)][:-1]
    return cgs_levels


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
    print(f'{raw_mem_files[0] = }')
    # 2. Spatiotemporal sub-selection and coarse-graining (cg)
    event_region = dict(lat=slice(50,65),lon=slice(-10,130))
    event_time_interval = [datetime.datetime(2018,2,21),datetime.datetime(2018,3,8)]
    reduced_data_dir = join('/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018/results_2024-05-04',gcm)
    makedirs(reduced_data_dir,exist_ok=True)

    # spatial coarse graining (cgs)
    cgs_levels = analysis_multiparams()

    # Plotting dir 
    figdir = '/home/users/ju26596/snapsi_analysis_figures/feb2018/figures_2024-05-04'
    makedirs(figdir,exist_ok=True)
       
    workflow = (
            gcm,expt,init,
            event_region,event_time_interval,
            raw_mem_files,mem_labels,reduced_data_dir,figdir,
            cgs_levels,
            )
    return workflow

def coarse_grain_time(raw_mem_files, mem_labels, event_region, event_time_interval, use_dask=False):
    timesel = dict(time=slice(event_time_interval[0],event_time_interval[1]))
    preprocess = lambda dsmem: preprocess_gcm_6hrPt(dsmem, event_time_interval[0], timesel, event_region)
    print(f'{all([exists(f) for f in raw_mem_files]) = }')

    if use_dask:
        ds_ens = xr.open_mfdataset(raw_mem_files, preprocess=preprocess, parallel=False, combine='nested', concat_dim='member').assign_coords(member=mem_labels)
    else:
        ds_ens = []
        for f in raw_mem_files:
            ds_ens.append(preprocess(xr.open_dataset(f)))
            print(f'Appended file {f}')
        ds_ens = xr.concat(ds_ens, dim='member').assign_coords(member=mem_labels)
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

def coarse_grain_space(ds_cgt, cgs_level):
    data_vars = dict()
    Nlon,Nlat = (ds_cgt[dim].size for dim in ('lon','lat'))
    print(f'{ds_cgt.dims = }')
    print(f'{ds_cgt.shape = }')
    dim = {'lon': int(round(Nlon/cgs_level[0])), 'lat': int(round(Nlat/cgs_level[1]))}
    print(f'{dim = }')
    coslat = np.cos(np.deg2rad(ds_cgt['lat'])) * xr.ones_like(ds_cgt)
    coarsen_kwargs = dict(dim=dim, boundary='pad', coord_func={'lon': 'mean', 'lat': 'mean'})
    ds_cgts = (ds_cgt * coslat).coarsen(**coarsen_kwargs).sum() / coslat.coarsen(**coarsen_kwargs).sum()
    print(f'{ds_cgts.shape = }')
    return ds_cgts # awkward to put into a single dataset because of differing lon/lat coordinates between coarsening levels



def preprocess_gcm_6hrPt(dsmem,fcdate,timesel,spacesel):
    dsmem_tas = (
            utils.rezero_lons(
                dsmem['tas']
                .assign_coords(time=np.arange(fcdate,fcdate+datetime.timedelta(hours=6*dsmem.time.size),datetime.timedelta(hours=6)))
                .sel(timesel)
                )
            )
    dsmem_tas = (
            dsmem_tas
            .sel(spacesel)
            .isel(time=slice(0,4*int(dsmem_tas.time.size/4)))
            .expand_dims(member=[dsmem.attrs['variant_label']])
            )
    print(f'Preprocessing done; {dsmem_tas.time.size = }')
    return dsmem_tas

def compare_gev_maps(i_gcm):
    # Plot maps of GEV parameters at different locations (for the finely-grained maps)
    # Also plot probabilities as a function of threshold for the total-area average
    gcms,expts,inits = gcm_multiparams()
    cgs_levels = analysis_multiparams()
    for cgs_level in cgs_levels:
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        gevpar = []
        t2m_cgts_mint = []
        for i_expt,expt in enumerate(expts):
            gevpar_expt = []
            t2m_cgts_mint = []
            for i_init,init in enumerate(inits):
                gcm,expt,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,figdir,_ = gcm_workflow(i_gcm,i_expt,i_init)
                gevpar_file = join(reduced_data_dir,f'gevpar_e{expt}_i{init}_cgs{cgs_key}.nc')
                gevpar_expt_init = xr.open_dataarray(gevpar_file)
                gevpar_expt.append(gevpar_expt_init)
                # Plot maps of return period
                if min(cgs_level) > 1:
                    makedirs(figdir,exist_ok=True)
                    fig,axes = plt.subplots(nrows=3,ncols=1,figsize=(6,10))
                    for i_param,param in enumerate(['shape','loc','scale']):
                        ax = axes[i_param]
                        xr.plot.pcolormesh(gevpar_expt_init.sel(param=param,daily_stat='daily_mean'),x='lon',y='lat',cmap='coolwarm',ax=ax)
                        ax.set_xlabel('Longitude')
                        ax.set_ylabel('Latitude')
                        ax.set_title(param)
                    fig.suptitle(r'%s, init %s'%(expt,init))
                    fig.savefig(join(figdir,f'plot_gevpar_e{expt}_i{init}_cgs{cgs_key}.png'), **pltkwargs)
                    plt.close(fig)
                # 
                t2m_cgts_file = join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day_cgs{cgs_key}.nc')
                t2m_cgts_mint.append(xr.open_dataarray(t2m_cgts_file).min(dim='time'))
            gevpar_expt = xr.concat(gevpar_expt, dim='init').assign_coords(init=inits)
            gevpar.append(gevpar_expt)
        gevpar = xr.concat(gevpar, dim='expt').assign_coords(expt=expts)
    return

def reduce_gcm(i_gcm,i_expt,i_init):
    # One GCM, one forcing (expt), one initialization (init), multiple coarse-grainings in space 
    tododict = dict({
        'coarse_grain_time':           1,
        'coarse_grain_space':          1,
        'fit_gev':                     1,
        })
    gcm,expt,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,figdir,cgs_levels = gcm_workflow(i_gcm,i_expt,i_init)

    # ------------- Coarse-grain in time (cgt) and space (cgts) -------------
    ens_file_cgt = join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day.nc')
    if tododict['coarse_grain_time']:
        ds_cgt = coarse_grain_time(raw_mem_files, mem_labels, event_region, event_time_interval)
        ds_cgt.to_netcdf(ens_file_cgt)
    else:
        ds_cgt = xr.open_dataarray(ens_file_cgt)

    for cgs_level in cgs_levels:
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day_cgs{cgs_key}.nc')
        if tododict['coarse_grain_space']:
            ds_cgts = coarse_grain_space(ds_cgt, cgs_level)
            ds_cgts.to_netcdf(ens_file_cgts)
        else:
            ds_cgts = xr.open_dataarray(ens_file_cgts)
        # ---- Minimize in time -----------------------------
        ds_cgts_mint = ds_cgts.min(dim='time')
        # ----------- Perform GEV fitting (on negative temperature) --------------
        gev_param_file = join(reduced_data_dir,f'gevpar_e{expt}_i{init}_cgs{cgs_key}.nc')
        if tododict['fit_gev']:
            memdim = ds_cgts_mint.dims.index('member')
            print(f'{memdim = }')
            gevpar_array = np.apply_along_axis(spgex.fit, memdim, -ds_cgts_mint.to_numpy())
            gevpar_dims = list(ds_cgts_mint.dims).copy()
            gevpar_dims[memdim] = 'param'
            gevpar_coords = dict(ds_cgts_mint.coords).copy()
            gevpar_coords.pop('member')
            gevpar_coords['param'] = ['shape','loc','scale']
            gevpar = xr.DataArray(
                    coords=gevpar_coords,
                    dims=gevpar_dims,
                    data=gevpar_array)
            gevpar.loc[dict(param='shape')] *= -1
            gevpar.to_netcdf(gev_param_file)
        else:
            gevpar = xr.open_dataarray(gev_param_file)
        ds_cgts.close()
    ds_cgt.close()
    return 

if __name__ == "__main__":
    idx_gcm = [4]
    idx_expt = [0,1,2]
    idx_init = [0,1]
    procedure = sys.argv[1]
    if procedure == 'reduce':
        for i_gcm in idx_gcm:
            for i_expt in idx_expt:
                for i_init in idx_init:
                    reduce_gcm(i_gcm,i_expt,i_init)
    elif procedure == 'gevmap':
        for i_gcm in idx_gcm:
            compare_gev_maps(i_gcm)
