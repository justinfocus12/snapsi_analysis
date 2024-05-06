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

def gcm_multiparams():
    # Sets out all the options for which piece of data to ingest
    gcm2institute = all_gcms_institutes()
    gcms = list(gcm2institute.keys())
    expts = ['control','free','nudged']
    inits = ['20180125','20180208']
    return gcms, expts, inits

def analysis_multiparams():
    # lon/lat ratios are 6/1 at the bottom and 4/1 at the top; stick to 5/1 
    cgs_levels = [(1,1),(5,1),(10,2),(20,4),(40,8),(141,16)]
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
    ens_path_skeleton = join(raw_data_dir,'r*i*p*f*')
    mem_labels = [basename(p) for p in glob.glob(ens_path_skeleton)]

    path_skeleton = join(ens_path_skeleton,'6hrPt','tas','g*','v*','*.nc')
    raw_mem_files = glob.glob(path_skeleton)
    # 2. Spatiotemporal sub-selection and coarse-graining (cg)
    event_region = dict(lat=slice(50,65),lon=slice(-10,130))
    event_time_interval = [datetime.datetime(2018,2,21),datetime.datetime(2018,3,8)]
    reduced_data_dir = join('/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018/results_2024-05-04',gcm)
    makedirs(reduced_data_dir,exist_ok=True)

    # spatial coarse graining (cgs)
    cgs_levels = analysis_multiparams()

    # Plotting dir 
    figdir = f'/home/users/ju26596/snapsi_analysis_figures/feb2018/figures_2024-05-04/{gcm}'
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


def plot_t2m_map(ds):
    # TODO mirror the GEV plots so they give similar information 
    fig,axes = plt.subplots(figsize=(20,5),nrows=2,ncols=2,subplot_kw={'projection': ccrs.PlateCarree()},gridspec_kw={'hspace': 0.02, 'wspace': 0.004})
    pcmargs = dict(x='lon',y='lat',cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree(),cbar_kwargs={'orientation': 'horizontal', 'label': '', 'shrink': 0.5, 'pad': 0.14, 'aspect': 40})
    Tmin = ds.min().item()
    Tmax = ds.max().item()
    # ensemble mean of time mean; ensemble mean of time min; ensemble std of time mean; ensemble std of time min
    ax = axes[0,0]
    xr.plot.pcolormesh(ds.mean('time').mean('member'), ax=ax, **pcmargs, vmin=Tmin, vmax=Tmax)
    ax.coastlines()
    ax.set_title(r'Time mean, ens. mean')
    ax.set_xlabel('')
    ax.set_ylabel('Lat')
    ax = axes[0,1]
    xr.plot.pcolormesh(ds.min('time').mean('member'), ax=ax, **pcmargs, vmin=Tmin, vmax=Tmax)
    ax.coastlines()
    ax.set_title(r'Time min, ens. mean')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax = axes[1,0]
    xr.plot.pcolormesh(ds.mean('time').std('member'), ax=ax, **pcmargs)
    ax.coastlines()
    ax.set_title(r'Time mean, ens. std.')
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    ax = axes[1,1]
    xr.plot.pcolormesh(ds.min('time').std('member'), ax=ax, **pcmargs)
    ax.coastlines()
    ax.set_title(r'Time min, ens. std.')
    ax.set_xlabel('Lon')
    ax.set_ylabel('')
    return fig

def plot_gev_map(gevpar):
    fig,axes = plt.subplots(figsize=(20,8),nrows=3,ncols=1,subplot_kw={'projection': ccrs.PlateCarree()},gridspec_kw={'hspace': 0.02, 'wspace': 0.004})
    pcmargs = dict(x='lon',y='lat',cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree(),cbar_kwargs={'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15})
    ax = axes[0]
    xr.plot.pcolormesh(gevpar.sel(param='shape'), ax=ax, **pcmargs)
    ax.coastlines()
    ax.set_title('Shape')
    ax.set_xlabel('')
    ax.set_ylabel('Lat')
    ax = axes[1]
    xr.plot.pcolormesh(-gevpar.sel(param='loc'), ax=ax, **pcmargs)
    ax.coastlines()
    ax.set_title('Location')
    ax.set_xlabel('')
    ax.set_ylabel('Lat')
    ax = axes[2]
    xr.plot.pcolormesh(gevpar.sel(param='scale'), ax=ax, **pcmargs)
    ax.coastlines()
    ax.set_title('Scale')
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    return fig

def plot_gev_with_t2m_map(t2m,gevpar):
    # Essentially Gaussian parameters next to GEV parameters
    fig,axes = plt.subplots(figsize=(20,8),nrows=3,ncols=2,subplot_kw={'projection': ccrs.PlateCarree()},gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
    pcmargs = dict(x='lon',y='lat',cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree(),cbar_kwargs={'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 'ticks': ticker.MaxNLocator(nbins=5)})
    loc_fields = (t2m.min('time').mean('member'), -gevpar.sel(param='loc'))
    loc_vmin,loc_vmax = (min((da.min().item() for da in loc_fields)), max((da.max().item() for da in loc_fields)))
    loc_titles = [r'Mean',r'GEV location']
    scale_fields = (t2m.min('time').std('member'), gevpar.sel(param='scale'))
    scale_vmin,scale_vmax = (min((da.min().item() for da in scale_fields)), max((da.max().item() for da in scale_fields)))
    scale_titles = [r'Std. Dev.',r'GEV scale']
    shape_fields = (None,gevpar.sel(param='shape'))
    shape_vmin,shape_vmax = (-max((np.abs(da).max().item() for da in scale_fields if da is not None)), max((np.abs(da).max().item() for da in scale_fields)))
    shape_titles = [None,r'GEV shape']

    fields = loc_fields + scale_fields + shape_fields
    vmin = [loc_vmin]*2 + [scale_vmin]*2 + [shape_vmin]*2
    vmax = [loc_vmax]*2 + [scale_vmax]*2 + [shape_vmax]*2
    titles = loc_titles + scale_titles + shape_titles

    print(f'{vmin = }')
    print(f'{vmax = }')
    for i in range(6):
        ax = axes.flat[i]
        if fields[i] is None:
            ax.axis('off')
            continue
        xr.plot.pcolormesh(fields[i], ax=ax, vmin=vmin[i], vmax=vmax[i], **pcmargs)
        ax.set_title(titles[i])
    for ax in axes[-1,:]:
        ax.set_xlabel('Lon')
    for ax in axes[:,0]:
        ax.set_ylabel('Lat')
    return fig,axes






def reduce_gcm(i_gcm,i_expt,i_init):
    # One GCM, one forcing (expt), one initialization (init), multiple coarse-grainings in space 
    tododict = dict({
        'coarse_grain_time':           0,
        'plot_t2m_map':                0,
        'coarse_grain_space':          1,
        'fit_gev':                     1,
        'plot_gev_map':                0,
        'plot_gev_with_t2m_map':       1,
        })
    gcm,expt,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,figdir,cgs_levels = gcm_workflow(i_gcm,i_expt,i_init)

    # ------------- Coarse-grain in time (cgt) and space (cgts) -------------
    ens_file_cgt = join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day.nc')
    if tododict['coarse_grain_time']:
        ds_cgt = coarse_grain_time(raw_mem_files, mem_labels, event_region, event_time_interval)
        ds_cgt.to_netcdf(ens_file_cgt)
    else:
        ds_cgt = xr.open_dataarray(ens_file_cgt)

    if tododict['plot_t2m_map']:
        fig = plot_t2m_map(ds_cgt.sel(daily_stat='daily_mean'))
        fig.savefig(join(figdir,f't2m_summary_e{expt}_i{init}_cgt1day.png'),**pltkwargs)
        plt.close(fig)

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

        if tododict['plot_gev_map'] and min(cgs_level) > 1:
            fig = plot_gev_map(gevpar.sel(daily_stat='daily_mean'))
            fig.savefig(join(figdir,f'gev_map_e{expt}_i{init}_cgs{cgs_key}.png'), **pltkwargs)
            plt.close(fig)

        if tododict['plot_gev_with_t2m_map'] and min(cgs_level) > 1:
            fig,axes = plot_gev_with_t2m_map(ds_cgt.sel(daily_stat='daily_mean'), gevpar.sel(daily_stat='daily_mean'))
            fig.savefig(join(figdir,f'gevwitht2m_map_e{expt}_i{init}_cgs{cgs_key}.png'), **pltkwargs)
            plt.close(fig)
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
