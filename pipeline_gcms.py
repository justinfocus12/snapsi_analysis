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
import pipeline_base

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
    event_time_interval = (datetime.datetime(2018,2,21,0),datetime.datetime(2018,3,8,18))
    reduced_data_dir = join('/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018/results_2024-05-04',gcm)
    makedirs(reduced_data_dir,exist_ok=True)

    # spatial coarse graining (cgs)
    cgs_levels = analysis_multiparams()

    # Plotting dir 
    figdir = f'/home/users/ju26596/snapsi_analysis_figures/feb2018/figures_2024-05-04/{gcm}'
    makedirs(figdir,exist_ok=True)
       
    select_regions = ( # Indexed by cgs_level
            (), # level (1,1)
            (), # level (5,1)
            ((1,1),(3,1),(4,1)),
            (), # level (20,4)
            (), # level (40,8)
            (), # level (141,16)
            )
    select_points = ()
    risk_levels = np.exp(np.linspace(np.log(0.001),np.log(49/50),30))
    workflow = (
            gcm,expt,init,
            event_region,event_time_interval,
            raw_mem_files,mem_labels,reduced_data_dir,figdir,
            cgs_levels,select_regions,risk_levels
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




def compare_statpar_maps_2expts(i_gcm,i0_expt,i1_expt,i_init):
    tododict = dict({
        'plot_statpar_map_diff':           1,
        'plot_gev_select_regions_diff':    1,
        })
    # Assumes both have been reduced
    gcm,expt0,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,figdir,cgs_levels,select_regions,risk_levels = gcm_workflow(i_gcm,i0_expt,i_init)
    _,expt1,_,_,_,_,_,_,_,_,_,_ = gcm_workflow(i_gcm,i1_expt,i_init)
    ds_cgt_0,ds_cgt_1 = (xr.open_dataarray(join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day.nc')) for expt in (expt0,expt1))
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        ds_cgts_mint_0,ds_cgts_mint_1 = (xr.open_dataarray(join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day_cgs{cgs_key}.nc')).sel(daily_stat='daily_mean').min('time') for expt in (expt0,expt1))
        gevpar_0,gevpar_1 = (xr.open_dataarray(join(reduced_data_dir,f'gevpar_e{expt}_i{init}_cgs{cgs_key}.nc')).sel(daily_stat='daily_mean') for expt in (expt0,expt1))
        if tododict['plot_param_diff_map']:
            fig,axes = pipeline_base.plot_statpar_map_difference(ds_cgts_mint_0,ds_cgts_mint_1,gevpar_0,gevpar_1)
            datestr = datetime.datetime.strptime(init,"%Y%m%d").strftime("%Y-%m-%d")
            fig.suptitle(f'{expt1} - {expt0}, init {datestr}')
            fig.savefig(join(figdir,f'statpar_diffmap_e0{expt0}_e1{expt1}_i{init}_cgs{cgs_key}.png'), **pltkwargs)
            plt.close(fig)
    return






def reduce_gcm(i_gcm,i_expt,i_init):
    # One GCM, one forcing (expt), one initialization (init), multiple coarse-grainings in space 
    tododict = dict({
        'coarse_grain_time':           0,
        'plot_t2m_sumstats_map':       0,
        'coarse_grain_space':          0,
        'fit_gev':                     0,
        'plot_statpar_map':            0,
        'fit_gev_select_regions':      1,
        'plot_gev_select_regions':     1,
        })
    gcm,expt,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,figdir,cgs_levels,select_regions,risk_levels = gcm_workflow(i_gcm,i_expt,i_init)

    # ------------- Coarse-grain in time (cgt) and space (cgts) -------------
    ens_file_cgt = join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day.nc')
    if tododict['coarse_grain_time']:
        ds_cgt = coarse_grain_time(raw_mem_files, mem_labels, event_region, event_time_interval)
        ds_cgt.to_netcdf(ens_file_cgt)
    else:
        ds_cgt = xr.open_dataarray(ens_file_cgt)

    ds_cgt_mint = ds_cgt.min(dim='time')
    if tododict['plot_t2m_sumstats_map']:
        for daily_stat in ['daily_min','daily_mean']:
            fig,axes = pipeline_base.plot_sumstats_map(ds_cgt_mint.sel(daily_stat=daily_stat))
            fig.suptitle(f'{gcm}, {expt}, init {init} {daily_stat}')
            fig.savefig(join(figdir,f't2m_sumstats_map_{daily_stat}_e{expt}_i{init}_cgt1day.png'),**pltkwargs)
            plt.close(fig)

    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day_cgs{cgs_key}.nc')
        if tododict['coarse_grain_space']:
            ds_cgts = pipeline_base.coarse_grain_space(ds_cgt, cgs_level)
            ds_cgts.to_netcdf(ens_file_cgts)
        else:
            ds_cgts = xr.open_dataarray(ens_file_cgts)
        # ---- Minimize in time -----------------------------
        ds_cgts_mint = ds_cgts.min(dim='time')
        # ----------- Perform GEV fitting (on negative temperature) --------------
        gev_param_file = join(reduced_data_dir,f'gevpar_e{expt}_i{init}_cgs{cgs_key}.nc')
        if tododict['fit_gev']:
            gevpar = pipeline_base.fit_gev_mintemp(ds_cgts_mint)
            gevpar.to_netcdf(gev_param_file)
        else:
            gevpar = xr.open_dataarray(gev_param_file)
        if tododict['plot_statpar_map'] and min(cgs_level) > 1:
            for daily_stat in ['daily_mean']:
                fig,axes = pipeline_base.plot_statpar_map(ds_cgts_mint.sel(daily_stat=daily_stat), gevpar.sel(daily_stat=daily_stat))
                datestr = datetime.datetime.strptime(init,"%Y%m%d").strftime("%Y-%m-%d")
                fig.suptitle(f'{expt}, init {datestr}')
                fig.savefig(join(figdir,f'statpar_map_e{expt}_i{init}_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
                plt.close(fig)
        # Do a more thorough uncertainty quantification at a specific site or region, using bootstrap analysis and goodness-of-fit etc. Maybe parallelize over all sites too

        for (i_lon,i_lat) in select_regions[i_cgs_level]:
            mintemp = ds_cgts_mint.sel(daily_stat='daily_mean').isel(lon=i_lon,lat=i_lat).to_numpy()
            if tododict['fit_gev_select_regions']:
                gevpar_reg,mintemp_levels_reg = pipeline_base.fit_gev_mintemp_1d_uq(mintemp,risk_levels)
                gevpar_reg.to_netcdf(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                np.save(join(data_dir,f'mintemp_levels_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy', mintemp_levels_reg))
            else:
                mintemp_levels_reg = np.load(join(data_dir,f'mintemp_levels_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                gevpar_reg = xr.open_dataarray(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))

            if tododict['plot_gev_select_regions']:
                fig,ax = plt.subplots()
                # Left: probability plot; right: timeseries of minima over years 
                order = np.argsort(mintemp)
                rank = np.argsort(order)
                risk_empirical = np.arange(1,len(mintemp)+1)/len(mintemp)
                ax.scatter(risk_empirical, mintemp[order], color='black', marker='+')
                ax.plot(risk_levels,mintemp_levels_reg[0,:],color='red')
                ax.fill_between(risk_levels, np.quantile(mintemp_levels_reg, 0.25, axis=0), np.quantile(mintemp_levels_reg, 0.75, axis=0), fc='red', ec='none', alpha=0.3, zorder=-1)
                ax.set_xscale('log')
                ax.set_xlabel(r'$\mathbb{P}\{\min_t\langle T(t)\rangle_{\mathrm{region}}\leq T\}$')
                ax.set_ylabel(r'$T$')
                shape,location,scale = (gevpar_reg.sel(param=p).isel(boot=0) for p in ('shape','location','scale'))
                param_label = '\n'.join([
                    r'$\mu=%d$'%(-location),
                    r'$\sigma=%d$'%(scale),
                    r'$\xi=%.2f$'%(shape)
                    ])
                ax.text(0.1,0.9,param_label, transform=ax.transAxes, ha='left', va='top')
                ax.set_title(f'{expt}, init {init}')

                fig.savefig(join(figdir,f'riskplot_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                plt.close(fig)
        ds_cgts.close()
    ds_cgt.close()
    return 

if __name__ == "__main__":
    idx_gcm = [4]
    idx_expt = [0,1,2]
    idx_expt_pairs = [(1,0),(1,2)]
    idx_init = [0,1]
    procedure = sys.argv[1]
    if procedure == 'reduce':
        for i_gcm in idx_gcm:
            for i_expt in idx_expt:
                for i_init in idx_init:
                    reduce_gcm(i_gcm,i_expt,i_init)
    elif procedure == 'compare_expts':
        for i_gcm in idx_gcm:
            for (i0_expt,i1_expt) in idx_expt_pairs:
                for i_init in idx_init:
                    compare_statpar_maps_2expts(i_gcm,i0_expt,i1_expt,i_init)
