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
    cgs_levels = [(1,1),(5,1),(10,2),(20,4),(40,8)] #,(141,16)]
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
    print(f'{raw_data_dir = }')
    ens_path_skeleton = join(raw_data_dir,'r*i*p*f*')
    mem_labels = [basename(p) for p in glob.glob(ens_path_skeleton)]

    path_skeleton = join(ens_path_skeleton,'6hrPt','tas','g*','v*','*.nc')
    raw_mem_files = glob.glob(path_skeleton)
    print(f'{len(raw_mem_files) = }')
    # 2. Spatiotemporal sub-selection and coarse-graining (cg)
    event_region = dict(lat=slice(50,65),lon=slice(-10,130))
    event_time_interval = (datetime.datetime(2018,2,21,0),datetime.datetime(2018,3,8,18))
    reduced_data_dir = join('/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018/results_2024-05-04',gcm)
    makedirs(reduced_data_dir,exist_ok=True)
    reduced_data_dir_era5 = '/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018/results_2024-05-04/era5'

    # spatial coarse graining (cgs)
    cgs_levels = analysis_multiparams()

    # bootstrap parameters
    n_boot = 1000
    confint_width = 0.5


    # Plotting dir 
    figdir = f'/home/users/ju26596/snapsi_analysis_figures/feb2018/figures_2024-05-04/{gcm}'
    makedirs(figdir,exist_ok=True)
       
    select_regions = ( # Indexed by cgs_level
            (), # level (1,1)
            (), # level (5,1)
            ((1,1),(3,1),(4,1),(6,1)),
            (), # level (20,4)
            (), # level (40,8)
            (), # level (141,16)
            )
    select_points = ()
    risk_levels = np.exp(np.linspace(np.log(0.001),np.log(49/50),30))
    workflow = (
            gcm,expt,init,
            event_region,event_time_interval,
            raw_mem_files,mem_labels,reduced_data_dir,reduced_data_dir_era5,figdir,
            cgs_levels,select_regions,risk_levels,n_boot,confint_width
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
    gcm,expt0,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,reduced_data_dir_era5,figdir,cgs_levels,select_regions,risk_levels,n_boot,confint_width = gcm_workflow(i_gcm,i0_expt,i_init)
    _,expt1,_,_,_,_,_,_,_,_,_,_,_,_,_ = gcm_workflow(i_gcm,i1_expt,i_init)
    ds_cgt_0,ds_cgt_1 = (xr.open_dataarray(join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day.nc')) for expt in (expt0,expt1))
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        ds_cgts_mint_0,ds_cgts_mint_1 = (xr.open_dataarray(join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day_cgs{cgs_key}.nc')).sel(daily_stat='daily_mean').min('time') for expt in (expt0,expt1))
        gevpar_0,gevpar_1 = (xr.open_dataarray(join(reduced_data_dir,f'gevpar_e{expt}_i{init}_cgs{cgs_key}.nc')).sel(daily_stat='daily_mean') for expt in (expt0,expt1))
        if tododict['plot_param_diff_map']:
            fig,axes = pipeline_base.plot_statpar_map_difference(ds_cgts_mint_0,ds_cgts_mint_1,gevpar_0,gevpar_1,locsign=-1)
            datestr = datetime.datetime.strptime(init,"%Y%m%d").strftime("%Y-%m-%d")
            fig.suptitle(f'{expt1} - {expt0}, init {datestr}')
            fig.savefig(join(figdir,f'statpar_diffmap_e0{expt0}_e1{expt1}_i{init}_cgs{cgs_key}.png'), **pltkwargs)
            plt.close(fig)
    return



def compare_expts(i_gcm, i_init):
    tododict = dict({
        'plot_statpar_map_diff':           0,
        'plot_relrisk_map_diff':           1,
        'plot_gev_select_regions':         0,
        })
    expts = []
    for i_expt in range(3):
        gcm,expt,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,reduced_data_dir_era5,figdir,cgs_levels,select_regions,risk_levels,n_boot,confint_width = gcm_workflow(i_gcm,i_expt,i_init)
        expts.append(expt)
    datestr = datetime.datetime.strptime(init,"%Y%m%d").strftime("%Y-%m-%d")

    daily_stat = 'daily_mean'
    boot_type = 'percentile'
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        ds_cgts_mints = dict()
        gevpars = dict()
        for expt in expts:
            ds_cgts_mints[expt] = xr.open_dataarray(join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day_cgs{cgs_key}.nc')).min('time')
            gevpars[expt] = xr.open_dataarray(join(reduced_data_dir,f'gevpar_e{expt}_i{init}_cgs{cgs_key}.nc'))
        ds_cgts_mints['era5'] = xr.open_dataarray(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc')).min('time')
        if min(cgs_level) > 1: 
            lon_blocksize,lat_blocksize = (ds_cgts_mints['era5'].coords[d][:2].diff(d).item() for d in ('lon','lat'))
        if tododict['plot_statpar_map_diff']:
            # control-free, nudged-free
            statsel = dict(daily_stat=daily_stat)
            for (expt0,expt1) in (('free','control'),('free','nudged')):
                fig,axes = pipeline_base.plot_statpar_map_difference(ds_cgts_mints[expt0].sel(statsel),ds_cgts_mints[expt1].sel(statsel), gevpars[expt0].sel(statsel), gevpars[expt1].sel(statsel), locsign=-1)
                fig.suptitle(f'{gcm} ({expt1} - {expt0}), init {datestr}')
                fig.savefig(join(figdir,f'statpar_map_e{expt1}minus{expt0}_i{init}_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
                plt.close(fig)

        for (i_lon,i_lat) in select_regions[i_cgs_level]:
            mintemps = dict()
            gevpar_regs = dict()
            mintemp_levels_regs = dict()
            for expt in expts:
                mintemps[expt] = ds_cgts_mints[expt].sel(daily_stat=daily_stat).isel(lon=i_lon,lat=i_lat)
                mintemp_levels_regs[expt] = np.load(join(reduced_data_dir,f'mintemp_levels_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                gevpar_regs[expt] = xr.open_dataarray(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
            mintemps['era5'] = ds_cgts_mints['era5'].sel(daily_stat=daily_stat).isel(lon=i_lon,lat=i_lat)
            mintemp_levels_regs['era5'] = np.load(join(reduced_data_dir_era5,f'mintemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
            gevpar_regs['era5'] = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
            center_lon,center_lat = (mintemps['era5'].coords[d].item() for d in ('lon','lat'))
            lonlatstr = r'$\lambda=%d\pm%d,\phi=%d\pm%d$'%(center_lon,lon_blocksize/2,center_lat,lat_blocksize/2)
            if tododict['plot_gev_select_regions']:
                print(f'Plotting GEV select regions')
                # Plot all four curves on one plot (ERA5, free, control, nudged)
                colors = dict({'era5': 'black', 'control': 'dodgerblue', 'free': 'limegreen', 'nudged': 'red'})

                fig,axes = plt.subplots(ncols=2, figsize=(12,4), sharey=True)
                for ax in axes: 
                    ax.axhline(mintemps['era5'].sel(member=2018).item(), color='black', linestyle='--')
                ax = axes[0]
                handles = []

                # Interpolate for relative risk 
                mintemp_levels_range = [mintemp_levels_regs['era5'][0,:].min().item(),mintemp_levels_regs['era5'][0,:].max().item()] #(max([T.min().item() for T in mintemp_levels_regs.values()]), min([T.max().item() for T in mintemp_levels_regs.values()]))
                mintemp_levels_common = np.linspace(*mintemp_levels_range, 30)
                risk_at_levels = dict()
                for expt in expts + ['era5']:
                    shape,location,scale = (gevpar_regs[expt].sel(param=p).isel(boot=0) for p in ('shape','location','scale'))
                    param_label = ','.join([
                        r'$\mu=%d$'%(-location),
                        r'$\sigma=%d$'%(scale),
                        r'$\xi=%+.2f$'%(shape)
                        ])
                    mintemp = mintemps[expt].to_numpy()
                    mintemp_levels_reg = mintemp_levels_regs[expt]
                    order = np.argsort(mintemp)
                    rank = np.argsort(order)
                    risk_empirical = np.arange(1,len(mintemp)+1)/len(mintemp)
                    ax.scatter(risk_empirical, mintemp[order], color=colors[expt], marker='+')
                    h, = ax.plot(risk_levels,mintemp_levels_reg[0,:],color=colors[expt],label=r'%s (%s)'%(expt+' '*(7-len(expt)),param_label))
                    handles.append(h)
                    boot_quant_lo,boot_quant_hi = (np.quantile(mintemp_levels_reg[1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                    if boot_type == 'percentile':
                        lo,hi = boot_quant_lo,boot_quant_hi
                    elif boot_type == 'basic':
                        lo,hi = (2*mintemp_levels_reg[0,:]-boot_quant_hi,2*mintemp_levels_reg[0,:]-boot_quant_lo)
                    ax.fill_between(risk_levels, lo, hi, fc=colors[expt], ec='none', alpha=0.3, zorder=-1)
                    # Interpolate every bootstrap
                    func = lambda T: np.interp(mintemp_levels_common, T, risk_levels)
                    risk_at_levels[expt] = np.apply_along_axis(func, 1, mintemp_levels_reg)
                ax.set_xscale('log')
                ax.set_xlabel(r'$\mathbb{P}\{\min_t\langle T(t)\rangle_{\mathrm{region}}\leq T\}$')
                ax.set_ylabel(r'$T$')
                ax.legend(handles=handles, bbox_to_anchor=(-0.25,0.5), loc='center right')

                # Relative risk
                ax = axes[1]
                ax.axvline(1.0, color=colors['free'], linestyle='-')
                # Interpolate to common set of levels and plot relative risk 
                handles = []
                for (expt0,expt1) in (('free','control'),('free','nudged')):
                    risk_ratio = risk_at_levels[expt1] / risk_at_levels[expt0]
                    h, = ax.plot(risk_ratio[0,:], mintemp_levels_common, color=colors[expt1], label=r'%s/%s'%(expt1,expt0))
                    handles.append(h)
                    boot_quant_lo,boot_quant_hi = (np.quantile(risk_ratio[1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                    if boot_type == 'percentile':
                        lo,hi = boot_quant_lo,boot_quant_hi
                    else:
                        lo,hi = 2*risk_ratio[0,:]-boot_quant_hi, 2*risk_ratio[0,:]-boot_quant_lo
                    ax.fill_betweenx(mintemp_levels_common, lo, hi, fc=colors[expt1], ec='none', alpha=0.3, zorder=-1)
                ax.set_xscale('log')
                ax.set_xlim([0.5,5.0])
                ax.get_xaxis().set_major_formatter(ticker.NullFormatter())
                ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())
                xticks = [0.5,1.0,2.0,5.0]
                ax.set_xticks(xticks, [r'%g'%(xtick) for xtick in xticks])
                #ax.legend(handles=handles)
                ax.set_xlabel(r'Relative risk w.r.t. free')

                for ax in axes:
                    ax.set_ylim(mintemp_levels_range)

                fig.suptitle(f'{gcm}, init {datestr} at {lonlatstr}')

                fig.savefig(join(figdir,f'riskplot_reg_eall_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
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
        'compute_risk':                1,
        'plot_risk_map':               1,
        'fit_gev_select_regions':      0,
        'plot_gev_select_regions':     0,
        })
    gcm,expt,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,reduced_data_dir_era5,figdir,cgs_levels,select_regions,risk_levels,n_boot,confint_width = gcm_workflow(i_gcm,i_expt,i_init)

    # ------------- Coarse-grain in time (cgt) and space (cgts) -------------
    ens_file_cgt = join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day.nc')
    if tododict['coarse_grain_time']:
        ds_cgt = coarse_grain_time(raw_mem_files, mem_labels, event_region, event_time_interval)
        ds_cgt.to_netcdf(ens_file_cgt)
    else:
        ds_cgt = xr.open_dataarray(ens_file_cgt)

    ds_cgt_mint = ds_cgt.min(dim='time')
    # Load ERA5 as well
    era5_file_cgt = join(reduced_data_dir_era5,f't2m_cgt1day.nc')
    ds_cgt_era5 = xr.open_dataarray(era5_file_cgt)
    ds_cgt_mint_era5 = ds_cgt_era5.min(dim='time')
    if tododict['plot_t2m_sumstats_map']:
        for daily_stat in ['daily_min','daily_mean']:
            fig,axes = pipeline_base.plot_sumstats_map(ds_cgt_mint.sel(daily_stat=daily_stat))
            fig.suptitle(f'{gcm}, {expt}, init {init} {daily_stat}')
            fig.savefig(join(figdir,f't2m_sumstats_map_{daily_stat}_e{expt}_i{init}_cgt1day.png'),**pltkwargs)
            plt.close(fig)

    daily_stat = 'daily_mean'
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        if min(cgs_level) > 1: 
            lon_blocksize,lat_blocksize = (ds_cgt_era5[d][:2].diff(d).item() for d in ('lon','lat'))
        ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day_cgs{cgs_key}.nc')
        if tododict['coarse_grain_space']:
            ds_cgts = pipeline_base.coarse_grain_space(ds_cgt, cgs_level)
            ds_cgts.to_netcdf(ens_file_cgts)
        else:
            ds_cgts = xr.open_dataarray(ens_file_cgts)
        ds_cgts_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))
        # ---- Minimize in time -----------------------------
        ds_cgts_mint = ds_cgts.min(dim='time')
        ds_cgts_mint_era5 = ds_cgts_era5.min(dim='time')
        # ----------- Perform GEV fitting (on negative temperature) --------------
        gev_param_file = join(reduced_data_dir,f'gevpar_e{expt}_i{init}_cgs{cgs_key}.nc')
        if tododict['fit_gev']:
            gevpar = pipeline_base.fit_gev_mintemp(ds_cgts_mint,method='PWM')
            gevpar.to_netcdf(gev_param_file)
        else:
            gevpar = xr.open_dataarray(gev_param_file)
        gevpar_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_cgs{cgs_key}.nc'))
        # ----------------- Compute risk w.r.t. ERA5 ---------------
        risk_file = join(reduced_data_dir,f'risk_e{expt}_i{init}_cgs{cgs_key}.nc')
        if tododict['compute_risk']:
            kwargs = dict(daily_stat=daily_stat,drop=True)
            risk = pipeline_base.compute_risk(
                    ds_cgts_mint.sel(**kwargs),
                    ds_cgts_mint_era5.sel(member=2018,**kwargs),
                    gevpar.sel(**kwargs),
                    gevpar_era5.sel(**kwargs),
                    locsign=-1)
            risk.to_netcdf(risk_file)
        else:
            risk = xr.open_netcdf(risk_file)

        if tododict['plot_risk_map'] and min(cgs_level) > 1:
            fig,ax = pipeline_base.plot_risk_map(ds_cgts_mint.sel(daily_stat=daily_stat,drop=True), ds_cgts_mint_era5.sel(member=2018, daily_stat=daily_stat,drop=True), gevpar.sel(daily_stat=daily_stat,drop=True), gevpar_era5.sel(daily_stat=daily_stat,drop=True), locsign=-1)
            fig.savefig(join(figdir,f'risk_map_e{expt}_i{init}_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
            plt.close(fig)
        if tododict['plot_statpar_map'] and min(cgs_level) > 1:
            for daily_stat in ['daily_mean']:
                statsel = dict(daily_stat=daily_stat)
                # Maps as-is
                fig,axes = pipeline_base.plot_statpar_map(ds_cgts_mint.sel(statsel), gevpar.sel(statsel), locsign=-1)
                datestr = datetime.datetime.strptime(init,"%Y%m%d").strftime("%Y-%m-%d")
                fig.suptitle(f'{gcm} {expt}, init {datestr}')
                fig.savefig(join(figdir,f'statpar_map_e{expt}_i{init}_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
                plt.close(fig)
                # Differences from ERA5
                print(f'{ds_cgts_mint_era5.shape = }')
                print(f'{ds_cgts_mint.shape = }')
                print(f'{gevpar_era5.shape = }')
                print(f'{gevpar.shape = }')
                fig,axes = pipeline_base.plot_statpar_map_difference(ds_cgts_mint_era5.sel(statsel),ds_cgts_mint.sel(statsel),gevpar_era5.sel(statsel),gevpar.sel(statsel),locsign=-1)
                fig.suptitle(f'({gcm} {expt}, init {datestr}) - ERA5')
                fig.savefig(join(figdir,f'statpar_map_e{expt}_i{init}_cgs{cgs_key}_{daily_stat}_minusera5.png'), **pltkwargs)
                plt.close(fig)

                    
        # Do a more thorough uncertainty quantification at a specific site or region, using bootstrap analysis and goodness-of-fit etc. Maybe parallelize over all sites too

        for (i_lon,i_lat) in select_regions[i_cgs_level]:
            mintemp = ds_cgts_mint.sel(daily_stat='daily_mean').isel(lon=i_lon,lat=i_lat).to_numpy()
            mintemp_era5 = ds_cgts_mint_era5.sel(daily_stat='daily_mean').isel(lon=i_lon,lat=i_lat)
            center_lon,center_lat = mintemp_era5.lon.item(),mintemp_era5.lat.item()
            mintemp_era5 = mintemp_era5.to_numpy()
            lonlatstr = r'$\lambda=%d\pm%d,\phi=%d\pm%d$'%(center_lon,lon_blocksize/2,center_lat,lat_blocksize/2)
            if tododict['fit_gev_select_regions']:
                if np.any(np.isnan(mintemp)):
                    raise Exception(f'{mintemp = }')
                gevpar_reg,mintemp_levels_reg = pipeline_base.fit_gev_mintemp_1d_uq(mintemp,risk_levels, method='PWM', n_boot=n_boot)
                gevpar_reg.to_netcdf(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                np.save(join(reduced_data_dir,f'mintemp_levels_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'), mintemp_levels_reg)
            else:
                mintemp_levels_reg = np.load(join(reduced_data_dir,f'mintemp_levels_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                gevpar_reg = xr.open_dataarray(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                # Also load the ERA5 data 
            mintemp_levels_reg_era5 = np.load(join(reduced_data_dir_era5,f'mintemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
            gevpar_reg_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
            print(f'{i_lon = }, {i_lat = }')
            print(f'{gevpar_reg_era5.isel(boot=0) = }')

            if tododict['plot_gev_select_regions']:
                fig,ax = plt.subplots()
                shape,location,scale = (gevpar_reg.sel(param=p).isel(boot=0) for p in ('shape','location','scale'))
                shape_era5,location_era5,scale_era5 = (gevpar_reg_era5.sel(param=p).isel(boot=0) for p in ('shape','location','scale'))
                param_label = ','.join([
                    r'$\mu=%d$'%(-location),
                    r'$\sigma=%d$'%(scale),
                    r'$\xi=%.2f$'%(shape)
                    ])
                param_label_era5 = ','.join([
                    r'$\mu=%d$'%(-location_era5),
                    r'$\sigma=%d$'%(scale_era5),
                    r'$\xi=%.2f$'%(shape_era5)
                    ])
                # GCM data
                order = np.argsort(mintemp)
                rank = np.argsort(order)
                risk_empirical = np.arange(1,len(mintemp)+1)/len(mintemp)
                ax.scatter(risk_empirical, mintemp[order], color='red', marker='+')
                hgcm, = ax.plot(risk_levels,mintemp_levels_reg[0,:],color='red',label=r'%s (%s)'%(gcm,param_label))
                ax.fill_between(risk_levels, np.quantile(mintemp_levels_reg[1:], 0.25, axis=0), np.quantile(mintemp_levels_reg[1:], 0.75, axis=0), fc='red', ec='none', alpha=0.3, zorder=-1)
                # Now ERA5
                order = np.argsort(mintemp_era5)
                rank = np.argsort(order)
                risk_empirical = np.arange(1,len(mintemp_era5)+1)/len(mintemp_era5)
                ax.scatter(risk_empirical, mintemp_era5[order], color='black', marker='+')
                # Special marker for 2018
                i_mem_2018 = np.where(ds_cgts_mint_era5.member == 2018)[0][0]

                ax.scatter(risk_empirical[rank[i_mem_2018]], mintemp_era5[i_mem_2018], color='black', marker='o')
                hera5, = ax.plot(risk_levels,mintemp_levels_reg_era5[0,:],color='black',label=r'ERA5 (%s)'%(param_label_era5))
                ax.fill_between(risk_levels, np.quantile(mintemp_levels_reg_era5, 0.25, axis=0), np.quantile(mintemp_levels_reg_era5, 0.75, axis=0), fc='gray', ec='none', alpha=0.3, zorder=-1)
                ax.legend(handles=[hera5,hgcm])
                ax.set_xscale('log')
                ax.set_xlabel(r'$\mathbb{P}\{\min_t\langle T(t)\rangle_{\mathrm{region}}\leq T\}$')
                ax.set_ylabel(r'$T$')
                ax.set_title(f'{gcm} {expt}, init {datestr} at {lonlatstr}')

                fig.savefig(join(figdir,f'riskplot_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                plt.close(fig)
        ds_cgts.close()
    ds_cgt.close()
    return 

if __name__ == "__main__":
    idx_gcm = [4,9,11] #[4,9,11]
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
            for i_init in idx_init:
                compare_expts(i_gcm, i_init)
