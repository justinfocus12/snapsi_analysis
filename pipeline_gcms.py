import numpy as np
import xarray as xr
from cartopy import crs as ccrs
import netCDF4
from matplotlib import pyplot as plt, rcParams, ticker, colors as mplcolors, patches as mplpatches
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

def gcm_multiparams(which_ssw):
    # Sets out all the options for which piece of data to ingest
    gcm2institute = all_gcms_institutes()
    gcms = list(gcm2institute.keys())
    expts = ['control','free','nudged']
    if "feb2018" == which_ssw:
        inits = ['20180125','20180208']
    elif "jan2019" == which_ssw:
        inits = ['20181213','20190108']
    elif "sep2019" == which_ssw:
        inits = ['20190829','20191001']
    return gcms, expts, inits

def analysis_multiparams(which_ssw):
    # lon/lat ratios are 6/1 at the bottom and 4/1 at the top; stick to 5/1 
    if "feb2018" == which_ssw:
        cgs_levels = [(1,1),(5,1),(10,2),(20,4),(40,8)] #,(141,16)]
        select_regions = ( # Indexed by cgs_level
                ((0,0),), # level (1,1)
                (), # level (5,1)
                ((i,j) for i in range(10) for j in range(2)),
                (), # level (20,4)
                (), # level (40,8)
                (), # level (141,16)
                )
    elif "jan2019" == which_ssw:
        cgs_levels = [(1,1),(2,1),(5,3),(15,9)]
        select_regions = ( # Indexed by cgs_level
                ((0,0),), # level (1,1)
                ((0,0),(1,0)), # level (2,1)
                ((i,j) for i in range(5) for j in range(3)),
                (),
                )
    elif "sep2019" == which_ssw:
        cgs_levels = [(1,1),(2,2),(5,5),(15,15)]
        select_regions = ( # Indexed by cgs_level
                ((0,0),), # level (1,1)
                ((0,0),(1,0),(0,1),(1,1)), # level (2,1)
                ((i,j) for i in range(5) for j in range(5)),
                (),
                )
        cgs_levels = [(1,1),(2,2),(7,6)]
        select_regions = ( # Indexed by cgs_level
                ((0,0),), # level (1,1)
                ((0,0),(1,0),(0,1),(1,1)), # level (2,1)
                ((i,j) for i in range(7) for j in range(6)),
                )
    return cgs_levels,select_regions


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


def gcm_workflow(which_ssw, i_gcm, i_expt, i_init, verbose=False):
    # Sets out the folders necessary to ingest a chunk of data specified by the input arguments 
    gcms,expts,inits = gcm_multiparams(which_ssw)
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
    print(f'{mem_labels = }')
    print(f'{len(mem_labels) = }')

    if "GloSea6" == gcm: # for some reason there's a single odd version file 
        path_skeleton = join(ens_path_skeleton,'6hrPt','tas','g*','v20230403*','*.nc')
    else:
        path_skeleton = join(ens_path_skeleton,'6hrPt','tas','g*','v*','*.nc')
    print(f'{path_skeleton = }')
    raw_mem_files = glob.glob(path_skeleton)
    print(f'{len(raw_mem_files) = }')
    for rmf in raw_mem_files:
        print(rmf)
    assert len(raw_mem_files) == len(mem_labels)
    

    analysis_date = '2024-10-09'
    # 2. Spatiotemporal sub-selection and coarse-graining (cg)
    reduced_data_dir = join(f'/gws/nopw/j04/snapsi/processed/wg2/ju26596',gcm,analysis_date,gcm)
    if "feb2018" == which_ssw:
        event_time_interval = [datetime.datetime(2018,2,21,0), datetime.datetime(2018,3,8,22)] # for the reference year 
        event_region = dict(lat=slice(50,65),lon=slice(-10,130))
    elif "jan2019" == which_ssw:
        event_time_interval = [datetime.datetime(2019,1,1,0), datetime.datetime(2019,1,31,22)] # for the reference year 
        event_region = dict(lat=slice(30,45),lon=slice(-95,-70))
    elif "sep2019" == which_ssw:
        event_time_interval = [datetime.datetime(2019,10,1,0), datetime.datetime(2019,10,14,22)]
        event_region = dict(lat=slice(-46,-10), lon=slice(112,154))

    processed_data_dir = '/gws/nopw/j04/snapsi/processed/wg2/ju26596'
    reduced_data_dir = join(processed_data_dir,which_ssw,analysis_date,gcm)
    reduced_data_dir_era5 = join(processed_data_dir,which_ssw,analysis_date,'era5')
    figdir = join('/home/users/ju26596/snapsi_analysis_figures',which_ssw,analysis_date,gcm)
    makedirs(reduced_data_dir,exist_ok=True)
    makedirs(figdir,exist_ok=True)
    # spatial coarse graining (cgs)
    cgs_levels,select_regions = analysis_multiparams(which_ssw)

    # bootstrap parameters
    n_boot = 1000
    confint_width = 0.5
    risk_levels = np.exp(np.linspace(np.log(0.001),np.log(49/50),30))
    workflow = (
            gcm,
            expt,
            init,
            event_region,
            event_time_interval,
            raw_mem_files,
            mem_labels,
            reduced_data_dir,
            reduced_data_dir_era5,
            figdir,
            cgs_levels,
            select_regions,
            risk_levels,
            n_boot,
            confint_width
            )
    return workflow

def coarse_grain_time(raw_mem_files, mem_labels, event_region, fcdate, event_time_interval, use_dask=False):
    timesel = dict(time=slice(event_time_interval[0],event_time_interval[1]))
    preprocess = lambda dsmem: preprocess_gcm_6hrPt(dsmem, fcdate, timesel, event_region)
    print(f'{all([exists(f) for f in raw_mem_files]) = }')

    if use_dask:
        ds_ens = xr.open_mfdataset(raw_mem_files, preprocess=preprocess, parallel=False, combine='nested', concat_dim='member').assign_coords(member=mem_labels)
    else:
        ds_ens = []
        for f in raw_mem_files:
            ds_ens.append(preprocess(xr.open_dataset(f)))
            print(f'\n\nAppended file {f}, with dimensions \n{ds_ens[-1].dims}\nshape\n{ds_ens[-1].shape}')
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
    daily_max = (
            ds_ens
            .coarsen({'time': 4}, side='left', coord_func='min')
            .max()
            ).compute()
    ds_ens.close()
    ds_ens_cgt = xr.concat([daily_mean,daily_min,daily_max], dim='daily_stat').assign_coords(daily_stat=['daily_mean','daily_min','daily_max'])
    return ds_ens_cgt




def preprocess_gcm_6hrPt(dsmem,fcdate,timesel,spacesel,verbose=False):
    if verbose:
        print(f'{fcdate = }')
        print(f'{dsmem.time = }')
        print(f'{timesel = }')
    dsmem_tas = (
            utils.rezero_lons(
                dsmem['tas']
                .assign_coords(time=np.arange(fcdate,fcdate+datetime.timedelta(hours=6*dsmem.time.size),datetime.timedelta(hours=6)))
                #.sel(timesel)
                )
            )
    if verbose:
        print(f'{dsmem_tas.time = }')
    dsmem_tas = dsmem_tas.sel(timesel)
    dsmem_tas = (
            dsmem_tas
            .sel(spacesel)
            .isel(time=slice(0,4*int(dsmem_tas.time.size/4)))
            .expand_dims(member=[dsmem.attrs['variant_label']])
            )
    if verbose:
        print(f'Preprocessing done; {dsmem_tas.time.size = }')
        print(f'{dsmem_tas.time = }')
    #sys.exit()
    return dsmem_tas




def compare_statpar_maps_2expts(i_gcm,i0_expt,i1_expt,i_init):
    todo = dict({
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
        if todo['plot_param_diff_map']:
            fig,axes = pipeline_base.plot_statpar_map_difference(ds_cgts_mint_0,ds_cgts_mint_1,gevpar_0,gevpar_1,locsign=-1)
            datestr = datetime.datetime.strptime(init,"%Y%m%d").strftime("%Y-%m-%d")
            fig.suptitle(f'{expt1} - {expt0}, init {datestr}')
            fig.savefig(join(figdir,f'statpar_diffmap_e0{expt0}_e1{expt1}_i{init}_cgs{cgs_key}.png'), **pltkwargs)
            plt.close(fig)
    return

def compare_gcms(which_ssw, idx_gcms):
    # Plot relative risk and absolute risk, perhaps in a 2D space 
    if "sep2019" == which_ssw:
        ext_sign = 1
        ext_symb = "max"
        event_year = 2019
    elif "jan2019" == which_ssw:
        ext_sign = -1
        ext_symb = "min"
        event_year = 2019
    elif "feb2018" == which_ssw:
        ext_sign = -1
        ext_symb = "min"
        event_year = 2018
    cgs_levels,select_regions = analysis_multiparams(which_ssw)
    gcms,expts,inits = gcm_multiparams(which_ssw)
    print(f'{gcms = }')
    colors = dict({'era5': 'black', 'control': 'dodgerblue', 'free': 'limegreen', 'nudged': 'red'})
    yoffsets = dict({'control': 1/3, 'free': 0.0, 'nudged': -1/3})
    figdir = f'/home/users/ju26596/snapsi_analysis_figures/{which_ssw}/2024-10-09/multimodel'
    print(f'{figdir = }')
    xlims = [0.2, 5.0]
    ylims = [-0.5, len(gcms)-0.5]
    slims = (np.array([1.0, 4.0])*rcParams['lines.markersize']) # bounds on the marker size 
    makedirs(figdir, exist_ok=True)
    dpi = 200
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        fig,axes = plt.subplots(figsize=(12,10),ncols=2,nrows=1,sharey=True, dpi=dpi)
        for ax in axes:
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
        # Determine the radius-to-points conversion
        unit_radius_data = 1/6
        unit_radius_display = axes[0].transData.transform([0, unit_radius_data])[1] - axes[0].transData.transform([0,0])[1]
        unit_radius_points = unit_radius_display * 72/fig.dpi
        handles = []
        for (i_gcm,gcm) in enumerate(gcms):
            print(f'Starting {i_gcm,gcm = }')
            if not (i_gcm in idx_gcms):
                continue
            for (i_init,init) in enumerate(inits):
                ax = axes[i_init]
                workflow = gcm_workflow(which_ssw,i_gcm,0,i_init)
                reduced_data_dir = workflow[7]
                reduced_data_dir_era5 = workflow[8]
                risk_levels = workflow[12]
                confint_width = workflow[14]
                event_region = workflow[3]
                for (expt0,expt1) in (('free','free'),('free','control'),('free','nudged')):
                    risk0_file,risk1_file = (join(reduced_data_dir,f'risk_e{expt}_i{init}_cgs{cgs_key}.nc') for expt in (expt0,expt1))
                    risk0 = xr.open_dataarray(risk0_file).to_numpy()
                    risk1 = xr.open_dataarray(risk1_file).to_numpy()
                    rr = np.maximum(xlims[0], np.minimum(xlims[1], risk1/risk0))
                    print(f'{rr.shape = }')
                    h = ax.scatter(rr.flat, (i_gcm+yoffsets[expt1])*np.ones(rr.size), ec=colors[expt1], linewidth=2, fc='none', s=(2*unit_radius_points * (1/4 + risk1*3/4))**2, label=expt1)
                    # Add error bars if the level of coarse-graining is 1x1
                    if len(handles) < 3: handles.append(h)
                    if i_cgs_level == 0:
                        ds_cgts_extts_era5 = (xr.open_dataarray(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc')) * ext_sign).max('time') * ext_sign
                        for i_lon in range(rr.shape[0]):
                            for i_lat in range(rr.shape[1]):
                                exttemp_levels_reg_0 = np.load(join(reduced_data_dir,f'exttemp_levels_reg_e{expt0}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                                exttemp_levels_reg_1 = np.load(join(reduced_data_dir,f'exttemp_levels_reg_e{expt1}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                                exttemp_reg_era5 = ds_cgts_extts_era5.sel(member=event_year,daily_stat='daily_mean').isel(lon=i_lon,lat=i_lat).item()
                                exttemp_levels_range = tuple(extfun(exttemp_reg_era5).item() for extfun in (np.min, np.max))
                                func = lambda T: np.interp(exttemp_reg_era5, T, risk_levels)
                                risk_at_levels_0 = np.apply_along_axis(func, 1, exttemp_levels_reg_0)
                                risk_at_levels_1 = np.apply_along_axis(func, 1, exttemp_levels_reg_1)
                                print(f'{risk_at_levels_0 = }')
                                print(f'{risk_at_levels_1 = }')
                                rel_risk_lo,rel_risk_hi = (np.quantile(risk_at_levels_1/risk_at_levels_0, 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                                print(f'{i_init = }, {expt0 = }, {expt1 = }, {i_cgs_level = }, {rel_risk_lo = }, {rel_risk_hi = }')
                                sys.exit()
                                h, = ax.plot([rel_risk_lo,rel_risk_hi], (i_gcm+yoffsets[expt1])*np.ones(2),color=colors[expt1], linewidth=3, label=expt1)
                            


        filename = join(figdir, f'relrisk_multimodel_cgs{cgs_key}.png')
        axes[0].set_yticks(range(len(gcms)), labels=gcms)
        for (i_ax,ax) in enumerate(axes):
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.axvline(1.0, color='black', linestyle='--')
            ax.axvline(xlims[0], color='black', linestyle='--')
            ax.axvline(xlims[1], color='black', linestyle='--')
            datestr = datetime.datetime.strptime(inits[i_ax],"%Y%m%d").strftime("%Y-%m-%d")
            ax.set_title(f"init {datestr}")
            ax.set_xscale('log')
            xticks = [xlims[0], 1.0, xlims[1]]
            ax.set_xticks(xticks, labels=[str(xtick) for xtick in xticks])
            ax.set_xlabel('rel. risk w.r.t. free')
            for i_gcm in range(len(gcms)):
                ax.axhline(i_gcm+0.5, color='gray')
                ax.axhline(i_gcm-0.5, color='gray')
                ax.axhline(i_gcm+1/6, color='gray', linestyle='dotted')
                ax.axhline(i_gcm-1/6, color='gray', linestyle='dotted')
        dlon = (event_region['lon'].stop - event_region['lon'].start)/(cgs_level[0])
        dlat = (event_region['lat'].stop - event_region['lat'].start)/(cgs_level[1])
        fig.suptitle(r'Relative risk ($%d^\circ$lon$\times%d^\circ$lat regions)'%(dlon, dlat))
        fig.legend(handles=handles, bbox_to_anchor=(0.5,0.0), loc='upper center', ncol=3)
        fig.savefig(filename, **pltkwargs)

        plt.close(fig)
    return
            


def compare_expts(which_ssw, i_gcm, i_init):
    todo = dict({
        'plot_statpar_map_diff':           1,
        'plot_relrisk_map':                1,
        'plot_gev_select_regions':         1,
        })
    expts = []
    for i_expt in range(3):
        gcm,expt,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,reduced_data_dir_era5,figdir,cgs_levels,select_regions,risk_levels,n_boot,confint_width = gcm_workflow(which_ssw, i_gcm,i_expt,i_init)
        expts.append(expt)
    datestr = datetime.datetime.strptime(init,"%Y%m%d").strftime("%Y-%m-%d")

    if "sep2019" == which_ssw:
        ext_sign = 1
        ext_symb = "max"
        event_year = 2019
    elif "jan2019" == which_ssw:
        ext_sign = -1
        ext_symb = "min"
        event_year = 2019
    elif "feb2018" == which_ssw:
        ext_sign = -1
        ext_symb = "min"
        event_year = 2018

    daily_stat = 'daily_mean'
    boot_type = 'percentile'
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        ds_cgts_extts = dict()
        gevpars = dict()
        for expt in expts:
            ds_cgts_extts[expt] = (
                    xr.open_dataarray(join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day_cgs{cgs_key}.nc'))
                    * ext_sign
                    ).max('time') * ext_sign
            gevpars[expt] = xr.open_dataarray(join(reduced_data_dir,f'gevpar_e{expt}_i{init}_cgs{cgs_key}.nc'))
        ds_cgts_extts['era5'] = (
                xr.open_dataarray(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))
                    * ext_sign
                    ).max('time') * ext_sign
        lon_blocksize,lat_blocksize = ((event_region[d].stop - event_region[d].start)/cgs_level[i_d] for (i_d,d) in enumerate(('lon','lat')))
        if todo['plot_statpar_map_diff'] and min(cgs_level) > 1:
            # control-free, nudged-free
            statsel = dict(daily_stat=daily_stat)
            for (expt0,expt1) in (('free','control'),('free','nudged')):
                fig,axes = pipeline_base.plot_statpar_map_difference(ds_cgts_extts[expt0].sel(statsel),ds_cgts_extts[expt1].sel(statsel), gevpars[expt0].sel(statsel), gevpars[expt1].sel(statsel), locsign=ext_sign)
                fig.suptitle(f'{gcm} ({expt1} - {expt0}), init {datestr}')
                fig.savefig(join(figdir,f'statpar_map_e{expt1}minus{expt0}_i{init}_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
                plt.close(fig)
        if todo['plot_relrisk_map'] and min(cgs_level) > 1:
            # control-free, nudged-free
            statsel = dict(daily_stat=daily_stat)
            for (expt0,expt1) in (('free','control'),('free','nudged')):
                risk0_file,risk1_file = (join(reduced_data_dir,f'risk_e{expt}_i{init}_cgs{cgs_key}.nc') for expt in (expt0,expt1))
                risk0,risk1 = (xr.open_dataarray(risk_file) for risk_file in (risk0_file,risk1_file))
                rr = risk1/risk0
                #logrr = xr.where(np.isfinite(logrr), logrr, np.nan)
                fig,ax = pipeline_base.plot_relative_risk_map(risk0, risk1, locsign=1)
                ax.set_title(f'{gcm} RR ({expt1}/{expt0})\ninit {datestr}')
                fig.savefig(join(figdir,f'relrisk_map_e{expt1}over{expt0}_i{init}_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
                plt.close(fig)
                print(f'Saved the figure')

        if todo['plot_gev_select_regions']:
            for (i_lon,i_lat) in select_regions[i_cgs_level]:
                exttemps = dict()
                gevpar_regs = dict()
                exttemp_levels_regs = dict()
                for expt in expts:
                    exttemps[expt] = ds_cgts_extts[expt].sel(daily_stat=daily_stat).isel(lon=i_lon,lat=i_lat)
                    exttemp_levels_regs[expt] = np.load(join(reduced_data_dir,f'exttemp_levels_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                    gevpar_regs[expt] = xr.open_dataarray(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                exttemps['era5'] = ds_cgts_extts['era5'].sel(daily_stat=daily_stat).isel(lon=i_lon,lat=i_lat)
                exttemp_levels_regs['era5'] = np.load(join(reduced_data_dir_era5,f'exttemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                gevpar_regs['era5'] = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                center_lon,center_lat = (exttemps['era5'].coords[d].item() for d in ('lon','lat'))
                lonlatstr = r'$\lambda=%d\pm%d,\phi=%d\pm%d$'%(center_lon,lon_blocksize/2,center_lat,lat_blocksize/2)
                if max(cgs_level) == 1:
                    lonlatstr = r'%s (whole region)'%(lonlatstr)
                print(f'Plotting GEV select regions')
                # Plot all four curves on one plot (ERA5, free, control, nudged)
                colors = dict({'era5': 'black', 'control': 'dodgerblue', 'free': 'limegreen', 'nudged': 'red'})

                fig,axes = plt.subplots(ncols=2, figsize=(12,4), sharey=True)
                for ax in axes: 
                    ax.axhline(exttemps['era5'].sel(member=event_year).item(), color='black', linestyle='--')
                ax = axes[0]
                handles = []

                # Interpolate for relative risk 
                exttemp_levels_range = [exttemp_levels_regs['era5'][0,:].min().item(),exttemp_levels_regs['era5'][0,:].max().item()] 
                exttemp_levels_common = np.linspace(*exttemp_levels_range, 30)
                risk_at_levels = dict()
                for expt in expts + ['era5']:
                    shape,loc,scale = (gevpar_regs[expt].sel(param=p).isel(boot=0) for p in ('shape','loc','scale'))
                    param_label = ','.join([
                        r'$\mu=%d$'%(-loc),
                        r'$\sigma=%d$'%(scale),
                        r'$\xi=%+.2f$'%(shape)
                        ])
                    exttemp = exttemps[expt].to_numpy()
                    exttemp_levels_reg = exttemp_levels_regs[expt]
                    order = np.argsort(exttemp)
                    rank = np.argsort(order)
                    if ext_sign == -1:
                        risk_empirical = np.arange(1,len(exttemp)+1)/len(exttemp)
                    else:
                        risk_empirical = np.arange(len(exttemp),0,-1)/len(exttemp)
                    ax.scatter(risk_empirical, exttemp[order], color=colors[expt], marker='+')
                    h, = ax.plot(risk_levels,exttemp_levels_reg[0,:],color=colors[expt],label=r'%s (%s)'%(expt+' '*(7-len(expt)),param_label))
                    handles.append(h)
                    boot_quant_lo,boot_quant_hi = (np.quantile(exttemp_levels_reg[1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                    if boot_type == 'percentile':
                        lo,hi = boot_quant_lo,boot_quant_hi
                    elif boot_type == 'basic':
                        lo,hi = (2*exttemp_levels_reg[0,:]-boot_quant_hi,2*exttemp_levels_reg[0,:]-boot_quant_lo)
                    ax.fill_between(risk_levels, lo, hi, fc=colors[expt], ec='none', alpha=0.3, zorder=-1)
                    # Interpolate every bootstrap
                    func = lambda T: np.interp(exttemp_levels_common, T, risk_levels)
                    risk_at_levels[expt] = np.apply_along_axis(func, 1, exttemp_levels_reg)
                ax.set_xscale('log')
                extstr = "min" if ext_sign==-1 else "max"
                ax.set_xlabel(r'$\mathbb{P}\{\%s_t\langle T(t)\rangle_{\mathrm{region}}\leq T\}$'%(extstr))
                ax.set_ylabel(r'$T$')
                ax.legend(handles=handles, bbox_to_anchor=(-0.25,0.5), loc='center right')

                # Relative risk
                ax = axes[1]
                ax.axvline(1.0, color=colors['free'], linestyle='-')
                # Interpolate to common set of levels and plot relative risk 
                handles = []
                for (expt0,expt1) in (('free','control'),('free','nudged')):
                    risk_ratio = risk_at_levels[expt1] / risk_at_levels[expt0]
                    h, = ax.plot(risk_ratio[0,:], exttemp_levels_common, color=colors[expt1], label=r'%s/%s'%(expt1,expt0))
                    handles.append(h)
                    boot_quant_lo,boot_quant_hi = (np.quantile(risk_ratio[1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                    if boot_type == 'percentile':
                        lo,hi = boot_quant_lo,boot_quant_hi
                    else:
                        lo,hi = 2*risk_ratio[0,:]-boot_quant_hi, 2*risk_ratio[0,:]-boot_quant_lo
                    ax.fill_betweenx(exttemp_levels_common, lo, hi, fc=colors[expt1], ec='none', alpha=0.3, zorder=-1)
                ax.set_xscale('log')
                ax.set_xlim([0.5,5.0])
                ax.get_xaxis().set_major_formatter(ticker.NullFormatter())
                ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())
                xticks = [0.5,1.0,2.0,5.0]
                ax.set_xticks(xticks, [r'%g'%(xtick) for xtick in xticks])
                #ax.legend(handles=handles)
                ax.set_xlabel(r'Relative risk w.r.t. free')

                for ax in axes:
                    ax.set_ylim(exttemp_levels_range)

                fig.suptitle(f'{gcm}, init {datestr} at {lonlatstr}')

                fig.savefig(join(figdir,f'riskplot_reg_eall_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                plt.close(fig)
    return




def reduce_gcm(which_ssw,i_gcm,i_expt,i_init):
    # One GCM, one forcing (expt), one initialization (init), multiple coarse-grainings in space 
    todo = dict({
        'coarse_grain_time':           1,
        'plot_t2m_sumstats_map':       1,
        'coarse_grain_space':          1,
        'fit_gev':                     1,
        'plot_statpar_map':            1,
        'compute_risk':                1,
        'plot_risk_map':               1,
        'fit_gev_select_regions':      1,
        'plot_gev_select_regions':     1,
        })
    gcm,expt,init,event_region,event_time_interval,raw_mem_files,mem_labels,reduced_data_dir,reduced_data_dir_era5,figdir,cgs_levels,select_regions,risk_levels,n_boot,confint_width = gcm_workflow(which_ssw,i_gcm,i_expt,i_init)
    fcdate = datetime.datetime.strptime(init,'%Y%m%d')
    datestr = fcdate.strftime("%Y-%m-%d")

    # ------------- Coarse-grain in time (cgt) and space (cgts) -------------
    ens_file_cgt = join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day.nc')
    if todo['coarse_grain_time']:
        ds_cgt = coarse_grain_time(raw_mem_files, mem_labels, event_region, fcdate, event_time_interval)
        ds_cgt.to_netcdf(ens_file_cgt)
    else:
        ds_cgt = xr.open_dataarray(ens_file_cgt)

    if "sep2019" == which_ssw:
        ext_sign = 1
        ext_symb = "max"
        event_year = 2019
    elif "jan2019" == which_ssw:
        ext_sign = -1
        ext_symb = "min"
        event_year = 2019
    elif "feb2018" == which_ssw:
        ext_sign = -1
        ext_symb = "min"
        event_year = 2018

    print(f'{ds_cgt.shape = }')
    ds_cgt_extt = ext_sign * (ext_sign*ds_cgt).max(dim='time')
    # Load ERA5 as well
    era5_file_cgt = join(reduced_data_dir_era5,f't2m_cgt1day.nc')
    ds_cgt_era5 = xr.open_dataarray(era5_file_cgt)
    ds_cgt_extt_era5 = ext_sign * (ext_sign*ds_cgt_era5).max(dim='time')
    if todo['plot_t2m_sumstats_map']:
        for daily_stat in ['daily_mean']:
            fig,axes = pipeline_base.plot_sumstats_map(ds_cgt_extt.sel(daily_stat=daily_stat))
            fig.suptitle(f'{gcm}, {expt}, init {init} {daily_stat}')
            fig.savefig(join(figdir,f't2m_sumstats_map_{daily_stat}_e{expt}_i{init}_cgt1day.png'),**pltkwargs)
            plt.close(fig)

    daily_stat = 'daily_mean'
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        lon_blocksize,lat_blocksize = ((event_region[d].stop - event_region[d].start)/cgs_level[i_d] for (i_d,d) in enumerate(('lon','lat')))
        #if min(cgs_level) > 1: 
        #    lon_blocksize,lat_blocksize = (ds_cgt_era5[d][:2].diff(d).item() for d in ('lon','lat'))
        #else:
        #    lon_blocksize,lat_blocksize = (event_region[d].stop - event_region[d].start for d in ('lon','lat'))
        ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day_cgs{cgs_key}.nc')
        if todo['coarse_grain_space']:
            ds_cgts = pipeline_base.coarse_grain_space(ds_cgt, cgs_level)
            ds_cgts.to_netcdf(ens_file_cgts)
        else:
            ds_cgts = xr.open_dataarray(ens_file_cgts)
        ds_cgts_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))
        # ---- Minimize in time -----------------------------
        ds_cgts_extt = ext_sign * (ext_sign*ds_cgts).max(dim='time')
        ds_cgts_extt_era5 = ext_sign * (ext_sign*ds_cgts_era5).max(dim='time')
        # ----------- Perform GEV fitting (on negative temperature) --------------
        gev_param_file = join(reduced_data_dir,f'gevpar_e{expt}_i{init}_cgs{cgs_key}.nc')
        if todo['fit_gev']:
            gevpar = pipeline_base.fit_gev_exttemp(ds_cgts_extt,ext_sign,method='PWM')
            gevpar.to_netcdf(gev_param_file)
        else:
            gevpar = xr.open_dataarray(gev_param_file)
        gevpar_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_cgs{cgs_key}.nc'))
        # ----------------- Compute risk w.r.t. ERA5 ---------------
        risk_file = join(reduced_data_dir,f'risk_e{expt}_i{init}_cgs{cgs_key}.nc')

        if todo['compute_risk']:
            dskwargs = dict(daily_stat=daily_stat,drop=True)
            risk = pipeline_base.compute_risk(
                    ds_cgts_extt.sel(**dskwargs),
                    ds_cgts_extt_era5.sel(member=event_year,**dskwargs),
                    gevpar.sel(**dskwargs),
                    gevpar_era5.sel(**dskwargs),
                    locsign=ext_sign)
            risk.to_netcdf(risk_file)
        else:
            risk = xr.open_dataarray(risk_file)

        if todo['plot_risk_map'] and min(cgs_level) > 1:
            fig,ax = pipeline_base.plot_risk_map(risk, locsign=ext_sign)
            ax.set_title(f'Risk ({gcm}) \n {expt}, init {datestr})')
            fig.savefig(join(figdir,f'risk_map_e{expt}_i{init}_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
            plt.close(fig)
        if todo['plot_statpar_map'] and min(cgs_level) > 1:
            statsel = dict(daily_stat=daily_stat)
            # Maps as-is
            fig,axes = pipeline_base.plot_statpar_map(ds_cgts_extt.sel(statsel), gevpar.sel(statsel), locsign=ext_sign)
            fig.suptitle(f'{gcm} {expt}, init {datestr}')
            fig.savefig(join(figdir,f'statpar_map_e{expt}_i{init}_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
            plt.close(fig)
            # Differences from ERA5
            print(f'{ds_cgts_extt_era5.shape = }')
            print(f'{ds_cgts_extt.shape = }')
            print(f'{gevpar_era5.shape = }')
            print(f'{gevpar.shape = }')
            fig,axes = pipeline_base.plot_statpar_map_difference(ds_cgts_extt_era5.sel(statsel),ds_cgts_extt.sel(statsel),gevpar_era5.sel(statsel),gevpar.sel(statsel),locsign=ext_sign)
            fig.suptitle(f'({gcm} {expt}, init {datestr}) - ERA5')
            fig.savefig(join(figdir,f'statpar_map_e{expt}_i{init}_cgs{cgs_key}_{daily_stat}_minusera5.png'), **pltkwargs)
            plt.close(fig)

                    
        # Do a more thorough uncertainty quantification at a specific site or region, using bootstrap analysis and goodness-of-fit etc. Maybe parallelize over all sites too

        for (i_lon,i_lat) in select_regions[i_cgs_level]:
            exttemp = ds_cgts_extt.sel(daily_stat=daily_stat).isel(lon=i_lon,lat=i_lat).to_numpy()
            exttemp_era5 = ds_cgts_extt_era5.sel(daily_stat=daily_stat).isel(lon=i_lon,lat=i_lat)
            center_lon,center_lat = exttemp_era5.lon.item(),exttemp_era5.lat.item()
            exttemp_era5 = exttemp_era5.to_numpy()
            lonlatstr = r'$\lambda=%d\pm%d,\phi=%d\pm%d$'%(center_lon,lon_blocksize/2,center_lat,lat_blocksize/2)
            if cgs_level == (1,1):
                lonlatstr = r'%s (whole region)'%(lonlatstr)

            if todo['fit_gev_select_regions']:
                if np.any(np.isnan(exttemp)):
                    raise Exception(f'{exttemp = }')
                gevpar_reg,exttemp_levels_reg = pipeline_base.fit_gev_exttemp_1d_uq(exttemp,risk_levels, ext_sign, method='PWM', n_boot=n_boot)
                gevpar_reg.to_netcdf(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                np.save(join(reduced_data_dir,f'exttemp_levels_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'), exttemp_levels_reg)
            else:
                exttemp_levels_reg = np.load(join(reduced_data_dir,f'exttemp_levels_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                gevpar_reg = xr.open_dataarray(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                # Also load the ERA5 data 
            exttemp_levels_reg_era5 = np.load(join(reduced_data_dir_era5,f'exttemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
            gevpar_reg_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
            print(f'{i_lon = }, {i_lat = }')
            print(f'{gevpar_reg_era5.isel(boot=0) = }')

            if todo['plot_gev_select_regions']:
                fig,axes = plt.subplots(ncols=2,figsize=(12,6),sharey=True)

                # Plot timeseries on right
                ax = axes[1]
                for i_mem in range(ds_cgts.member.size):
                    temp_gcm = ds_cgts.sel(daily_stat=daily_stat).isel(lat=i_lat,lon=i_lon,member=i_mem)
                    #print(f'{temp_gcm.time = }')
                    ax.plot(np.arange(len(temp_gcm)),temp_gcm.values)
                    ax.axhline(ext_sign * np.max(ext_sign*temp_gcm.values),color='red',linestyle='--')
                for i_mem in range(ds_cgts_era5.member.size):
                    temp_era5 = ds_cgts_era5.sel(daily_stat=daily_stat).isel(lat=i_lat,lon=i_lon,member=i_mem).to_numpy()
                    ax.plot(np.arange(len(temp_era5)), temp_era5,color='black')
                    ax.axhline(ext_sign * np.max(ext_sign*temp_era5),color='black',linestyle='--')
                shape,loc,scale = (gevpar_reg.sel(param=p).isel(boot=0) for p in ('shape','loc','scale'))
                shape_era5,loc_era5,scale_era5 = (gevpar_reg_era5.sel(param=p).isel(boot=0) for p in ('shape','loc','scale'))
                param_label = ','.join([
                    r'$\mu=%d$'%(ext_sign*loc),
                    r'$\sigma=%d$'%(scale),
                    r'$\xi=%.2f$'%(shape)
                    ])
                param_label_era5 = ','.join([
                    r'$\mu=%d$'%(ext_sign*loc_era5),
                    r'$\sigma=%d$'%(scale_era5),
                    r'$\xi=%.2f$'%(shape_era5)
                    ])
                ax = axes[0]
                # GCM data
                order = np.argsort(exttemp)
                rank = np.argsort(order)
                if ext_sign == -1:
                    risk_empirical = np.arange(1,len(exttemp)+1)/len(exttemp)
                else:
                    risk_empirical = np.arange(len(exttemp),0,-1)/len(exttemp)
                ax.scatter(risk_empirical, exttemp[order], color='red', marker='+')
                hgcm, = ax.plot(risk_levels,exttemp_levels_reg[0,:],color='red',label=r'%s (%s)'%(gcm,param_label))
                ax.fill_between(risk_levels, np.quantile(exttemp_levels_reg[1:], 0.25, axis=0), np.quantile(exttemp_levels_reg[1:], 0.75, axis=0), fc='red', ec='none', alpha=0.3, zorder=-1)
                # Now ERA5
                order = np.argsort(exttemp_era5)
                rank = np.argsort(order)
                if ext_sign == -1:
                    risk_empirical = np.arange(1,len(exttemp_era5)+1)/len(exttemp_era5)
                else:
                    risk_empirical = np.arange(len(exttemp_era5),0,-1)/len(exttemp_era5)
                ax.scatter(risk_empirical, exttemp_era5[order], color='black', marker='+')
                # Special marker for 2018
                i_mem_event_year = np.where(ds_cgts_extt_era5.member == event_year)[0][0]

                ax.scatter(risk_empirical[rank[i_mem_event_year]], exttemp_era5[i_mem_event_year], color='black', marker='o')
                hera5, = ax.plot(risk_levels,exttemp_levels_reg_era5[0,:],color='black',label=r'ERA5 (%s)'%(param_label_era5))
                ax.fill_between(risk_levels, np.quantile(exttemp_levels_reg_era5, 0.25, axis=0), np.quantile(exttemp_levels_reg_era5, 0.75, axis=0), fc='gray', ec='none', alpha=0.3, zorder=-1)
                ax.legend(handles=[hera5,hgcm])
                ax.set_xscale('log')
                ax.set_xlabel(r'$\mathbb{P}\{\%s_t\langle T(t)\rangle_{\mathrm{region}}\leq T\}$'%(ext_symb))
                ax.set_ylabel(r'$T$')
                ax.set_title(f'{gcm} {expt}, init {datestr} at {lonlatstr}')

                filename = f'riskplot_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'
                print(f'{filename = }')
                fig.savefig(join(figdir,f'riskplot_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                plt.close(fig)
                #sys.exit()
        ds_cgts.close()
    ds_cgt.close()
    return 

if __name__ == "__main__":
    gcms2ignore = [0,1,2,3,5]
    idx_gcms = [i for i in range(11) if i not in gcms2ignore]
    idx_expt = [0,1,2]
    idx_expt_pairs = [(1,0),(1,2)]
    idx_init = [0,1]
    ssws = ['feb2018','jan2019','sep2019'][2:]
    procedures = sys.argv[1:]
    for which_ssw in ssws:
        if 'reduce' in procedures:
            for i_gcm in idx_gcms:
                for i_expt in idx_expt:
                    for i_init in idx_init:
                        reduce_gcm(which_ssw,i_gcm,i_expt,i_init)
        if 'compare_expts' in procedures:
            for i_gcm in idx_gcms:
                for i_init in idx_init:
                    compare_expts(which_ssw, i_gcm, i_init)
        if 'compare_gcms' in procedures:
            compare_gcms(which_ssw, idx_gcms)
