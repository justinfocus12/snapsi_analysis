import numpy as np
import xarray as xr
import pdb
import psutil
from cartopy import crs as ccrs
import netCDF4
from matplotlib import pyplot as plt, rcParams, ticker, colors as mplcolors, patches as mplpatches, gridspec
pltkwargs = dict({
    'bbox_inches': 'tight',
    'pad_inches': 0.2,
    })
rcParams.update({
    'font.family': 'monospace',
    'font.size': 15,
    })
pltkwargs = {"bbox_inches": "tight", "pad_inches": 0.2}
import datetime as dtlib
import sys
from os import listdir, makedirs
from os.path import join, exists, basename
import glob
from scipy.stats import norm as spnorm, genextreme as spgex

# My own modules
from importlib import reload
import utils
from utils import dict2args as dtoa
import pipeline_base
import pipeline_era5

def gcm_multiparams(which_ssw):
    # Sets out all the options for which piece of data to ingest
    gcm2institute = all_gcms_institutes()
    gcms = list(gcm2institute.keys())
    expts = ['control','free','nudged']
    fc_dates,onset_date_nominal,term_date = pipeline_base.dates_of_interest(which_ssw)
    return gcms, expts, fc_dates, onset_date_nominal, term_date 


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


def gcm_workflow(which_ssw, i_gcm, i_expt, i_fc_date, verbose=False):
    # 0. Global constants
    gcms,expts,fc_dates,onset_date,term_date = gcm_multiparams(which_ssw)
    fc_dates,onset_date_nominal,term_date = pipeline_base.dates_of_interest(which_ssw)
    gcm = gcms[i_gcm]
    expt = expts[i_expt]
    fc_date = fc_dates[i_fc_date]
    gcm2institute = all_gcms_institutes()
    onset_date = pipeline_base.least_sensible_onset_date(which_ssw)

    # ----------- Files for each stage of analysis -------------
    # 1. Raw data
    fc_date_abbrv = dtlib.datetime.strftime(fc_date,'%Y%m%d')
    raw_data_dir = join('/badc/snap/data/post-cmip6/SNAPSI', gcm2institute[gcm], gcm, expt, 's'+fc_date_abbrv)
    ens_path_skeleton = join(raw_data_dir,'r*i*p*f*')
    mem_labels = [basename(p) for p in glob.glob(ens_path_skeleton)]
    if "GloSea6" == gcm: # for some reason there's a single odd version file 
        path_skeleton = join(ens_path_skeleton,'6hrPt','tas','g*','v20230403*','*.nc')
    else:
        path_skeleton = join(ens_path_skeleton,'6hrPt','tas','g*','v*','*.nc')
    raw_mem_files = glob.glob(path_skeleton)
    assert len(raw_mem_files) == len(mem_labels)

    # 2. Processed data 
    analysis_date = '2025-05-20'
    processed_data_dir = '/gws/nopw/j04/snapsi/processed/wg2/ju26596'
    reduced_data_dir = join(processed_data_dir,which_ssw,analysis_date,f'{gcm}')
    figdir = join('/home/users/ju26596/snapsi_analysis_figures',which_ssw,analysis_date,gcm)
    makedirs(reduced_data_dir,exist_ok=True)
    makedirs(figdir,exist_ok=True)

    # 3. Files 
    ens_file_cgt = join(reduced_data_dir,'t2m_e%s_i%s_cgt1day.nc'%(expt,fc_date_abbrv))
    ens_files_cgts = []
    ens_files_cgts_extt = []
    gevpar_files = []
    risk_files = []
    valatrisk_files = []
    gevsevlev_files = []
    cgs_levels,select_regions = pipeline_base.analysis_multiparams(which_ssw)
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        ens_files_cgts.append(join(reduced_data_dir,'t2m_e%s_i%s_cgt1day_cgs%dx%d.nc'%(expt,fc_date_abbrv,cgs_level[0],cgs_level[1])))
        ens_files_cgts_extt.append(join(reduced_data_dir,'t2m_e%s_i%s_cgt1day_cgs%dx%d_extt.nc'%(expt,fc_date_abbrv,cgs_level[0],cgs_level[1])))
        gevpar_files.append('gevpar_e%s_i%s_cgs%dx%d.nc'%(expt,fc_date_abbrv,cgs_level[0],cgs_level[1]))
        risk_files.append(join(reduced_data_dir,'risk_e%s_i%s_cgs%dx%d.nc'%(expt,fc_date_abbrv,cgs_level[0],cgs_level[1])))
        valatrisk_files.append(join(reduced_data_dir,'valatrisk_e%s_i%s_cgs%dx%d.nc'%(expt,fc_date_abbrv,cgs_level[0],cgs_level[1])))
        gevsevlev_files.append([])
        for (i_lon,i_lat) in select_regions[i_cgs_level]:
            gevsevlev_files[i_cgs_level].append(join(reduced_data_dir,'gevsevlev_e%s_i%s_cgs%dx%d_ilon%d_ilat%d.nc'%(expt,fc_date_abbrv,cgs_level[0],cgs_level[1],i_lon,i_lat)))



    # bootstrap parameters
    workflow = dict(
            gcm=gcm,
            expt=expt,
            fc_date=fc_date,
            figfile_tag='e%s_i%s'%(expt,fc_date_abbrv),
            figtitle_affix = '%s, %s, FC %s'%(gcm, expt, dtlib.datetime.strftime(fc_date, "%Y/%m/%d")),
            fc_dates=fc_dates,
            fc_date_abbrv=fc_date_abbrv,
            init_date = fc_dates[0],
            onset_date_nominal=onset_date_nominal,
            onset_date=onset_date, 
            term_date=term_date,
            raw_mem_files=raw_mem_files,
            mem_labels=mem_labels,
            reduced_data_dir=reduced_data_dir,
            figdir=figdir,
            cgs_levels=cgs_levels,
            ens_file_cgt = ens_file_cgt, 
            ens_files_cgts = ens_files_cgts,
            ens_files_cgts_extt = ens_files_cgts_extt,
            gevpar_files = gevpar_files,
            risk_files = risk_files,
            valatrisk_files = valatrisk_files,
            gevsevlev_files = gevsevlev_files,
            select_regions=select_regions,
            )
    # Append the applicable items from the ERA5  
    workflow_era5 = pipeline_era5.era5_workflow(which_ssw)
    workflow.update({
        key: workflow_era5[key] for key in map(str.strip, 
            '''
            event_year,
            event_region,
            context_region,
            Nlon_interp,
            Nlat_interp,
            ext_sign,
            ext_symb,
            prob_symb,
            ineq_symb,
            leq_symb,
            landmask_interp_file,
            param_bounds_file,
            daily_stat,
            risk_levels,
            n_boot,
            confint_width,
            boot_type
            '''.split(',')
            )
        })
    return workflow



def coarse_grain_time(raw_mem_files, mem_labels, event_region, context_region, Nlon_interp, Nlat_interp, init_date, term_date, ens_file_cgt, use_dask=False):
    timesel = dict(time=slice(init_date,term_date))
    #region_padded = dict(lat=slice(region['lat'].start-2,region['lat'].stop+2), lon=slice(region['lon'].start-2, region['lon'].stop+2)) 
    preprocess = lambda dsmem: preprocess_gcm_6hrPt(dsmem, init_date, timesel, context_region)
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
    # Interpolate to ERA5 grid
    dlon = (event_region['lon'].stop - event_region['lon'].start)/Nlon_interp
    dlat = (event_region['lat'].stop - event_region['lat'].start)/Nlat_interp
    lon_interp = np.linspace(event_region['lon'].start+dlon/2, event_region['lon'].stop-dlon/2, Nlon_interp)
    lat_interp = np.linspace(event_region['lat'].start+dlat/2, event_region['lat'].stop-dlat/2, Nlat_interp)

    ds_ens = ds_ens.interp(lon=lon_interp, lat=lat_interp, method="linear").sel(event_region)
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
    #ds_ens.close()
    ds_ens_cgt = xr.concat([daily_mean,daily_min,daily_max], dim='daily_stat').assign_coords(daily_stat=['daily_mean','daily_min','daily_max'])
    # also just include the full dataset
    ds = xr.Dataset(data_vars=dict({'1xday': ds_ens_cgt, '4xday': ds_ens.rename({'time': 'time_6h'})}))
    ds.to_netcdf(ens_file_cgt)
    return #ds




def preprocess_gcm_6hrPt(dsmem,fcdate,timesel,spacesel,verbose=False):
    if verbose:
        print(f'{fcdate = }')
        print(f'{dsmem.time = }')
        print(f'{timesel = }')
    dsmem_tas = (
            utils.rezero_lons(
                dsmem['tas']
                .assign_coords(time=np.arange(fcdate,fcdate+dtlib.timedelta(hours=6*dsmem.time.size),dtlib.timedelta(hours=6)))
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
    gcm,expt0,init,event_region,context_region,event_time_interval,landmask_file,raw_mem_files,mem_labels,reduced_data_dir,reduced_data_dir_era5,figdir,cgs_levels,select_regions,risk_levels,n_boot,confint_width = gcm_workflow(i_gcm,i0_expt,i_init)
    _,expt1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = gcm_workflow(i_gcm,i1_expt,i_init)
    ds_cgt_0,ds_cgt_1 = (xr.open_dataarray(join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day.nc')) for expt in (expt0,expt1))
    landmask = (
            utils.rezero_lons(
                xr.open_dataarray(landmask_file)
                .isel(time=0,drop=True)
                .isel(latitude=slice(None,None,-1)) # Flip lat to go in increasing order
                .rename(dict(latitude='lat',longitude='lon'))
                )
            .sel(event_region)
            )
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        ds_cgts_mint_0,ds_cgts_mint_1 = (xr.open_dataarray(join(reduced_data_dir,f't2m_e{expt}_i{init}_cgt1day_cgs{cgs_key}.nc')).sel(daily_stat='daily_mean').min('time') for expt in (expt0,expt1))
        gevpar_0,gevpar_1 = (xr.open_dataarray(join(reduced_data_dir,f'gevpar_e{expt}_i{init}_cgs{cgs_key}.nc')).sel(daily_stat='daily_mean') for expt in (expt0,expt1))
        if todo['plot_param_diff_map']:
            fig_gaussian,axes_gaussian,fig_gev,axes_gev = pipeline_base.plot_statpar_map_difference(ds_cgts_mint_0,ds_cgts_mint_1,gevpar_0,gevpar_1,locsign=-1)
            datestr = dtlib.datetime.strptime(init,"%Y%m%d").strftime("%Y-%m-%d")
            for (fig,distname) in [(fig_gaussian,'gaussian'),(fig_gev,'gev')]:
                fig.suptitle(f'{expt1} - {expt0}, init {datestr}')
                fig.savefig(join(figdir,f'statpar_diffmap_e0{expt0}_e1{expt1}_i{init}_cgs{cgs_key}_{distname}.png'), **pltkwargs)
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
    cgs_levels,select_regions = pipeline_base.analysis_multiparams(which_ssw)
    gcms,expts,inits = gcm_multiparams(which_ssw)
    print(f'{gcms = }')
    colors = dict({'era5': 'black', 'control': 'dodgerblue', 'free': 'limegreen', 'nudged': 'red'})
    yoffsets = dict({'control': 1/3, 'free': 0.0, 'nudged': -1/3})
    figdir = f'/home/users/ju26596/snapsi_analysis_figures/{which_ssw}/2025-05-20/multimodel'
    print(f'{figdir = }')
    xlims = [0.25, 4.0]
    ylims = [-0.5, len(idx_gcms)-0.5]
    slims = (np.array([1.0, 4.0])*rcParams['lines.markersize']) # bounds on the marker size 
    makedirs(figdir, exist_ok=True)
    dpi = 200
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        fig_rr,axes_rr = plt.subplots(figsize=(12,8),ncols=2,nrows=1,sharey=True, sharex=True, dpi=dpi) # relative risk
        fig_dv,axes_dv = plt.subplots(figsize=(12,8),ncols=2,nrows=1,sharey=True, sharex=True, dpi=dpi) # difference in value at risk 
        for ax in axes_rr:
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
        # Determine the radius-to-points conversion
        unit_radius_data = 1/6
        unit_radius_display = axes_rr[0].transData.transform([0, unit_radius_data])[1] - axes_rr[0].transData.transform([0,0])[1]
        unit_radius_points = unit_radius_display * 72/fig_rr.dpi
        handles = []
        i_gcm2plot = -1
        for i_gcm in idx_gcms:
            gcm = gcms[i_gcm]
            print(f'Starting {i_gcm,gcm = }')
            i_gcm2plot += 1
            for (i_init,init) in enumerate(inits):
                ax_rr = axes_rr[i_init]
                ax_dv = axes_dv[i_init]
                workflow = gcm_workflow(which_ssw,i_gcm,0,i_init)
                reduced_data_dir = workflow[9]
                reduced_data_dir_era5 = workflow[10]
                risk_levels = workflow[14]
                confint_width = workflow[16]
                event_region = workflow[3]
                for (expt0,expt1) in (('free','free'),('free','control'),('free','nudged')):
                    print(f'{expt0 = }, {expt1 = }')
                    print(f'{reduced_data_dir = }')
                    risk0_file,risk1_file = (join(reduced_data_dir,f'risk_e{expt}_i{init}_cgs{cgs_key}.nc') for expt in (expt0,expt1))
                    rvar0_file,rvar1_file = (join(reduced_data_dir,f'risk_valatrisk_e{expt}_i{init}_cgs{cgs_key}.nc') for expt in (expt0,expt1))
                    risk0 = xr.open_dataarray(risk0_file).to_numpy()
                    risk1 = xr.open_dataarray(risk1_file).to_numpy()
                    rr = np.maximum(xlims[0], np.minimum(xlims[1], risk1/risk0))
                    rvar0 = xr.open_dataarray(rvar0_file)
                    rvar1 = xr.open_dataarray(rvar1_file)
                    dvar = (rvar1.sel(quantity='valatrisk',drop=True) - rvar0.sel(quantity='valatrisk',drop=True)).to_numpy()
                    #pdb.set_trace()
                    print(f'{rr.shape = }')
                    h = ax_rr.scatter(rr.flat, (i_gcm2plot+yoffsets[expt1])*np.ones(rr.size), ec=colors[expt1], linewidth=2, fc='none', s=(2*unit_radius_points * (1/4 + risk1*3/4))**2, label=expt1)
                    # Add error bars if the level of coarse-graining is 1x1
                    if len(handles) < 3: handles.append(h)
                    h = ax_dv.scatter(dvar.flat, (i_gcm2plot+yoffsets[expt1])*np.ones(dvar.size), ec=colors[expt1], linewidth=2, fc='none', s=(2*unit_radius_points * (1/4 + risk1*3/4))**2, label=expt1)
                    # Add error bars if the level of coarse-graining is 1x1
                    if i_cgs_level == 0:
                        ds_cgts_extts_era5 = (xr.open_dataarray(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc')) * ext_sign).max('time') * ext_sign
                        for i_lon in range(rr.shape[0]):
                            for i_lat in range(rr.shape[1]):
                                #if not np.all([np.all(np.isfinite(ds_cgts_extts[expt].isel(lon=i_lon,lat=i_lat))) for expt in expts+['era5']]):
                                #    continue
                                exttemp_levels_reg_0 = np.load(join(reduced_data_dir,f'exttemp_levels_reg_e{expt0}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                                exttemp_levels_reg_1 = np.load(join(reduced_data_dir,f'exttemp_levels_reg_e{expt1}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                                exttemp_reg_era5 = ds_cgts_extts_era5.sel(member=event_year,daily_stat='daily_mean').isel(lon=i_lon,lat=i_lat).item()
                                if not np.isfinite(exttemp_reg_era5):
                                    print(f'Oops, not finite for {i_lon = }, {i_lat = }')
                                    sys.exit()
                                    continue

                                exttemp_levels_range = tuple(extfun(exttemp_reg_era5).item() for extfun in (np.min, np.max))
                                print(f'{exttemp_levels_range = }')
                                order = np.arange(0,len(risk_levels),dtype=int)
                                if ext_sign == 1:
                                    order = order[::-1]
                                print(f'{order = }')
                                func = lambda T: (np.interp(exttemp_reg_era5, T[order], risk_levels[order]))
                                risk_at_levels_0 = np.apply_along_axis(func, 1, exttemp_levels_reg_0)
                                risk_at_levels_1 = np.apply_along_axis(func, 1, exttemp_levels_reg_1)
                                print(f'{risk_at_levels_0 = }')
                                print(f'{risk_at_levels_1 = }')
                                rel_risk_lo,rel_risk_hi = (np.quantile(risk_at_levels_1[1:]/risk_at_levels_0[1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                                # TODO basic-bootstrap if called for
                                # now same for value at risk 
                                i_risklev = np.argmin(np.abs(risk_levels - rvar0.sel(quantity='risk').isel(lon=i_lon,lat=i_lat).item()))
                                #pdb.set_trace()
                                dvar_los,dvar_his = (np.quantile(exttemp_levels_reg_1[1:,:] - exttemp_levels_reg_0[1:,:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                                dvar_lo = np.interp(rvar0.sel(quantity='risk').isel(lon=i_lon,lat=i_lat).item(), risk_levels, dvar_los)
                                dvar_hi = np.interp(rvar0.sel(quantity='risk').isel(lon=i_lon,lat=i_lat).item(), risk_levels, dvar_his)
                                print(f'{i_init = }, {expt0 = }, {expt1 = }, {i_cgs_level = }, {dvar_lo = }, {dvar_hi = }')
                                print(f'{rel_risk_hi - rel_risk_lo = }')
                                print(f'{dvar_hi - dvar_lo = }')
                                #pdb.set_trace()
                                if False and (expt0 != expt1):
                                    sys.exit()
                                h, = ax_rr.plot([rel_risk_lo,rel_risk_hi], (i_gcm2plot+yoffsets[expt1])*np.ones(2),color=colors[expt1], linewidth=3, label=expt1)
                                h, = ax_dv.plot([dvar_lo,dvar_hi], (i_gcm2plot+yoffsets[expt1])*np.ones(2),color=colors[expt1], linewidth=3, label=expt1)
        # value at risk comparison
        filename = join(figdir, f'dvalatrisk_multimodel_cgs{cgs_key}.png')
        axes_dv[0].set_yticks(range(len(idx_gcms)), labels=[gcms[i] for i in idx_gcms])
        for (i_ax,ax) in enumerate(axes_dv):
            ax.set_ylim(ylims)
            ax.axvline(0.0, color='black', linestyle='--')
            datestr = dtlib.datetime.strptime(inits[i_ax],"%Y%m%d").strftime("%Y-%m-%d")
            ax.set_title(f"init {datestr}")
            ax.set_xlabel(r'$T-T_{\mathrm{free}}$')
            for i_gcm2plot in range(len(idx_gcms)):
                ax.axhline(i_gcm2plot+0.5, color='gray')
                ax.axhline(i_gcm2plot-0.5, color='gray')
                ax.axhline(i_gcm2plot+1/6, color='gray', linestyle='dotted')
                ax.axhline(i_gcm2plot-1/6, color='gray', linestyle='dotted')
        dlon = (event_region['lon'].stop - event_region['lon'].start)/(cgs_level[0])
        dlat = (event_region['lat'].stop - event_region['lat'].start)/(cgs_level[1])
        if i_cgs_level == 0:
            suptitle = r'$\Delta T$ (whole region)'
        else:
            suptitle = r'$\Delta T$ ($%d^\circ$lon$\times%d^\circ$lat regions)'%(dlon, dlat)
        fig_dv.suptitle(suptitle)
        fig_dv.legend(handles=handles, bbox_to_anchor=(0.5,0.0), loc='upper center', ncol=3)
        fig_dv.savefig(filename, **pltkwargs)

        plt.close(fig_dv)
                            
        # relative risk comparison
        filename = join(figdir, f'relrisk_multimodel_cgs{cgs_key}.png')
        axes_rr[0].set_yticks(range(len(idx_gcms)), labels=[gcms[i] for i in idx_gcms])
        for (i_ax,ax) in enumerate(axes_rr):
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.axvline(1.0, color='black', linestyle='--')
            ax.axvline(xlims[0], color='black', linestyle='--')
            ax.axvline(xlims[1], color='black', linestyle='--')
            datestr = dtlib.datetime.strptime(inits[i_ax],"%Y%m%d").strftime("%Y-%m-%d")
            ax.set_title(f"init {datestr}")
            ax.set_xscale('log')
            xticks = [xlims[0], 1.0, xlims[1]]
            ax.set_xticks(xticks, labels=[str(xtick) for xtick in xticks])
            ax.set_xlabel('rel. risk w.r.t. free')
            for i_gcm2plot in range(len(idx_gcms)):
                ax.axhline(i_gcm2plot+0.5, color='gray')
                ax.axhline(i_gcm2plot-0.5, color='gray')
                ax.axhline(i_gcm2plot+1/6, color='gray', linestyle='dotted')
                ax.axhline(i_gcm2plot-1/6, color='gray', linestyle='dotted')
        dlon = (event_region['lon'].stop - event_region['lon'].start)/(cgs_level[0])
        dlat = (event_region['lat'].stop - event_region['lat'].start)/(cgs_level[1])
        if i_cgs_level == 0:
            suptitle = r'Relative risk (whole region)'
        else:
            suptitle = r'Relative risk ($%d^\circ$lon$\times%d^\circ$lat regions)'%(dlon, dlat)
        fig_rr.suptitle(suptitle)
        fig_rr.legend(handles=handles, bbox_to_anchor=(0.5,0.0), loc='upper center', ncol=3)
        fig_rr.savefig(filename, **pltkwargs)

        plt.close(fig_rr)
    return
            


def compare_expts(which_ssw, i_gcm, i_init):
    todo = dict({
        'plot_statpar_map_diff':           1,
        'plot_relrisk_map':                0,
        # TODO add an option for 'plot_qshift_map'
        'plot_gev_select_regions':         0,
        })
    gcms,expts,fc_dates,_,term_date = gcm_multiparams(which_ssw)
    wkfs = dict({
            expt: gcm_workflow(which_ssw,i_gcm,i_expt,i_init)
            for (i_expt,expt) in enumerate(expts)
            })
    expt_pairs = (('free','nudged'),('free','control'))
    if todo['plot_statpar_map_diff']:
        for (expt0,expt1) in expt_pairs:
            # TODO need a system 
            pipeline_base.plot_gevpar_difference_maps_flat(
                    *dtoa(wkfs[expt0], 'gevpar_files,expt,'),
                    *dtoa(wkfs[expt1], 'gevpar_files,expt,'),
                    *dtoa(wkfs[expt0], 'param_bounds_file,ext_sign,cgs_levels,event_region,figdir,gcm,fc_date,fc_date_abbrv')
                    )
    return
    if False:
        for i_cgs_level,cgs_level in enumerate(cgs_levels):
            if min(cgs_level) <= 1:
                continue
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            lon_blocksize,lat_blocksize = ((event_region[d].stop - event_region[d].start)/cgs_level[i_d] for (i_d,d) in enumerate(('lon','lat')))
            da_cgts_extts = dict()
            gevpars = dict()
            for expt in expts:
                da_cgts_extts[expt] = ext_sign * (ext_sign * 
                        xr.open_dataset(join(reduced_data_dir,f't2m_e{expt}_i{fc_date_abbrv}_cgt1day_cgs{cgs_key}.nc'))['1xday']
                        .sel(daily_stat=daily_stat)
                        .sel(time=slice(onset_date,term_date))
                        ).max(dim='time')
                gevpars[expt] = xr.open_dataarray(join(reduced_data_dir,f'gevpar_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.nc'))
            da_cgts_extts['era5'] = ext_sign * (ext_sign * 
                    xr.open_dataset(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))['1xday']
                    .sel(daily_stat=daily_stat)
                    .sel(time=slice(onset_date,term_date))
                    ).max(dim='time')
            # control-free, nudged-free
            for (expt0,expt1) in (('free','control'),('free','nudged')):
                fig,axes = pipeline_base.plot_gevpar_difference_maps_flat(gevpars[expt0], gevpars[expt1], ['dloc','dscale','dshape'], cgs_levels[2], ext_sign=ext_sign)
                fig.savefig(join(figdir,f'gevpar_diff_map_e{expt0}to{expt1}_i{fc_date_abbrv}_cgs{cgs_key}_{daily_stat}'), **pltkwargs)
                plt.close(fig)
                #fig_gaussian,axes_gaussian,fig_gev,axes_gev = pipeline_base.plot_statpar_map_difference(da_cgts_extts[expt0],da_cgts_extts[expt1], gevpars[expt0], gevpars[expt1], locsign=ext_sign)
                #for (fig,distname) in [(fig_gaussian,'gaussian'),(fig_gev,'gev')]:
                #    fig.suptitle(f'{gcm} ({expt1} - {expt0}), init {datestr}')
                #    fig.savefig(join(figdir,f'statpar_map_e{expt1}minus{expt0}_i{init}_cgs{cgs_key}_{daily_stat}_{distname}.png'), **pltkwargs)
                #    plt.close(fig)
    if todo['plot_relrisk_map']:
        print(f'----------------\ngot into relrisk_map block\n---------------')
        fmtfun = lambda date: dtlib.datetime.strftime(date, "%Y/%m/%d")
        for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
            if min(cgs_level) <= 1:
                continue
            cgs_key = "%dx%d"%(cgs_level[0],cgs_level[1])
            print(f' --- %s '%(cgs_key))
            # control/free, nudged/free
            for (expt0,expt1) in (('free','free'),('free','control'),('free','nudged')):
                risk0_file,risk1_file = (join(reduced_data_dir,f'risk_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.nc') for expt in (expt0,expt1))
                risk0,risk1 = (xr.open_dataarray(risk_file) for risk_file in (risk0_file,risk1_file))
                # Interpolate 
                lon_interp,lat_interp = [np.linspace(risk0.coords[c][0].item(), risk0.coords[c][-1].item(), 200) for c in ['lon','lat']]
                risk0,risk1 = [risk.interp(lon=lon_interp, lat=lat_interp, method='linear') for risk in [risk0,risk1]]
                fig,ax = pipeline_base.plot_relative_risk_map_flat(risk0, risk1, event_region, ext_sign=ext_sign, plot_contour_ratio=(expt0!=expt1))
                ineq_symb = "\u2265" if 1==ext_sign else "\u2264"
                title = "%s, FC %s, %s\n\u2119{%s{T2M(t):%s\u2264t\u2264%s}%s(ERA5 value)}"%(gcm, fmtfun(fc_date), expt1, ext_symb, fmtfun(onset_date), fmtfun(term_date), ineq_symb)
                ax.set_title(title, loc='left')
                figfile = join(figdir,f'relrisk_map_e{expt1}over{expt0}_i{fc_date_abbrv}_cgs{cgs_key}_{daily_stat}.png')
                print(f'About to save figfile')
                fig.savefig(figfile, **pltkwargs)
                plt.close(fig)
                print(f'Saved the figure to {figfile}')

    if todo['plot_gev_select_regions']:
        print(f'Plotting GEV select regions')
        fmtfun = lambda date: dtlib.datetime.strftime(date, "%Y/%m/%d")
        # Interpolate every bootstrap
        # exttemp_levels_common has to have the right order!!!
        # risk_levels should be an increasing array from 0 to 1; low risk means very extreme which means very (cold if ext_sign == -1 else warm)
        if -1 == ext_sign:
            ordering_for_interp = np.arange(0,len(risk_levels),1)
        else:
            ordering_for_interp = np.arange(len(risk_levels)-1,-1,-1)
        for (i_cgs_level, cgs_level) in enumerate(cgs_levels):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            # Load the temperature data for all experiments 
            das_cgts = dict()
            das_cgts_extt = dict()
            for expt in expts:
                ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{fc_date_abbrv}_cgt1day_cgs{cgs_key}.nc')
                das_cgts[expt] = xr.open_dataset(ens_file_cgts)['1xday'].sel(daily_stat=daily_stat)
                das_cgts_extt[expt] = ext_sign * (ext_sign*das_cgts[expt].sel(time=slice(onset_date,term_date))).max(dim='time')
            # Load temperature data from ERA5
            das_cgts['era5'] = xr.open_dataset(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))['1xday'].sel(daily_stat=daily_stat)
            das_cgts_extt['era5'] = ext_sign * (ext_sign*das_cgts['era5'].sel(time=slice(onset_date,term_date))).max(dim='time')

            lon_blocksize,lat_blocksize = ((event_region[d].stop - event_region[d].start)/cgs_level[i_d] for (i_d,d) in enumerate(('lon','lat')))
            print(f' --- Starting loop over lons and lats')
            for (i_lon,i_lat) in select_regions[i_cgs_level]:
                if not np.all([np.all(np.isfinite(das_cgts_extt[expt].isel(lon=i_lon,lat=i_lat))) for expt in expts+['era5']]):
                    continue
                exttemps = dict()
                gevpar_regs = dict()
                exttemp_levels_regs = dict()
                for expt in expts:
                    exttemps[expt] = das_cgts_extt[expt].isel(lon=i_lon,lat=i_lat)
                    exttemp_levels_regs[expt] = np.load(join(reduced_data_dir,f'exttemp_levels_reg_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                    gevpar_regs[expt] = xr.open_dataarray(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                exttemps['era5'] = das_cgts_extt['era5'].isel(lon=i_lon,lat=i_lat)
                exttemp_levels_regs['era5'] = np.load(join(reduced_data_dir_era5,f'exttemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                gevpar_regs['era5'] = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                center_lon,center_lat = (exttemps['era5'].coords[d].item() for d in ('lon','lat'))
                lonlatstr = r'$\lambda=%d\pm%d,\phi=%d\pm%d$'%(center_lon,lon_blocksize/2,center_lat,lat_blocksize/2)
                if max(cgs_level) == 1:
                    lonlatstr = r'%s (whole region)'%(lonlatstr)
                # Plot all four curves on one plot (ERA5, free, control, nudged)
                colors = dict({'era5': 'black', 'control': 'dodgerblue', 'free': 'limegreen', 'nudged': 'red'})

                fig,axes = plt.subplots(ncols=2, nrows=2, figsize=(9,9), sharey='row', sharex='col', width_ratios=[3,2], height_ratios=[2,1])
                for ax in axes[0,:]: 
                    ax.axhline(exttemps['era5'].sel(member=event_year).item(), color='black', linestyle='--')
                ax = axes[0,0]
                handles = []

                # Interpolate for relative risk 
                exttemp_levels_range = [exttemp_levels_regs['era5'][0,:].min().item(),exttemp_levels_regs['era5'][0,:].max().item()] 
                exttemp_levels_common = np.linspace(*exttemp_levels_range, 30)
                risk_at_levels = dict()
                risks_empirical = dict()
                exttemps_sorted = dict()
                for expt in expts + ['era5']:
                    shape,loc,scale = (gevpar_regs[expt].sel(param=p).isel(boot=0) for p in ('shape','loc','scale'))
                    if not np.all([np.isfinite(p) for p in [shape,loc,scale]]):
                        risk_at_levels[expt] = np.nan*np.ones_like(exttemp_levels_common)
                        continue
                    param_label = ','.join([
                        r'$%.1f$'%(ext_sign*loc),
                        r'$%.1f$'%(scale),
                        r'$%+.2f$'%(shape)
                        ])
                    exttemp = exttemps[expt].to_numpy()
                    exttemp_levels_reg = exttemp_levels_regs[expt]
                    order = np.argsort(exttemp)
                    exttemps_sorted[expt] = exttemp[order]
                    rank = np.argsort(order)
                    if ext_sign == -1:
                        risks_empirical[expt] = np.arange(1,len(exttemp)+1)/len(exttemp)
                    else:
                        risks_empirical[expt] = np.arange(len(exttemp),0,-1)/len(exttemp)
                    ax.scatter(risks_empirical[expt], exttemps_sorted[expt], color=colors[expt], marker='+')
                    h, = ax.plot(risk_levels,exttemp_levels_regs[expt][0,:],color=colors[expt],label=r'%s (%s)'%(expt+' '*(7-len(expt)),param_label))
                    handles.append(h)
                    boot_quant_lo,boot_quant_hi = (np.quantile(exttemp_levels_regs[expt][1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                    if boot_type == 'percentile':
                        lo,hi = boot_quant_lo,boot_quant_hi
                    elif boot_type == 'basic':
                        lo,hi = (2*exttemp_levels_regs[expt][0,:]-boot_quant_hi,2*exttemp_levels_regs[expt][0,:]-boot_quant_lo)
                    ax.fill_between(risk_levels, lo, hi, fc=colors[expt], ec='none', alpha=0.3, zorder=-1)
                    func = lambda T: np.interp(exttemp_levels_common, T[ordering_for_interp], risk_levels[ordering_for_interp])
                    risk_at_levels[expt] = np.apply_along_axis(func, 1, exttemp_levels_regs[expt])
                axes[1,0].legend(handles=handles, title=r'$(\mu,\sigma,\xi)=$', bbox_to_anchor=(1.0,1.0), loc='upper left')
                ax.set_ylabel(r'$T$')

                # Relative risk
                ax = axes[0,1]
                ax.axvline(1.0, color=colors['free'], linestyle='-')
                # Interpolate to common set of levels and plot relative risk 
                handles = []
                for (expt0,expt1) in (('free','control'),('free','nudged')):
                    risk_ratio = risk_at_levels[expt1] / risk_at_levels[expt0]

                    #pdb.set_trace()

                    h, = ax.plot(risk_ratio[0,:], exttemp_levels_common, color=colors[expt1], label=r'%s/%s'%(expt1,expt0))
                    handles.append(h)
                    boot_quant_lo,boot_quant_hi = (np.quantile(risk_ratio[1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                    if boot_type == 'percentile':
                        lo,hi = boot_quant_lo,boot_quant_hi
                    else:
                        lo,hi = 2*risk_ratio[0,:]-boot_quant_hi, 2*risk_ratio[0,:]-boot_quant_lo
                    ax.fill_betweenx(exttemp_levels_common, lo, hi, fc=colors[expt1], ec='none', alpha=0.3, zorder=-1)

                for ax in axes[0,:]:
                    ax.set_ylim(exttemp_levels_range)
                    ax.tick_params(which="both",labelbottom=True)

                # ------- Lower left: plot difference in temperature-at-risk as a function of risk --------
                ax = axes[1,0]
                for (expt0,expt1) in (('free','free'),('free','control'),('free','nudged')):
                    exttemp_levels_diff = exttemp_levels_regs[expt1] - exttemp_levels_regs[expt0]
                    ax.plot(risk_levels, exttemp_levels_diff[0,:], color=colors[expt1])
                    boot_quant_lo,boot_quant_hi = (np.quantile(exttemp_levels_diff[1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                    if boot_type == 'percentile':
                        lo,hi = boot_quant_lo,boot_quant_hi
                    else:
                        lo,hi = 2*exttemp_levels_diff[0,:]-boot_quant_hi, 2*exttemp_levels_diff[0,:]-boot_quant_lo
                    ax.fill_between(risk_levels, lo, hi, fc=colors[expt1], alpha=0.3, ec='none', zorder=-1)
                    # scatter the empirical relative risk...
                    if len(risks_empirical[expt1]) == len(risks_empirical[expt0]) and np.all(risks_empirical[expt1] == risks_empirical[expt0]) and expt0 != expt1:
                        ax.scatter(risks_empirical[expt1], exttemps_sorted[expt1] - exttemps_sorted[expt0], color=colors[expt1], marker='+')
                ax.set_ylabel(r"$T-T_{\mathrm{free}}$")
                    
                #pdb.set_trace()
                risk_at_level_observed_era5 = np.interp(exttemps['era5'].sel(member=event_year).item(), exttemp_levels_common, risk_at_levels['era5'][0])
                for ax in axes[:,0]:
                    ax.axvline(risk_at_level_observed_era5, color='black', linestyle='dashed') 
                    ax.set_xscale('log')

                for ax in axes[0,1:2]:
                    ax.get_xaxis().set_major_formatter(ticker.NullFormatter())
                    ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())
                    ax.set_xscale('log')
                    ax.set_xlim([0.25,4.0])
                    xticks = np.array([0.25,1.0,4.0])
                    ax.set_xticks(xticks, np.array([r'%g'%(xtick) for xtick in xticks]))
                    #ax.legend(handles=handles)
                    ax.set_xlabel(r'$\mathrm{risk}/\mathrm{risk}_{\mathrm{free}}$')

                #pdb.set_trace()
                extstr = "min" if ext_sign==-1 else "max"
                ineqstr = "leq" if ext_sign==-1 else "geq"
                axes[1,0].set_xlabel(r'$\mathbb{P}\{\%s_t\langle T(t)\rangle_{\mathrm{region}}\%s T\}$'%(extstr,ineqstr))

                fig.suptitle(f'{gcm}, init {datestr} at {lonlatstr}')

                axes[1,1].axis('off')

                fig.savefig(join(figdir,f'riskplot_reg_eall_i{fc_date_abbrv}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                plt.close(fig)
                print(f'------------------ SAVED THE FIG -----------------')
    return

def onset_date_sensitivity_analysis(
        ens_files_cgts, fc_dates, onset_date_nominal, term_date, cgs_levels, ext_sign, figdir,
        ens_files_cgts_era5, event_year, 
        ):
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        da_cgts = xr.open_dataarray(ens_files_cgts[i_cgs_level])
        da_cgts_era5 = xr.open_dataarray(ens_files_cgts_era5[i_cgs_level])
        da_cgts_plus_era5 = xr.concat([
            da_cgts, 
            da_cgts_era5
            .sel(member=slice(event_year,event_year))
            .assign_coords(member=['era5'])
            ], dim='member')
    return
        
        
    



#def onset_date_sensitivity_analysis(which_ssw,i_gcm,i_expt):
#    # Compare onset date severities etc. across different inits 
#    _,_,fc_dates,_,_ = gcm_multiparams(which_ssw)
#    # Figure out the axis limits 
#    for i_cgs_level,cgs_level in enumerate(cgs_levels[:2]):
#        das_cgts = []
#        for (i_fc_date,fc_date) in enumerate(fc_dates):
#            gcm,expt,fc_date,ext_sign,event_region,fc_dates,onset_date_nominal,term_date,landmask_file,raw_mem_files,mem_labels,reduced_data_dir,reduced_data_dir_era5,figdir,cgs_levels,select_regions,risk_levels,n_boot,confint_width = gcm_workflow(which_ssw,i_gcm,i_expt,i_init)
#            datestr = fc_date.strftime("%Y-%m-%d")
#            fc_date_abbrv = dtlib.datetime.strftime(fc_date,'%Y%m%d')
#            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
#            da_cgts = []
#            exptstr = f'e{expt}_i{fc_date_abbrv}_cgt1day_cgs{cgs_key}'
#            ens_file_cgts = join(reduced_data_dir,f't2m_{exptstr}.nc')
#            da_cgts = xr.open_dataset(ens_file_cgts)['1xday']
#            das_cgts.append(da_cgts)
#        
#            onset_date_minsens = pipeline_base.onset_date_sensitivity_analysis(
#                    da_cgts,
#                    event_region,cgs_level, 
#                    fc_dates, fc_date, onset_date_nominal, term_date, ext_sign, 
#                    figdir, exptstr, gcm, mem_special=None, fc_date_special=fc_date
#                    )



def reduce_gcm(which_ssw,i_gcm,i_expt,i_init):
    # One GCM, one forcing (expt), one initialization (init), multiple coarse-grainings in space 
    todo = dict({
        'coarse_grain_time':                0,
        'coarse_grain_space':               0,
        'onset_date_sensitivity_analysis':  0,
        'compute_severities':               0,
        'plot_sumstats_map':                0,
        'fit_gev':                          0,
        'plot_gevpar_map':                  0,
        'compute_risk':                     0,
        'plot_risk_map':                    0,
        'plot_valatrisk_map':               0,
        'fit_gev_select_regions':           0,
        'plot_gev_select_regions':          1,
        })

    # In this main function, specify only the inputs and outputs as files 
    wkf = gcm_workflow(which_ssw,i_gcm,i_expt,i_init)
    wkf_era5 = pipeline_era5.era5_workflow(which_ssw)
    if todo['coarse_grain_time']:
        coarse_grain_time(
                *(wkf[key.strip()] for key in '''
                raw_mem_files,mem_labels,
                event_region,context_region,
                Nlon_interp, Nlat_interp, 
                fc_date,term_date, ens_file_cgt
                '''.split(',')),
                )

    if todo['coarse_grain_space']:
        pipeline_base.coarse_grain_space(
                *(wkf[key.strip()] for key in '''
                ens_file_cgt,ens_files_cgts, cgs_levels, 
                landmask_interp_file,
                event_region
                '''.split(','))
                )
    if todo['onset_date_sensitivity_analysis']:
        pipeline_base.onset_date_sensitivity_analysis( #arg0, arg1) 
                *dtoa(wkf, '''
                ens_files_cgts, event_region, cgs_levels, fc_dates,
                init_date, onset_date_nominal, term_date, daily_stat,
                ext_sign, figdir, gcm, figfile_tag,
                '''),
                *dtoa(wkf_era5, '''
                ens_files_cgts, event_year, 
                '''),
                idx_cgs_levels = [0,1]
                )
    if todo['compute_severities']:
        pipeline_base.compute_severity_from_intensity(
                *dtoa(wkf, '''
                ens_files_cgts, ens_files_cgts_extt, cgs_levels,
                ext_sign, onset_date, term_date, daily_stat,
                landmask_interp_file,
                '''),
                )
    if todo['plot_sumstats_map']:
        mem_special = xr.open_dataset(wkf['ens_files_cgts_extt'][0]).coords['member'][0].item()
        print(f'{mem_special = }')
        pipeline_base.plot_sumstats_maps_flat(
                wkf['ens_files_cgts_extt'], wkf_era5['ens_files_cgts_extt'],
                mem_special, wkf_era5['event_year'],
                *dtoa(wkf,'''
                ext_sign,param_bounds_file,cgs_levels,
                ext_symb,onset_date,term_date,
                figdir, figfile_tag, figtitle_affix,
                '''),
                )
    if todo['fit_gev']:
        pipeline_base.fit_gev_exttemp(
                *(wkf[key.strip()] for key in '''
                ens_files_cgts_extt,gevpar_files,
                ext_sign,cgs_levels
                '''.split(',')),
                method='PWM',
                )
    if todo['plot_gevpar_map']:
        pipeline_base.plot_gevpar_maps_flat(
                *dtoa(wkf, '''
                gevpar_files,
                ext_sign,cgs_levels,param_bounds_file,
                figdir, figfile_tag, figtitle_affix,
                '''),
                )
    if todo['compute_risk']:
        pipeline_base.compute_valatrisk(
                *dtoa(wkf_era5, 'ens_files_cgts_extt, event_year, gevpar_files,'),
                *dtoa(wkf, 'gevpar_files, valatrisk_files, cgs_levels, ext_sign'),
                )
    if todo['plot_risk_map']:
        pipeline_base.plot_risk_or_valatrisk_map(
                *dtoa(wkf, '''
                valatrisk_files, cgs_levels, ext_sign,
                onset_date, term_date, 
                prob_symb, ext_symb, leq_symb, ineq_symb, figtitle_affix, event_year,
                figdir, figfile_tag
                '''),
                True,
                )
    if todo['plot_valatrisk_map']:
        pipeline_base.plot_risk_or_valatrisk_map(
                *dtoa(wkf, '''
                valatrisk_files, cgs_levels, ext_sign,
                onset_date, term_date, 
                prob_symb, ext_symb, leq_symb, ineq_symb, figtitle_affix, event_year,
                figdir, figfile_tag
                '''),
                False,
                )
    if todo['fit_gev_select_regions']:
        pipeline_base.fit_regional_gevsevlev(
                *dtoa(wkf, '''
                ens_files_cgts_extt, gevsevlev_files, risk_levels, 
                cgs_levels, select_regions,
                ext_sign,
                '''),
                )
    if todo['plot_gev_select_regions']:
        pipeline_base.plot_regional_gevsevlev(
                *dtoa(wkf, 'ens_files_cgts_extt, gevsevlev_files'),
                *dtoa(wkf_era5, 'ens_files_cgts_extt, event_year, param_bounds_file'),
                *dtoa(wkf, '''
                cgs_levels, 
                event_region, select_regions,
                boot_type, confint_width,
                figdir, figfile_tag, figtitle_affix, onset_date, term_date, 
                prob_symb, ext_sign, ext_symb, leq_symb, ineq_symb
                '''),
                )
    print(psutil.virtual_memory())
    return 

if __name__ == "__main__":
    gcm2institute = all_gcms_institutes()
    gcms = list(gcm2institute.keys())
    gcms2ignore = ["BCC-CSM2-HR","GLOBO","GEM-NEMO","CanESM5","SPEAR"]

    idx_gcms = [i for i in range(len(gcms)) if ((gcms[i] not in gcms2ignore))]
    idx_gcms = [gcms.index(gcm) for gcm in ['CESM2-CAM6','IFS'][:1]] #,'CESM2-CAM6']]
    print(f'{idx_gcms = }')
    print(f'{[gcms[i] for i in idx_gcms] = }')
    idx_expt = [0,1,2]
    idx_expt_pairs = [(1,0),(1,2)]
    idx_init = [0,1]
    ssws = ['feb2018','jan2019','sep2019'][:1]
    procedures = sys.argv[1:]
    print(f'{procedures = }')
    for which_ssw in ssws:
        print(f'-------------STARTING SSW {which_ssw = }---------------')
        for i_gcm in idx_gcms:
            print(f'-------------- STARTING I_GCM {i_gcm = } -------------')
            for i_init in idx_init:
                print(f'----------------- STARTING INIT {i_init = } ---------------')
                for i_expt in idx_expt:
                    print(f'---------------- STARTING {i_expt = } ---------------------')
                    if 'reduce' in procedures:
                        reduce_gcm(which_ssw,i_gcm,i_expt,i_init)
                        print(f'------------------ finished reduction------------')
                if 'compare_expts' in procedures:
                    print(f'------------- STARTING compare_expts -----------')
                    compare_expts(which_ssw, i_gcm, i_init)
        if 'compare_gcms' in procedures:
            compare_gcms(which_ssw, idx_gcms)
