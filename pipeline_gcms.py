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
import utils
import pipeline_base

def gcm_multiparams(which_ssw):
    # Sets out all the options for which piece of data to ingest
    gcm2institute = all_gcms_institutes()
    gcms = list(gcm2institute.keys())
    expts = ['control','free','nudged']
    fc_dates,onset_date_nominal,term_date = pipeline_base.dates_of_interest(which_ssw)
    return gcms, expts, fc_dates, onset_date_nominal, term_date 

def analysis_multiparams(which_ssw):
    # lon/lat ratios are 6/1 at the bottom and 4/1 at the top; stick to 5/1 
    if "feb2018" == which_ssw:
        cgs_levels = [(1,1),(5,1),(10,2),(20,4),(40,8),(80,16)]
        select_regions = ( # Indexed by cgs_level
                ((0,0),), # level (1,1)
                (), # level (5,1)
                ((2*i,2*i//5) for i in range(5)),
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


def gcm_workflow(which_ssw, i_gcm, i_expt, i_fc_date, verbose=False):
    # Sets out the folders necessary to ingest a chunk of data specified by the input arguments 
    gcms,expts,fc_dates,onset_date,term_date = gcm_multiparams(which_ssw)
    gcm = gcms[i_gcm]
    expt = expts[i_expt]
    fc_date = fc_dates[i_fc_date]

    gcm2institute = all_gcms_institutes()

    # ----------- Files for each stage of analysis -------------
    # 1. Raw data
    fc_date_abbrv = dtlib.datetime.strftime(fc_date,'s%Y%m%d')
    raw_data_dir = join('/badc/snap/data/post-cmip6/SNAPSI', gcm2institute[gcm], gcm, expt, fc_date_abbrv)
    landmask_file = '/gws/nopw/j04/snapsi/processed/wg2/ju26596/era5/land_sea_mask.nc'
    print(f'{raw_data_dir = }')
    ens_path_skeleton = join(raw_data_dir,'r*i*p*f*')
    mem_labels = [basename(p) for p in glob.glob(ens_path_skeleton)]
    #print(f'{mem_labels = }')
    #print(f'{len(mem_labels) = }')

    if "GloSea6" == gcm: # for some reason there's a single odd version file 
        path_skeleton = join(ens_path_skeleton,'6hrPt','tas','g*','v20230403*','*.nc')
    else:
        path_skeleton = join(ens_path_skeleton,'6hrPt','tas','g*','v*','*.nc')
    print(f'{path_skeleton = }')
    raw_mem_files = glob.glob(path_skeleton)
    print(f'{len(raw_mem_files) = }')
    #for rmf in raw_mem_files:
    #    print(rmf)
    assert len(raw_mem_files) == len(mem_labels)
    
    daily_stat = 'daily_min'

    analysis_date = '2025-05-20'
    fc_dates,onset_date_nominal,term_date = pipeline_base.dates_of_interest(which_ssw)
    event_region,context_region = pipeline_base.region_of_interest(which_ssw)
    # 2. Spatiotemporal sub-selection and coarse-graining (cg)
    if "feb2018" == which_ssw:
        event_year = 2018
        ext_sign = -1
    elif "jan2019" == which_ssw:
        event_year = 2019
        ext_sign = -1
    elif "sep2019" == which_ssw:
        event_year = 2019
        ext_sign = 1

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
            fc_date,
            ext_sign,
            event_year,
            event_region,
            context_region,
            daily_stat,
            fc_dates,onset_date_nominal,term_date,
            landmask_file,
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

def visualize_ensemble_spread(raw_mem_files,):
    return


def coarse_grain_time(raw_mem_files, mem_labels, region, init_date, term_date, use_dask=False):
    timesel = dict(time=slice(init_date,term_date))
    region_padded = dict(lat=slice(region['lat'].start-2,region['lat'].stop+2), lon=slice(region['lon'].start-2, region['lon'].stop+2)) 
    preprocess = lambda dsmem: preprocess_gcm_6hrPt(dsmem, init_date, timesel, region_padded)
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
    #ds_ens.close()
    ds_ens_cgt = xr.concat([daily_mean,daily_min,daily_max], dim='daily_stat').assign_coords(daily_stat=['daily_mean','daily_min','daily_max'])
    # also just include the full dataset
    ds = xr.Dataset(data_vars=dict({'1xday': ds_ens_cgt, '4xday': ds_ens.rename({'time': 'time_6h'})}))
    return ds




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
    cgs_levels,select_regions = analysis_multiparams(which_ssw)
    gcms,expts,inits = gcm_multiparams(which_ssw)
    print(f'{gcms = }')
    colors = dict({'era5': 'black', 'control': 'dodgerblue', 'free': 'limegreen', 'nudged': 'red'})
    yoffsets = dict({'control': 1/3, 'free': 0.0, 'nudged': -1/3})
    figdir = f'/home/users/ju26596/snapsi_analysis_figures/{which_ssw}/2025-05-20/multimodel'
    print(f'{figdir = }')
    xlims = [0.2, 5.0]
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
    expts = []
    gcms,expts,inits = gcm_multiparams(which_ssw)
    #for i_expt in range(3):
    gcm,_,init,event_region,context_region,event_time_interval,landmask_file,_,_,reduced_data_dir,reduced_data_dir_era5,figdir,cgs_levels,select_regions,risk_levels,n_boot,confint_width = gcm_workflow(which_ssw, i_gcm,0,i_init)
    landmask = (
            utils.rezero_lons(
                xr.open_dataarray(landmask_file)
                .isel(time=0,drop=True)
                .isel(latitude=slice(None,None,-1)) # Flip lat to go in increasing order
                .rename(dict(latitude='lat',longitude='lon'))
                )
            .sel(event_region)
            )
    datestr = dtlib.datetime.strptime(init,"%Y%m%d").strftime("%Y-%m-%d")

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

    daily_stat = 'daily_min'
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
                fig_gaussian,axes_gaussian,fig_gev,axes_gev = pipeline_base.plot_statpar_map_difference(ds_cgts_extts[expt0].sel(statsel),ds_cgts_extts[expt1].sel(statsel), gevpars[expt0].sel(statsel), gevpars[expt1].sel(statsel), locsign=ext_sign)
                for (fig,distname) in [(fig_gaussian,'gaussian'),(fig_gev,'gev')]:
                    fig.suptitle(f'{gcm} ({expt1} - {expt0}), init {datestr}')
                    fig.savefig(join(figdir,f'statpar_map_e{expt1}minus{expt0}_i{init}_cgs{cgs_key}_{daily_stat}_{distname}.png'), **pltkwargs)
                    plt.close(fig)
        if todo['plot_relrisk_map'] and min(cgs_level) > 1:
            print(f'----------------\ngot into relrisk_map block\n---------------')
            # control-free, nudged-free
            statsel = dict(daily_stat=daily_stat)
            for (expt0,expt1) in (('free','control'),('free','nudged')):
                risk0_file,risk1_file = (join(reduced_data_dir,f'risk_e{expt}_i{init}_cgs{cgs_key}.nc') for expt in (expt0,expt1))
                risk0,risk1 = (xr.open_dataarray(risk_file) for risk_file in (risk0_file,risk1_file))
                rr = risk1/risk0
                #logrr = xr.where(np.isfinite(logrr), logrr, np.nan)
                fig,ax = pipeline_base.plot_relative_risk_map(risk0, risk1, locsign=1)
                ax.set_title(f'{gcm} RR ({expt1}/{expt0})\ninit {datestr}')
                figfile = join(figdir,f'relrisk_map_e{expt1}over{expt0}_i{init}_cgs{cgs_key}_{daily_stat}.png')
                print(f'About to save figfile')
                fig.savefig(figfile, **pltkwargs)
                plt.close(fig)
                print(f'Saved the figure to {figfile}')

        if todo['plot_gev_select_regions']:
            # Interpolate every bootstrap
            # exttemp_levels_common has to have the right order!!!
            if -1 == ext_sign:
                ordering_for_interp = np.arange(0,len(risk_levels),1)
            else:
                ordering_for_interp = np.arange(len(risk_levels)-1,-1,-1)
            for (i_lon,i_lat) in select_regions[i_cgs_level]:
                if not np.all([np.all(np.isfinite(ds_cgts_extts[expt].isel(lon=i_lon,lat=i_lat))) for expt in expts+['era5']]):
                    continue
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
                    ax.set_xlim([0.2,5.0])
                    xticks = np.array([0.2,1.0,5.0])
                    ax.set_xticks(xticks, np.array([r'%g'%(xtick) for xtick in xticks]))
                    #ax.legend(handles=handles)
                    ax.set_xlabel(r'$\mathrm{risk}/\mathrm{risk}_{\mathrm{free}}$')

                #pdb.set_trace()
                extstr = "min" if ext_sign==-1 else "max"
                ineqstr = "leq" if ext_sign==-1 else "geq"
                axes[1,0].set_xlabel(r'$\mathbb{P}\{\%s_t\langle T(t)\rangle_{\mathrm{region}}\%s T\}$'%(extstr,ineqstr))

                fig.suptitle(f'{gcm}, init {datestr} at {lonlatstr}')

                axes[1,1].axis('off')

                fig.savefig(join(figdir,f'riskplot_reg_eall_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                plt.close(fig)
                print(f'------------------ SAVED THE FIG -----------------')
    return

#def get_intensity_bounds(which_ssw,i_gcm,i_expt,i_cgs_level,i_lon,i_lat):
#    _,_,fc_dates,_,_ = gcm_multiparams(which_ssw)
#    intens_min,intens_max = np.inf, -np.inf


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
        'plot_t2m_sumstats_map':            0,
        'fit_gev':                          0,
        'plot_gevpar_map':                  0,
        'compute_risk':                     0,
        'plot_risk_map':                    0,
        'compute_valatrisk':                0,
        'plot_valatrisk_map':               0,
        'fit_gev_select_regions':           0,
        'plot_gev_select_regions':          1,
        })
    _,_,fc_dates,_,_ = gcm_multiparams(which_ssw)
    (
            gcm,
            expt,
            fc_date,
            ext_sign,
            event_year,
            event_region,
            context_region,
            daily_stat,
            fc_dates,onset_date_nominal,term_date,
            landmask_file,
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
            ) = gcm_workflow(which_ssw,i_gcm,i_expt,i_init)
    ext_symb = "max" if 1==ext_sign else "min"
    ineq_sign = "geq" if 1==ext_sign else "leq"

    datestr = fc_date.strftime("%Y-%m-%d")
    fc_date_abbrv = dtlib.datetime.strftime(fc_date,'%Y%m%d')

    # ------------- Coarse-grain in time (cgt) and space (cgts) -------------
    # Load ERA5 as well
    era5_file_cgt = join(reduced_data_dir_era5,f't2m_cgt1day.nc')
    ds_cgt_era5 = xr.open_dataset(era5_file_cgt)
    ens_file_cgt = join(reduced_data_dir,f't2m_e{expt}_i{fc_date_abbrv}_cgt1day.nc')
    if todo['coarse_grain_time']:
        # the coarse_grain_time function deliberately pads the event region so it can be interpolated without nans
        ds_cgt = coarse_grain_time(raw_mem_files, mem_labels, context_region, fc_date, term_date)
        #pdb.set_trace()
        ds_cgt = ds_cgt.interp({'lon': ds_cgt_era5['lon'], 'lat': ds_cgt_era5['lat']})
        #pdb.set_trace()
        ds_cgt.to_netcdf(ens_file_cgt)
    else:
        ds_cgt = xr.open_dataset(ens_file_cgt)
    # INTERPOLATE 
    landmask_full = (
            utils.rezero_lons(
                xr.open_dataarray(landmask_file)
                .isel(time=0,drop=True)
                .isel(latitude=slice(None,None,-1)) # Flip lat to go in increasing order
                .rename(dict(latitude='lat',longitude='lon'))
                )
            #.sel(event_region)
            )
    landmask = landmask_full.interp({'lat': ds_cgt.coords['lat'].values, 'lon': ds_cgt.coords['lon'].values}).sel(event_region)
    assert np.all(np.isfinite(landmask))

    

    # Ensure everything is interpolated right 
    coords_agree = True
    for coordname in ('lat','lon'):
        for ds in (ds_cgt, ds_cgt_era5):
            coords_agree *= np.all(ds[coordname].values == landmask[coordname].values)
    assert coords_agree

    print(f'************ About to coarse-grain space *******')


    if todo['coarse_grain_space']:
        for i_cgs_level,cgs_level in enumerate(cgs_levels):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{fc_date_abbrv}_cgt1day_cgs{cgs_key}.nc')
            ds_cgts = pipeline_base.coarse_grain_space(ds_cgt, cgs_level, landmask)
            ds_cgts.to_netcdf(ens_file_cgts)

    # ------------- Sensitivity analysis with respect to onset date ------------
    print(f'************ About to do ODSA *******')
    if todo['onset_date_sensitivity_analysis']:
        for i_cgs_level,cgs_level in enumerate(cgs_levels[:2]):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            da_cgts_era5 = xr.open_dataset(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))['1xday'].sel(daily_stat=daily_stat)
            exptstr = f'e{expt}_i{fc_date_abbrv}_cgt1day_cgs{cgs_key}'
            ens_file_cgts = join(reduced_data_dir,f't2m_{exptstr}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday'].sel(daily_stat=daily_stat)

            # TODO replicate conventions of pipeline_era5 around ds_cgt vs ds_cgts etc/ 

            da_cgts_plus_era5 = xr.concat([
                da_cgts
                .assign_coords(lon=da_cgts_era5['lon'],lat=da_cgts_era5['lat'])
                ,
                da_cgts_era5
                .sel(member=slice(event_year,event_year))
                .assign_coords(member=['era5'])
                ], dim='member')
            e5min,e5max = np.nanmin(da_cgts_era5),np.nanmax(da_cgts_era5)
            e5mid = 0.5*(e5min+e5max)
            vmin,vmax = e5mid + 1.25*np.array([-1,1])*(e5max-e5min)/2
            figtitle_prefix = f'{gcm} {expt}'
            onset_date_minsens = pipeline_base.onset_date_sensitivity_analysis(
                    da_cgts_plus_era5,
                    event_region,cgs_level, 
                    fc_dates, fc_date, onset_date_nominal, term_date, ext_sign, 
                    figdir, exptstr, figtitle_prefix, mem_special='era5', fc_date_special=fc_date,
                    intensity_lims=[vmin,vmax]
                    )
    # choose an onset date based on this 
    # --------------------------------------------------------------------------
    onset_date = pipeline_base.least_sensible_onset_date(which_ssw)

    # Load ERA5 for reference 
    da_cgt_era5 = ds_cgt_era5['1xday'].sel(daily_stat=daily_stat)
    da_cgt_extt_era5 = ext_sign * (ext_sign*da_cgt_era5).max(dim='time')
    #da_cgt_extt_era5 = ext_sign * (ext_sign*ds_cgt_era5['1xday'].sel(daily_stat=daily_stat).sel(time=slice(onset_date,term_date))).max(dim='time')
    da_cgt_extt = ext_sign * (ext_sign*ds_cgt['1xday'].sel(daily_stat=daily_stat).sel(time=slice(onset_date,term_date))).max(dim='time')
    param_bounds = dict({
        'loc': utils.padded_bounds(ext_sign*da_cgt_extt_era5.where(landmask>0, np.nan).mean(dim='member')),
        'scale': utils.padded_bounds(da_cgt_extt_era5.where(landmask>0, np.nan).std(dim='member')),
        'shape': np.array([-0.5,0.1]),
        })

    print(f'********** About to plot_t2m_sumstats_map ***********')
    if todo['plot_t2m_sumstats_map']:
        fmtfun = lambda date: dtlib.datetime.strftime(date, "%Y/%m/%d")
        mem_special = da_cgt_extt['member'][0].item()
        mem_special_ref = event_year
        titles = [
                r"%s, %s, FC %s, $\%s \{\text{T2M}(t): %s\leq t\leq%s\}$, %d-member mean"%(
                    gcm, expt, fmtfun(fc_date), ext_symb, fmtfun(onset_date), fmtfun(term_date), da_cgt_extt['member'].size),
                r"%s, %s, FC %s, $\%s \{\text{T2M}(t): %s\leq t\leq%s\}$, %d-member std. dev."%(
                    gcm, expt, fmtfun(fc_date), ext_symb, fmtfun(onset_date), fmtfun(term_date), da_cgt_extt['member'].size),
                r"%s, %s, FC %s, $\%s \{\text{T2M}(t): %s\leq t\leq%s\}$, member %s standardized anomaly"%(
                    gcm, expt, fmtfun(fc_date), ext_symb, fmtfun(onset_date), fmtfun(term_date), mem_special)
                ]
        fig,axes = pipeline_base.plot_sumstats_maps_flat(
                da_cgt_extt,
                da_cgt_extt_era5,
                landmask,
                mem_special, mem_special_ref,
                titles, 
                cgs_levels[2],
                param_bounds=param_bounds,
                ext_sign=ext_sign,
                )
        fig.savefig(join(figdir,f'sumstats_map_{daily_stat}_e{expt}_i{fc_date_abbrv}_cgt1day.png'), **pltkwargs)
        plt.close(fig)
    print(f'********** About to fit_gev ***********')
    if todo['fit_gev']:
        # Un-coarsened
        gevpar_cgt = pipeline_base.fit_gev_exttemp(da_cgt_extt, ext_sign, method='PWM')
        gevpar_file_cgt = join(reduced_data_dir,f'gevpar_e{expt}_i{fc_date_abbrv}.nc')
        gevpar_cgt.to_netcdf(gevpar_file_cgt)

        # Coarsened, regional
        for i_cgs_level,cgs_level in enumerate(cgs_levels):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{fc_date_abbrv}_cgt1day_cgs{cgs_key}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday'].sel(daily_stat=daily_stat)
            da_cgts_era5 = xr.open_dataset(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))['1xday'].sel(daily_stat=daily_stat)
            # ---- Minimize in time -----------------------------
            da_cgts_extt = ext_sign * (ext_sign*da_cgts.sel(time=slice(onset_date,term_date))).max(dim='time')
            da_cgts_extt_era5 = ext_sign * (ext_sign*da_cgts_era5.sel(time=slice(onset_date,term_date))).max(dim='time')
            # ----------- Perform GEV fitting  --------------
            gev_param_file = join(reduced_data_dir,f'gevpar_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.nc')
            gevpar = pipeline_base.fit_gev_exttemp(da_cgts_extt,ext_sign,method='PWM')
            gevpar.to_netcdf(gev_param_file)

    print(f'********** About to plot_gevpar_map ***********')
    if todo['plot_gevpar_map']:
        for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
            if min(cgs_level) <= 1:
                continue
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            gevpar_file_cgts = join(reduced_data_dir,f'gevpar_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.nc')
            gevpar_cgts_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_cgt1xday_cgs{cgs_key}.nc'))
            gevpar_cgts = xr.open_dataarray(gevpar_file_cgts)
            fmtfun = lambda date: dtlib.datetime.strftime(date, "%m/%d")
            titles = [
                    r"%s, %s, FC %s, %s {T2M(t): %s$\leq t\leq$%s}, location $\mu$"%(gcm, expt, fmtfun(fc_date), ext_symb, fmtfun(onset_date), fmtfun(term_date)),
                    r"%s, %s, FC %s, %s {T2M(t): %s$\leq t\leq$%s}, scale $\sigma$"%(gcm, expt, fmtfun(fc_date), ext_symb, fmtfun(onset_date), fmtfun(term_date)),
                    r"%s, %s, FC %s, %s {T2M($t$): %s$\leq t\leq$%s}, shape $\xi$"%(gcm, expt, fmtfun(fc_date), ext_symb, fmtfun(onset_date), fmtfun(term_date))
                    ]
            fig,axes = pipeline_base.plot_gevpar_maps_flat(
                    gevpar_cgts,
                    titles,
                    cgs_levels[2],
                    ext_sign,
                    landmask=landmask if i_cgs_level==0 else None,
                    param_bounds=param_bounds,
                    )
            fig.savefig(join(figdir,f'gevpar_map_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.png'), **pltkwargs)
            plt.close(fig)
        
    # ----------------- Compute risk w.r.t. ERA5 ---------------
    print(f'********** About to compute_risk ***********')
    if todo['compute_risk']:
        for (i_cgs_level,cgs_level) in enumerate(cgs_levels): 
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            # Load the ensemble data
            ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{fc_date_abbrv}_cgt1day_cgs{cgs_key}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday'].sel(daily_stat=daily_stat)
            da_cgts_era5 = xr.open_dataset(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))['1xday'].sel(daily_stat=daily_stat)
            da_cgts_extt = ext_sign * (ext_sign*da_cgts.sel(time=slice(onset_date,term_date))).max(dim='time')
            da_cgts_extt_era5 = ext_sign * (ext_sign*da_cgts_era5.sel(time=slice(onset_date,term_date))).max(dim='time')
            # Load the computed GEV parameters
            gevpar_file_cgts = join(reduced_data_dir,f'gevpar_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.nc')
            gevpar_cgts_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_cgt1xday_cgs{cgs_key}.nc'))
            gevpar_cgts = xr.open_dataarray(gevpar_file_cgts)
            # Specify the output file for risk
            risk_file = join(reduced_data_dir,f'risk_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.nc')
            # compute the risk
            risk = pipeline_base.compute_risk(
                    da_cgts_extt,
                    da_cgts_extt_era5.sel(member=event_year),
                    gevpar_cgts,
                    gevpar_cgts_era5,
                    locsign=ext_sign)
            risk.to_netcdf(risk_file)
    print(f'********** About to plot_risk_map ***********')
    if todo['plot_risk_map']: 
        for (i_cgs_level,cgs_level) in enumerate(cgs_levels): 
            if min(cgs_level) == 1:
                continue
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            risk_file = join(reduced_data_dir,f'risk_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.nc')
            risk = xr.open_dataarray(risk_file)
            fig,ax = pipeline_base.plot_risk_or_valatrisk_map(risk, is_risk=True, locsign=ext_sign)
            fmtfun = lambda date: dtlib.datetime.strftime(date, "%m/%d")
            ineq_symb = "\u2265" if 1==ext_sign else "\u2264"
            title = "%s, %s, FC %s, P{%s{T2M(t):%s\u2264t\u2264%s}%s(ERA5 value)}"%(gcm, expt, fmtfun(fc_date), ext_symb, fmtfun(onset_date), fmtfun(term_date), ineq_symb)
            ax.set_title(title, loc='left')
            fig.savefig(join(figdir,f'risk_map_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
            plt.close(fig)

        # TODO compute quantile shift associated with the same probability as the observed event was relative to the historical record
    print(f'********** About to compute_valatrisk ***********')
    if todo['compute_valatrisk']:
        for (i_cgs_level,cgs_level) in enumerate(cgs_levels): 
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            # Load the pre-computed stuff
            ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{fc_date_abbrv}_cgt1day_cgs{cgs_key}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday'].sel(daily_stat=daily_stat)
            da_cgts_era5 = xr.open_dataset(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))['1xday'].sel(daily_stat=daily_stat)
            da_cgts_extt = ext_sign * (ext_sign*da_cgts.sel(time=slice(onset_date,term_date))).max(dim='time')
            da_cgts_extt_era5 = ext_sign * (ext_sign*da_cgts_era5.sel(time=slice(onset_date,term_date))).max(dim='time')
            gevpar_cgts_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_cgt1xday_cgs{cgs_key}.nc'))
            gevpar_cgts = xr.open_dataarray(join(reduced_data_dir,f'gevpar_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.nc'))
            # Specify output file 
            risk_valatrisk_file = join(reduced_data_dir,f'risk_valatrisk_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.nc')
            risk_valatrisk = pipeline_base.compute_valatrisk(
                    da_cgts_extt, 
                    da_cgts_extt_era5.sel(member=event_year),
                    gevpar_cgts,
                    gevpar_cgts_era5,
                    locsign=ext_sign)
            risk_valatrisk.to_netcdf(risk_valatrisk_file)
            print(f'Computed valatrisk')
    print(f'********** About to plot_valatrisk_map ***********')
    if todo['plot_valatrisk_map']:
        for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
            if min(cgs_level) <= 1:
                continue
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            # Plot a map of the severity occurring at the same frequency that 2018 appeared within ERA5
            fmtfun = lambda date: dtlib.datetime.strftime(date, "%m/%d")
            risk_valatrisk = xr.open_dataarray(join(reduced_data_dir,f'risk_valatrisk_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}.nc'))
            fig,ax = pipeline_base.plot_risk_or_valatrisk_map(risk_valatrisk.sel(quantity='valatrisk',drop=True), is_risk=False, locsign=ext_sign)
            title = "%s, %s, FC %s, severity [K] at ERA5-equivalent risk"%(gcm, expt, fmtfun(fc_date),)
            ax.set_title(title, loc='left')
            fig.savefig(join(figdir,f'valatrisk_map_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
            plt.close(fig)

    print(f'********** About to fit_gev_select_regions***********')
    if todo['fit_gev_select_regions']:
        # Do a more thorough uncertainty quantification at a specific site or region, using bootstrap analysis and goodness-of-fit etc. Maybe parallelize over all sites too
        print(f'Starting loop over select regions')

        for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{fc_date_abbrv}_cgt1day_cgs{cgs_key}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday'].sel(daily_stat=daily_stat)
            da_cgts_era5 = xr.open_dataset(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))['1xday'].sel(daily_stat=daily_stat)
            da_cgts_extt = ext_sign * (ext_sign*da_cgts.sel(time=slice(onset_date,term_date))).max(dim='time')
            da_cgts_extt_era5 = ext_sign * (ext_sign*da_cgts_era5.sel(time=slice(onset_date,term_date))).max(dim='time')
            lon_blocksize,lat_blocksize = ((event_region[d].stop - event_region[d].start)/cgs_level[i_d] for (i_d,d) in enumerate(('lon','lat')))
            for (i_lon,i_lat) in select_regions[i_cgs_level]:
                # Load the ERA5 GEV results  
                print(f'About to load era5 stuff')
                exttemp_levels_reg_era5 = np.load(join(reduced_data_dir_era5,f'exttemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                gevpar_reg_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                print(f'{exttemp_levels_reg_era5.shape = }')
                print(f'{gevpar_reg_era5.shape = }')
                print(f'{i_lon = }, {i_lat = }')
                exttemp = da_cgts_extt.isel(lon=i_lon,lat=i_lat).to_numpy()
                exttemp_era5 = da_cgts_extt_era5.isel(lon=i_lon,lat=i_lat)
                if not (np.all(np.isfinite(exttemp)) and np.all(np.isfinite(exttemp_era5))):
                    continue
                center_lon = event_region['lon'].start + (i_lon+0.5)*lon_blocksize
                center_lat = event_region['lat'].start + (i_lat+0.5)*lat_blocksize
                lonlatstr = r'$\lambda=%d\pm%d,\phi=%d\pm%d$'%(center_lon,lon_blocksize/2,center_lat,lat_blocksize/2)
                exttemp_era5 = exttemp_era5.to_numpy()
                if cgs_level == (1,1):
                    lonlatstr = r'%s (whole region)'%(lonlatstr)

                gevpar_reg,exttemp_levels_reg = pipeline_base.fit_gev_exttemp_1d_uq(exttemp,risk_levels, ext_sign, method='PWM', n_boot=n_boot)
                gevpar_reg.to_netcdf(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                np.save(join(reduced_data_dir,f'exttemp_levels_reg_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'), exttemp_levels_reg)
                #exttemp_levels_reg = np.load(join(reduced_data_dir,f'exttemp_levels_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                #gevpar_reg = xr.open_dataarray(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{init}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))


    print(f'********** About to plot_gev_select_regions ***********')
    if todo['plot_gev_select_regions']:
        for (i_cgs_level,cgs_level) in enumerate(cgs_levels[:1]):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            ens_file_cgts = join(reduced_data_dir,f't2m_e{expt}_i{fc_date_abbrv}_cgt1day_cgs{cgs_key}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday'].sel(daily_stat=daily_stat)
            da_cgts_era5 = xr.open_dataset(join(reduced_data_dir_era5,f't2m_cgt1day_cgs{cgs_key}.nc'))['1xday'].sel(daily_stat=daily_stat)
            da_cgts_extt = ext_sign * (ext_sign*da_cgts.sel(time=slice(onset_date,term_date))).max(dim='time')
            da_cgts_extt_era5 = ext_sign * (ext_sign*da_cgts_era5.sel(time=slice(onset_date,term_date))).max(dim='time')
            lon_blocksize,lat_blocksize = ((event_region[d].stop - event_region[d].start)/cgs_level[i_d] for (i_d,d) in enumerate(('lon','lat')))
            print(f' --- Starting loop over lons and lats')
            for (i_lon,i_lat) in select_regions[i_cgs_level]:
                exttemp = da_cgts_extt.isel(lon=i_lon,lat=i_lat) #.to_numpy()
                exttemp_era5 = da_cgts_extt_era5.isel(lon=i_lon,lat=i_lat) #.to_numpy()
                gevpar_reg = xr.open_dataarray(join(reduced_data_dir,f'gevpar_reg_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                exttemp_levels_reg = np.load(join(reduced_data_dir,f'exttemp_levels_reg_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                gevpar_reg_era5 = xr.open_dataarray(join(reduced_data_dir_era5,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                exttemp_levels_reg_era5 = np.load(join(reduced_data_dir_era5,f'exttemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'))
                print(f'      --- Loaded the exttemps and gevpars')
                fig = plt.figure(figsize=(12,8))
                gs = gridspec.GridSpec(figure=fig, nrows=2, ncols=2, height_ratios=[1,18])
                ax_title = fig.add_subplot(gs[0,0:2])
                ax_gev = fig.add_subplot(gs[1,0])
                ax_timeseries = fig.add_subplot(gs[1,1], sharey=ax_gev)


                # Plot timeseries on right
                ax = ax_timeseries
                for i_mem in range(da_cgts.member.size):
                    temp_gcm = da_cgts.isel(lat=i_lat,lon=i_lon,member=i_mem)
                    #print(f'{temp_gcm.time = }')
                    xr.plot.plot(temp_gcm, x='time', ax=ax, color='red', alpha=0.25)
                    i_t_argmax = (ext_sign*temp_gcm).argmax(dim='time').item()
                    idx_argmax = range(max(i_t_argmax,0),min(i_t_argmax+1,temp_gcm['time'].size))
                    xr.plot.plot(temp_gcm.isel(time=idx_argmax), x='time', ax=ax, color='purple', marker='.', linestyle='-')
                for (i_mem,mem) in enumerate(da_cgts_era5.member.to_numpy()):
                    temp_era5 = da_cgts_era5.isel(lat=i_lat,lon=i_lon,member=i_mem)
                    xr.plot.plot(temp_era5, x='time', ax=ax, color='gray', alpha=0.25) 
                    if mem == event_year:
                        xr.plot.plot(temp_era5, x='time', ax=ax, color='black', linestyle='--', linewidth=2) 
                    i_t_argmax = (ext_sign*temp_era5).argmax(dim='time').item()
                    idx_argmax = range(max(i_t_argmax,0),min(i_t_argmax+1,temp_era5.time.size))
                    xr.plot.plot(temp_era5.isel(time=idx_argmax), x='time', ax=ax, color='black', marker='.', linestyle='-')
                ax.set_title('')
                ax.set_xlabel('Time')
                ax.yaxis.set_tick_params(which='both', labelbottom=True)
                ax.set_ylabel('')

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
                ax = ax_gev
                # GCM data
                order = np.argsort(exttemp.to_numpy())
                
                if ext_sign == -1:
                    risk_empirical = np.arange(1,len(exttemp)+1)/len(exttemp)
                else:
                    risk_empirical = np.arange(len(exttemp),0,-1)/len(exttemp)
                ax.scatter(risk_empirical, exttemp.isel(member=order).to_numpy(), color='red', marker='+')
                hgcm, = ax.plot(risk_levels,exttemp_levels_reg[0,:],color='red',label=r'%s (%s)'%(gcm,param_label))
                ax.fill_between(risk_levels, np.quantile(exttemp_levels_reg[1:], 0.25, axis=0), np.quantile(exttemp_levels_reg[1:], 0.75, axis=0), fc='red', ec='none', alpha=0.3, zorder=-1)
                # Now ERA5
                order = np.argsort(exttemp_era5.to_numpy())
                rank = np.argsort(order)
                if ext_sign == -1:
                    risk_empirical = np.arange(1,exttemp_era5.member.size+1)/exttemp_era5.member.size
                else:
                    risk_empirical = np.arange(exttemp_era5.member.size,0,-1)/exttemp_era5.member.size
                ax.scatter(risk_empirical, exttemp_era5.isel(member=order).to_numpy(), color='black', marker='+')
                # Special marker for the year of interest
                i_mem_event_year = np.where(da_cgts_extt_era5.member == event_year)[0][0]

                ax.scatter(risk_empirical[rank[i_mem_event_year]], exttemp_era5.isel(member=i_mem_event_year).item(), color='black', marker='o')
                hera5, = ax.plot(risk_levels,exttemp_levels_reg_era5[0,:],color='black',label=r'ERA5 (%s)'%(param_label_era5))
                ax.fill_between(risk_levels, np.quantile(exttemp_levels_reg_era5, 0.25, axis=0), np.quantile(exttemp_levels_reg_era5, 0.75, axis=0), fc='gray', ec='none', alpha=0.3, zorder=-1)
                ax.legend(handles=[hera5,hgcm])
                ax.set_xscale('log')
                ineq_symb = "geq" if 1==ext_sign else "leq"
                ax.set_xlabel(r'$\mathbb{P}\{\%s_t\langle T(t)\rangle_{\mathrm{region}}\%s T\}$'%(ext_symb,ineq_symb))

                center_lon = event_region['lon'].start + (i_lon+0.5)*lon_blocksize
                center_lat = event_region['lat'].start + (i_lat+0.5)*lat_blocksize
                lonlatstr = r'$\lambda=%d\pm%d,\phi=%d\pm%d$'%(center_lon,lon_blocksize/2,center_lat,lat_blocksize/2)
                # Title 
                ax_title.text(0.5, 0.5, f'{gcm} {expt}, init {datestr} at {lonlatstr}', transform=ax_title.transAxes, ha='center', va='center')
                ax.set_ylabel(r'$T$')
                ax.set_title('')

                fig.savefig(join(figdir,f'riskplot_reg_e{expt}_i{fc_date_abbrv}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                plt.close(fig)
                #sys.exit()
    print(f'Finished the loop over select regions')
    #ds_cgts.close()
    print(f'closed ds_cgts')
    #ds_cgt.close()
    print(f'closed ds_cgt')
    print(psutil.virtual_memory())
    return 

if __name__ == "__main__":
    gcm2institute = all_gcms_institutes()
    gcms = list(gcm2institute.keys())
    gcms2ignore = ["BCC-CSM2-HR","GLOBO","GEM-NEMO","CanESM5","SPEAR"]

    idx_gcms = [i for i in range(len(gcms)) if ((gcms[i] not in gcms2ignore))]
    idx_gcms = [gcms.index(gcm) for gcm in ['CESM2-CAM6','IFS'][1:]] #,'CESM2-CAM6']]
    print(f'{idx_gcms = }')
    print(f'{gcms[i] for i in idx_gcms = }')
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
