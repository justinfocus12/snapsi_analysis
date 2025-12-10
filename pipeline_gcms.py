import numpy as np
import xarray as xr
import pdb
import psutil
from cartopy import crs as ccrs
import gc as garbcol
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
        'CanESM5': 'CCCma',
        'IFS': 'ECMWF',
        'SPEAR': 'NOAA-GFDL',
        'GRIMs': 'SNU',
        'GloSea6-GC32': 'KMA',
        'CNRM-CM61': 'Meteo-France',
        'CESM2-CAM6': 'NCAR',
        'NAVGEM': 'NRL',
        'GloSea6': 'UKMO',
        })
    # indices that count: 3, 4, 6, 7, 8, 9, 11
    return gcm2institute

def gcm_plot_styles():            
    styles = dict({
        'BCC-CSM2-HR': dict({
            'marker': '.',
            }),
        'GLOBO': dict({
            'marker': '$H$',
            }),
        'GEM-NEMO': dict({
            'marker': '$I$',
            }),
        'CanESM5': dict({
            'marker': '$G$',
            }),
        'IFS': dict({
            'marker': "$A$",
            }),
        'SPEAR': dict({
            'marker': ".",
            }),
        'GRIMs': dict({
            'marker': "$B$",
            }),
        'GloSea6-GC32': dict({
            'marker': "$C$",
            }),
        'CNRM-CM61': dict({
            'marker': "$D$",
            }),
        'CESM2-CAM6': dict({
            'marker': "$E$",
            }),
        'NAVGEM': dict({
            'marker': "$.$",
            }),
        'GloSea6': dict({
            'marker': "$F$",
            }),
        })
    return styles

def sanity_check_2019(i_gcm):
    # Look at maps and timeseries of the model data to ERA5
    # -------------- set up location to save the experiment -------------
    fc_dates,onset_date_nominal,term_date = pipeline_base.dates_of_interest(which_ssw)
    wkf = gcm_workflow('sep2019',i_gcm, 1, 0)
    fc_date = fc_dates[0]
    print(f'{fc_date = }')
    wkf_era5 = pipeline_era5.era5_workflow(which_ssw)
    savedir_sanity = join(wkf['reduced_data_dir'], 'compare_with_era5')
    figdir_sanity = join(wkf['figdir'], 'sanitycheck')
    makedirs(savedir_sanity, exist_ok=True)
    makedirs(figdir_sanity, exist_ok=True)
    reduced_file_gcm = join(savedir_sanity, '%s.nc'%(wkf['gcm']))
    reduced_file_era5 = join(savedir_sanity, 'era5.nc')
    # -------------------------------------------------------------------
    era5_files = ['/gws/nopw/j04/snapsi/processed/wg2/ju26596/era5/t2m_2019-%02d.nc'%(m) for m in [8,9,10]]
    gcm_files = wkf['raw_mem_files']
    t2m_era5 = []
    for era5_file in era5_files:
        t2m_era5.append(xr.open_dataarray(era5_file).rename({'valid_time': 'time'}))
    t2m_era5 = xr.concat(t2m_era5, dim='time')
    t2m_era5 = utils.rezero_lons(
            t2m_era5
            .isel(latitude=slice(None,None,-1))
            .rename(longitude='lon',latitude='lat')
            )
    preprocess = lambda ds: preprocess_gcm_6hrPt(ds, fc_date, dict(time=slice(fc_date,term_date)), wkf['context_region'])
    t2m_gcm = []
    for gcm_file in gcm_files[:4]:
        t2m_gcm.append(
                preprocess(xr.open_dataset(gcm_file))
                .sel(wkf['event_region'])
                )
    t2m_gcm = xr.concat(t2m_gcm, dim='member')

    print(f'{era5_file = }')
    print(f'{gcm_file = }')

    t2m_era5.to_netcdf(reduced_file_era5)
    t2m_gcm.to_netcdf(reduced_file_gcm)

    # Interpolate 
    landmask = xr.open_dataarray(wkf['landmask_interp_file'])
    region = wkf['event_region']
    dlon = (region['lon'].stop - region['lon'].start)/wkf['Nlon_interp']
    dlat = (region['lat'].stop - region['lat'].start)/wkf['Nlat_interp']
    t2m_era5,t2m_gcm = (
            t2m_dummy
            .interp(
                lon=np.linspace(region['lon'].start+dlon/2, region['lon'].stop-dlon/2, wkf['Nlon_interp']), 
                lat=np.linspace(region['lat'].start+dlat/2, region['lat'].stop-dlat/2, wkf['Nlat_interp']), 
                method='linear'
                )
            for t2m_dummy in [t2m_era5,t2m_gcm]
            )
    # Area-means 
    t2m_areamean_era5 = (t2m_era5*landmask).sum(dim=['lat','lon'])/(landmask.sum(dim=['lat','lon']))
    t2m_areamean_gcm = (t2m_gcm*landmask).sum(dim=['lat','lon'])/(landmask.sum(dim=['lat','lon']))
    # point values 
    midlon,midlat = [(region[c].start + region[c].stop)/2 for c in ['lon','lat']]
    t2m_center_era5 = t2m_era5.sel(lon=midlon,lat=midlat,method='nearest')
    t2m_center_gcm = t2m_gcm.sel(lon=midlon,lat=midlat,method='nearest')

    fig,axes = plt.subplots(figsize=(12,6),ncols=2,nrows=1,sharey=True,sharex=True)
    ax = axes[0]
    for i_mem in range(t2m_center_gcm['member'].size):
        xr.plot.plot(t2m_center_gcm.isel(member=i_mem), ax=ax, x='time', color='red', label=wkf['gcm'])
    xr.plot.plot(t2m_center_era5, ax=ax, x='time', color='black', label='ERA5')
    ax.set_title('Center values')

    ax = axes[1]
    for i_mem in range(t2m_center_gcm['member'].size):
        xr.plot.plot(t2m_areamean_gcm.isel(member=i_mem), ax=ax, x='time', color='red', label=wkf['gcm'])
    xr.plot.plot(t2m_areamean_era5, ax=ax, x='time', color='black', label='ERA5')
    ax.set_title('Area averages')

    for ax in axes:
        ax.set_xlim([fc_date-dtlib.timedelta(days=2), fc_date+dtlib.timedelta(days=15)])

    fig.savefig(join(figdir_sanity,'sanity.png'), **pltkwargs)
    plt.close(fig)


    # Result: they seem to line up. What is going wrong when passing the data through the pipeline? 

    return
    
    

def gcm_comparison_workflow(which_ssw, idx_gcms):
    gcms,expts,fc_dates,onset_date,term_date = gcm_multiparams(which_ssw)
    fc_dates,onset_date_nominal,term_date = pipeline_base.dates_of_interest(which_ssw)
    wkfs_comp = dict({
        gcms[i_gcm]: expt_comparison_workflow(which_ssw, i_gcm)[2]
        for i_gcm in idx_gcms
        })
    wkf_era5 = pipeline_era5.era5_workflow(which_ssw)
    wkf_gcmcomp = dict(gcms=[gcms[i_gcm] for i_gcm in idx_gcms],expts=expts)
    keys2share = '''
    expt_pairs,
    expt_colors,
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
    boot_type,
    onset_date,
    term_date,
    fc_dates, 
    cgs_levels, 
    select_regions,
    '''
    wkf_gcmcomp.update({key: wkfs_comp[gcms[idx_gcms[0]]][key] for key in utils.unbag_args(keys2share)})
    wkf_gcmcomp['gevsevlev_comp_files'] = dict({
        gcms[i_gcm]: wkfs_comp[gcms[i_gcm]]['gevsevlev_comp_files']
        for i_gcm in idx_gcms
        })
    wkf_gcmcomp['figdir'] = wkfs_comp[gcms[idx_gcms[0]]]['figdir'].replace(gcms[idx_gcms[0]],'multimodel')
    makedirs(wkf_gcmcomp['figdir'], exist_ok=True)
    return wkf_gcmcomp,wkfs_comp,wkf_era5


def expt_comparison_workflow(which_ssw, i_gcm):
    # TODO clean up this dictionary from extraneous information 
    gcms,expts,fc_dates,onset_date,term_date = gcm_multiparams(which_ssw)
    fc_dates,onset_date_nominal,term_date = pipeline_base.dates_of_interest(which_ssw)
    wkfs = dict({
        expt: [
            gcm_workflow(which_ssw, i_gcm, i_expt, i_fc_date)
            for i_fc_date in range(len(fc_dates))
            ]
        for (i_expt,expt) in enumerate(expts)
        })
    wkf_era5 = pipeline_era5.era5_workflow(which_ssw)
    wkf_comp = dict(expts=expts)
    wkf_comp['expt_colors'] = dict({'era5': 'black', 'control': 'dodgerblue', 'free': 'limegreen', 'nudged': 'red'})
    keys2share = '''
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
    boot_type,
    gcm,
    onset_date,
    term_date,
    fc_dates, 
    cgs_levels, 
    reduced_data_dir,
    select_regions,
    figdir,
    '''
    wkf_comp.update({key: wkfs[expts[0]][0][key] for key in utils.unbag_args(keys2share)})
    wkf_comp['expt_baseline'] = 'free'
    wkf_comp['expt_pairs'] = [('free','nudged'),('free','control')]
    wkf_comp['valatrisk_comp_files'] = [] # a single netcdf file per coarse-graining level. TODO should include error bars
    wkf_comp['gevsevlev_comp_files'] = [] # this includes the relative risks and value at risks
    #wkf_comp['relrisk_dvalatrisk_comp_files'] = []
    for (i_cgs_level,cgs_level) in enumerate(wkf_comp['cgs_levels']):
        wkf_comp['valatrisk_comp_files'].append(join(wkf_comp['reduced_data_dir'],'valatrisk_comp_cgs%dx%d.nc'%(cgs_level[0],cgs_level[1])))
        wkf_comp['gevsevlev_comp_files'].append([])
        #wkf_comp['relrisk_dvalatrisk_comp_files'].append([])
        for (i_lon,i_lat) in wkf_comp['select_regions'][i_cgs_level]:
            wkf_comp['gevsevlev_comp_files'][i_cgs_level].append(join(wkf_comp['reduced_data_dir'],'gevsevlev_comp_cgs%dx%d_ilon%d_ilat%d.nc'%(cgs_level[0],cgs_level[1],i_lon,i_lat)))
            #wkf_comp['relrisk_dvalatrisk_comp_files'][i_cgs_level].append(join(wkf_comp['reduced_data_dir'],'relrisk_dvalatrisk_comp_cgs%dx%d_ilon%d_ilat%d.nc'%(cgs_level[0],cgs_level[1],i_lon,i_lat)))

    return wkf_era5,wkfs,wkf_comp



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
    analysis_date = '2025-12-04'
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
        gevpar_files.append(join(reduced_data_dir,'gevpar_e%s_i%s_cgs%dx%d.nc'%(expt,fc_date_abbrv,cgs_level[0],cgs_level[1])))
        risk_files.append(join(reduced_data_dir,'risk_e%s_i%s_cgs%dx%d.nc'%(expt,fc_date_abbrv,cgs_level[0],cgs_level[1])))
        valatrisk_files.append(join(reduced_data_dir,'valatrisk_e%s_i%s_cgs%dx%d.nc'%(expt,fc_date_abbrv,cgs_level[0],cgs_level[1])))
        gevsevlev_files.append([])
        for (i_lon,i_lat) in select_regions[i_cgs_level]:
            gevsevlev_files[i_cgs_level].append(join(reduced_data_dir,'gevsevlev_e%s_i%s_cgs%dx%d_ilon%d_ilat%d.nc'%(expt,fc_date_abbrv,cgs_level[0],cgs_level[1],i_lon,i_lat)))
    # One more set of files for context region 
    ens_files_cgts.append(join(reduced_data_dir,'t2m_cgt1day_cgscontext.nc'))
    ens_files_cgts_extt.append(join(reduced_data_dir,'t2m_cgt1day_cgscontext_extt.nc'))
    gevpar_files.append(join(reduced_data_dir,'gevpar_cgscontext.nc'))
    risk_files.append(join(reduced_data_dir,'risk_cgscontext.nc'))
    valatrisk_files.append(join(reduced_data_dir,'valatrisk_cgscontext.nc'))




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
            Nlon_pad_pre,
            Nlon_pad_post,
            Nlat_pad_pre,
            Nlat_pad_post,
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



def coarse_grain_time(raw_mem_files, mem_labels, event_region, context_region, Nlon_interp, Nlat_interp, Nlon_pad_pre, Nlon_pad_post, Nlat_pad_pre, Nlat_pad_post, init_date, term_date, ens_file_cgt, use_dask=False):
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
    lon_interp = np.linspace(context_region['lon'].start+dlon/2, context_region['lon'].stop-dlon/2, Nlon_interp+Nlon_pad_pre+Nlon_pad_post)
    lat_interp = np.linspace(context_region['lat'].start+dlat/2, context_region['lat'].stop-dlat/2, Nlat_interp+Nlat_pad_pre+Nlat_pad_post)

    ds_ens = ds_ens.interp(lon=lon_interp, lat=lat_interp, method="linear").sel(context_region)
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
    print("AAAUUGGGHHH")
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
    # TODO record the full timespan of the dataset
    return dsmem_tas

def plot_relrisks_dvalatrisks_allgcms(
        gevsevlev_comp_files: dict,
        gcms, fc_dates, 
        cgs_levels, select_regions,
        expts,expt_pairs,expt_colors, confint_width,
        figdir
        ):
    # 2D scatter plot: dvalatrisk vs relrisk 
    # different color for each GCM; different marker for free->control and free->nudged  
    # left and right panels for early and late FC date 
    gcmstyles = gcm_plot_styles()
    # TODO for the infinite cases, at least indicate which direction the partner expt_pair is 
    rrtickvalues = [0.25, 0.5, 1.0, 2.0, 4.0]
    logrrtickvalues = list(map(np.log10, rrtickvalues))
    logrrlims = [1.5*logrrtickvalues[0]-0.5*logrrtickvalues[1], 1.5*logrrtickvalues[-1]-0.5*logrrtickvalues[-2]]
    # Keep track of difference in differences and ratio of ratios
    def truncated_log10(rr):
        if rr > rrtickvalues[-1]:
            return (logrrtickvalues[-1]+logrrlims[1])/2
        if np.isnan(rr) or rr < rrtickvalues[0]:
            return (logrrlims[0]+logrrtickvalues[0])/2
        return np.log10(rr)
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        for (i_region,(i_lon,i_lat)) in enumerate(select_regions[i_cgs_level]):
            fig = plt.figure(figsize=(16,8))
            gs = gridspec.GridSpec(figure=fig, nrows=1, ncols=2, width_ratios=[1,1])
            ax_fc0 = fig.add_subplot(gs[0,0])
            ax_fc1 = fig.add_subplot(gs[0,1], sharex=ax_fc0, sharey=ax_fc0)
            axs_fc = [ax_fc0,ax_fc1]
            ylims_dvar = [np.inf,-np.inf]
            # Multi-model means 
            mmm = xr.DataArray(
                    coords={'quantity': ['logrr','dvar'], 'fc_date': fc_dates, 'expt_pair': list(map(expt_pair_coordval, expt_pairs))}, 
                    dims=['quantity','fc_date','expt_pair'],
                    data=0.0)
            for (i_gcm,gcm) in enumerate(gcms):
                gev_sev_risk_var_rr_dvar = xr.open_dataset(gevsevlev_comp_files[gcm][i_cgs_level][i_region])
                rr_dvar = gev_sev_risk_var_rr_dvar['rr_dvar'] #.sel(quantity=q) for q in ['relrisk','dvalatrisk'])
                for (fc_date,ax) in zip(fc_dates,axs_fc):
                    rrmids,dvarmids = [rr_dvar.sel(quantity=q,fc_date=fc_date,boot=0) for q in ['relrisk','dvalatrisk']]
                    ylims_dvar[0] = min(ylims_dvar[0], np.nanmin(dvarmids).item())
                    ylims_dvar[1] = max(ylims_dvar[1], np.nanmax(dvarmids).item())
                    for (i_expt_pair,expt_pair) in enumerate(expt_pairs):
                        rrmid,dvarmid = [dthing.sel(expt_pair=expt_pair_coordval(expt_pair)).item() for dthing in (rrmids,dvarmids)]
                        logrrmid = truncated_log10(rrmid)
                        mmm.loc[dict(quantity='logrr',fc_date=fc_date,expt_pair=expt_pair_coordval(expt_pair))] += logrrmid/len(gcms)
                        mmm.loc[dict(quantity='dvar',fc_date=fc_date,expt_pair=expt_pair_coordval(expt_pair))] += dvarmid/len(gcms)
                        logrrlo,logrrhi= [
                                np.quantile(
                                    np.vectorize(truncated_log10)(
                                        rr_dvar.sel(expt_pair=expt_pair_coordval(expt_pair),fc_date=fc_date,quantity='relrisk',boot=slice(1,None)).to_numpy()),
                                    0.5*(1+sgn*confint_width))
                                for sgn in [-1,1]]
                        #logrrlo,logrrhi = list(map(np.log10, [np.quantile(rr_dvar.sel(quantity='relrisk',expt_pair=expt_pair_coordval(expt_pair),fc_date=fc_date,boot=slice(1,None)), 0.5*(1+sgn*confint_width)).item() for sgn in [-1,1]]))
                        if not np.all(np.isfinite([logrrmid,dvarmid])):
                            continue
                        dvarlo,dvarhi = [np.quantile(rr_dvar.sel(quantity='dvalatrisk',expt_pair=expt_pair_coordval(expt_pair),fc_date=fc_date,boot=slice(1,None)), 0.5*(1+sgn*confint_width)).item() for sgn in [-1,1]]
                        ax.plot([logrrlo,logrrhi],[dvarmid,dvarmid], color=expt_colors[expt_pair[1]], zorder=-1)
                        ax.plot([logrrmid,logrrmid], [dvarlo,dvarhi], color=expt_colors[expt_pair[1]], zorder=-1)
                        ax.scatter(logrrmid,dvarmid,marker='o',fc='white',ec=expt_colors[expt_pair[1]], zorder=0, s=18**2)
                        ax.scatter(logrrmid,dvarmid,marker=gcmstyles[gcm]['marker'],color='black', zorder=1)
                        # draw an arrow toward the other one 
                        expt_pair_other = expt_pairs[(i_expt_pair+1) % 2]
                        ddvar = dvarmids.sel(expt_pair=expt_pair_coordval(expt_pair_other)) - dvarmid
                        dlogrr = truncated_log10(rrmids.sel(expt_pair=expt_pair_coordval(expt_pair_other))) - logrrmid
                        ax.plot(
                                [logrrmid + factor*dlogrr for factor in [0,0.5]],
                                [dvarmid + factor*ddvar for factor in [0,.5]],
                                color=expt_colors[expt_pair[1]],
                                linewidth=6, alpha=0.25, zorder=2)
            for (fc_date,ax) in zip(fc_dates,axs_fc):
                for (i_expt_pair,expt_pair) in enumerate(expt_pairs):
                    ax.scatter(
                            np.maximum(logrrlims[0], np.minimum(logrrlims[1], mmm.sel(quantity='logrr',fc_date=fc_date,expt_pair=expt_pair_coordval(expt_pair)))), 
                            mmm.sel(quantity='dvar',fc_date=fc_date,expt_pair=expt_pair_coordval(expt_pair)), 
                            fc='none', linewidth=3, ec=expt_colors[expt_pair[1]], s=38**2, marker='H', alpha=1.0, zorder=-2)
            grk = utils.greekletters()
            ylims_dvar = utils.padded_bounds(ylims_dvar, inflation=0.1)
            ylims_dvar = np.max(np.abs(ylims_dvar)) * np.array([-1,1])
            for (fc_date,ax) in zip(fc_dates,axs_fc):
                #ax.set_xscale('log')
                ax.set_xlim(logrrlims)
                xticklabels = list(map(
                        lambda rr: "1/%d"%(int(round(1/rr))) if rr<1 else "%d"%rr, 
                        rrtickvalues))
                xticklabels[0] = "\u003C"+xticklabels[0]
                xticklabels[-1] = xticklabels[-1]+"\u003E"
                ax.set_xticks(logrrtickvalues, xticklabels)
                ax.set_ylabel("severity - (free severity) [K]")
                ax.set_xlabel("risk / (free risk)")
                ax.set_title(dtlib.datetime.strftime(fc_date,"FC %Y/%m/%d"))
                ax.axhline(0, color=expt_colors['free'])
                ax.axvline(0, color=expt_colors['free'])
                ax.set_ylim(ylims_dvar)
                ax.fill_betweenx(ylims_dvar, logrrlims[0], logrrtickvalues[0], ec='none', fc='gray', alpha=0.3, zorder=-1)
                ax.fill_betweenx(ylims_dvar, logrrtickvalues[-1], logrrlims[-1], ec='none', fc='gray', alpha=0.3, zorder=-1)
                
            fig.savefig(join(figdir,f'relrisks_dvalatrisks_allgcms_cgs{cgs_level[0]}x{cgs_level[1]}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
            plt.close(fig)
                   
    return




def compare_gcms(which_ssw, idx_gcms):
    todo = dict({
        'plot_relrisks_dvalatrisks': 1,
        })
    gcms,expts,fc_dates,_,term_date = gcm_multiparams(which_ssw)
    wkf_gcmcomp,wkfs_comp,wkf_era5 = gcm_comparison_workflow(which_ssw, idx_gcms)

    if todo['plot_relrisks_dvalatrisks']:
        plot_relrisks_dvalatrisks_allgcms(
                *dtoa(wkf_gcmcomp,'''
                gevsevlev_comp_files,
                gcms, fc_dates, cgs_levels, select_regions,
                expts, expt_pairs, expt_colors, confint_width,
                figdir
                ''')
                )

    return


            
def plot_gevpar_difference_maps(wkfs,wkf_comp):
    for (expt0,expt1) in wkf_comp['expt_pairs']:
        for (i_fcdate,fc_date) in enumerate(wkf_comp['fc_dates']):
            pipeline_base.plot_gevpar_difference_maps_flat(
                    *dtoa(wkfs[expt0][i_fcdate], 'gevpar_files,expt,'),
                    *dtoa(wkfs[expt1][i_fcdate], 'gevpar_files,expt,'),
                    *dtoa(wkfs[expt0][i_fcdate], 'param_bounds_file,ext_sign,cgs_levels,event_region,figdir,gcm,fc_date,fc_date_abbrv')
                    )
    return

def expt_pair_coordval(expt_pair):
    return '%s2%s'%(expt_pair[0],expt_pair[1])

def compute_gevsevlev_comp_select_regions(
        # A more-detailed regional verion of valatrisk_comp 
        gevsevlev_files: dict, 
        gevsevlev_files_ref, 
        gevsevlev_comp_files, 
        risk_levels,
        expts, expt_pairs, cgs_levels, select_regions, fc_dates, n_boot
        ):
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        for (i_region,(i_lon,i_lat)) in enumerate(select_regions[i_cgs_level]):
            gev_sev_risk_var = xr.concat([
                xr.concat([
                    xr.open_dataset(gevsevlev_files[expt][i_fc_date][i_cgs_level][i_region])
                    for (i_fc_date,fc_date) in enumerate(fc_dates)
                    ], dim='fc_date').assign_coords(fc_date=fc_dates)
                for expt in expts
                ], dim='expt').assign_coords(expt=expts)
            #gev_sev_risk_var_ref = xr.open_dataset(gevsevlev_files_ref[i_cgs_level][i_region])
            # Compute relative risks and changes in value at risk 
            rr_dvar = xr.DataArray(coords={'expt_pair': list(map(expt_pair_coordval, expt_pairs)), 'fc_date': fc_dates, 'quantity': ['relrisk','dvalatrisk'], 'boot': np.arange(n_boot+1)}, dims=['expt_pair','fc_date','quantity','boot'], data=np.nan)
            for (i_expt_pair,expt_pair) in enumerate(expt_pairs):
                rr_dvar.loc[dict(expt_pair=expt_pair_coordval(expt_pair),quantity='relrisk')] = np.divide(*(
                    gev_sev_risk_var['risk_refgivenexpt']
                    .sel(expt=expt)
                    for expt in (expt_pair[1],expt_pair[0])
                    ))
                rr_dvar.loc[dict(expt_pair=expt_pair_coordval(expt_pair),quantity='dvalatrisk')] = np.subtract(*(
                    gev_sev_risk_var['valatrisk_refgivenexpt']
                    .sel(expt=expt)
                    for expt in (expt_pair[1],expt_pair[0])
                    ))
            gev_sev_risk_var_rr_dvar = xr.Dataset(data_vars={**{dv: gev_sev_risk_var[dv].copy() for dv in gev_sev_risk_var.data_vars.keys()}, 'rr_dvar': rr_dvar})
            print(f'{i_lon = }, {i_lat = }')
            #if (i_lon == 8 and i_lat == 1):
            #    pdb.set_trace()
            gev_sev_risk_var_rr_dvar.to_netcdf(gevsevlev_comp_files[i_cgs_level][i_region])
            
            gev_sev_risk_var.close()
            gev_sev_risk_var_rr_dvar.close()
    return



def compute_valatrisk_comp(
        valatrisk_files: dict,
        valatrisk_comp_files,
        expts, expt_pairs, cgs_levels, fc_dates,  
        ):
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        print(f"Starting {cgs_level = }")
        coords = xr.open_dataset(valatrisk_files[expts[0]][0][i_cgs_level]).coords
        valatrisk_comp = xr.DataArray(
                coords={'expt_pair': list(map(expt_pair_coordval, expt_pairs)), 'quantity': ['dvalatrisk','relrisk'], 'fc_date': fc_dates, 'lon': coords['lon'], 'lat': coords['lat']},
                dims=['expt_pair','quantity','fc_date','lon','lat'],
                data=np.nan)
        valatrisks = []
        for i_fcdate,fc_date in enumerate(fc_dates):
            valatrisks_fcdate = [xr.open_dataarray(valatrisk_files[expt][i_fcdate][i_cgs_level]) for expt in expts]
            valatrisks.append(xr.concat(valatrisks_fcdate, dim="expt").assign_coords(expt=expts))
            for expt_pair in expt_pairs:
                valatrisk_comp.loc[dict(fc_date=fc_date,expt_pair=expt_pair_coordval(expt_pair),quantity='relrisk')] = np.divide(*(
                    valatrisks[i_fcdate].sel(expt=expt,quantity='risk_refgivenexpt') 
                    for expt in (expt_pair[1],expt_pair[0])
                    ))
                valatrisk_comp.loc[dict(fc_date=fc_date,expt_pair=expt_pair_coordval(expt_pair),quantity='dvalatrisk')] = np.subtract(*(
                    valatrisks[i_fcdate].sel(expt=expt,quantity='risk_refgivenexpt') 
                    for expt in (expt_pair[1],expt_pair[0])
                    ))
        valatrisks_abs_comp = xr.Dataset(data_vars=dict({
            "valatrisks": xr.concat(valatrisks, dim="fc_date").assign_coords(fc_date=fc_dates),
            "valatrisk_comp": valatrisk_comp
            }))
        valatrisks_abs_comp.to_netcdf(valatrisk_comp_files[i_cgs_level])
    return

def plot_valatrisk_comp_maps(
        valatrisk_comp_files, expt_pairs, event_region, cgs_levels, fc_dates, ext_sign, figdir,
        gcm, onset_date, term_date, ineq_symb, ext_symb,
        ):
    # Two maps at each CG scale: relative risk, and quantile shift aka d(value at risk) 
    fmtfun = lambda date: dtlib.datetime.strftime(date, "%Y/%m/%d")
    for (i_cgs_level, cgs_level) in enumerate(cgs_levels):
        if min(cgs_level) == 1:
            continue
        valatrisks = xr.open_dataset(valatrisk_comp_files[i_cgs_level])['valatrisks']
        for expt_pair in expt_pairs:
            for i_fcdate, fc_date in enumerate(fc_dates):
                risk0,risk1 = (valatrisks.sel(quantity='risk_refgivenexpt',fc_date=fc_date,expt=expt) for expt in expt_pair)
                fig,ax = pipeline_base.plot_relative_risk_map_flat(risk0, risk1, event_region, ext_sign, plot_contour_ratio=False)
                fc_date_abbrv = dtlib.datetime.strftime(fc_date, "%Y%m%d")
                title = "%s, FC %s %s \u2192 %s\n\u2119{%s{T2M(t):%s\u2264t\u2264%s}%s(ERA5 value)}"%(gcm, fmtfun(fc_date), expt_pair[0], expt_pair[1], ext_symb, fmtfun(onset_date), fmtfun(term_date), ineq_symb)
                ax.set_title(title, loc='left')
                figfile = join(figdir,f'relrisk_map_{expt_pair[0]}2{expt_pair[1]}_i{fc_date_abbrv}_cgs{cgs_level[0]}x{cgs_level[1]}.png')
                fig.savefig(figfile, **pltkwargs)
                plt.close(fig)
    return


def plot_gevsevlev_comp_select_regions(
        # TODO read in the precomputed RRs and DVs to ease this  
        gevsevlev_comp_files, gevsevlev_files_era5, mem_special_era5,
        ens_files_cgts_extt, ens_files_cgts_extt_era5, 
        expts, expt_pairs, expt_baseline, cgs_levels, fc_dates, event_region, select_regions,
        gcm, onset_date, term_date, 
        ext_sign, confint_width, 
        ineq_symb, ext_symb, expt_colors, 
        figdir, 
        ):
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        da_cgts_extt_era5 = xr.open_dataarray(ens_files_cgts_extt_era5[i_cgs_level])
        i_mem_special_era5 = np.argmax([mem==mem_special_era5 for mem in da_cgts_extt_era5.coords['member'].values])
        da_cgts_extt = (
                xr.concat([
                    xr.concat([
                        xr.open_dataarray(ens_files_cgts_extt[expt][i_fcdate][i_cgs_level])
                        for i_fcdate in range(len(fc_dates))], dim='fc_date')
                    .assign_coords(fc_date=fc_dates)
                    for (i_expt,expt) in enumerate(expts)], dim='expt')
                .assign_coords(expt=expts)
                )
        gcmstyles = gcm_plot_styles()
        for (i_region,(i_lon,i_lat)) in enumerate(select_regions[i_cgs_level]):
            gev_sev_risk_var_rr_dvar = xr.open_dataset(gevsevlev_comp_files[i_cgs_level][i_region])
            sevlev = gev_sev_risk_var_rr_dvar['sevlev']
            gevsevlev_era5 = xr.open_dataset(gevsevlev_files_era5[i_cgs_level][i_region])
            sevlev_era5,risk_refgivenref = gevsevlev_era5['sevlev'],gevsevlev_era5['risk_refgivenexpt']
            sev_bounds = utils.padded_bounds(da_cgts_extt_era5.isel(lon=i_lon,lat=i_lat), inflation=0.5)
            risks = sevlev.coords["risk"].to_numpy() # increasing 
            ordering_for_interp = np.arange(len(risks)-1,-1,-1)
            if ext_sign == -1:
                ordering_for_interp = np.flip(ordering_for_interp)
            for (i_fcdate,fc_date) in enumerate(fc_dates):
                fig = plt.figure(figsize=(9,12))
                gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig, width_ratios=[2,1], height_ratios=[1/9, 2, 1], hspace=0.1, wspace=0.1)
                ax_title = fig.add_subplot(gs[0,0:2])
                axgev = fig.add_subplot(gs[1,0])
                axdvar = fig.add_subplot(gs[2,0], sharex=axgev)
                axrr = fig.add_subplot(gs[1,1], sharey=axgev)
                axrrdvar = fig.add_subplot(gs[2,1], sharey=axdvar, sharex=axrr)

                handles = []
                sevlev_common = np.linspace(*(utils.padded_bounds(sevlev, inflation=0.0)), 40)
                interp_fun = lambda sev: np.interp(sevlev_common, sev[ordering_for_interp], risks[ordering_for_interp])
                risks_interp_baseline = np.apply_along_axis(interp_fun, 1, sevlev.sel(fc_date=fc_date,expt=expt_baseline).to_numpy())
                # Delineate the plot limits to know where to truncate those off-the-charts
                rrtickvalues = [0.25, 0.5, 1.0, 2.0, 4.0]
                logrrtickvalues = list(map(np.log10, rrtickvalues))
                logrrlims = [1.5*logrrtickvalues[0]-0.5*logrrtickvalues[1], 1.5*logrrtickvalues[-1]-0.5*logrrtickvalues[-2]]
                risklim = [risks[0],risks[-1]]


                for expt in expts+['era5']:
                    if expt in expts:
                        exttemp = da_cgts_extt.isel(lon=i_lon,lat=i_lat).sel(expt=expt,fc_date=fc_date).to_numpy()
                        sevlev_expt = sevlev.sel(fc_date=fc_date,expt=expt)
                        Nmem = da_cgts_extt.coords['member'].size
                    else:
                        exttemp = da_cgts_extt_era5.isel(lon=i_lon,lat=i_lat).to_numpy()
                        sevlev_expt = sevlev_era5
                        Nmem = da_cgts_extt_era5.coords['member'].size
                    risk_empirical = (0.5+np.arange(0,Nmem,1))/Nmem
                    risklim[0] = max(risklim[0], risk_empirical[0])
                    ordering_for_interp_empirical = np.arange(Nmem-1,-1,-1)
                    if ext_sign == -1:
                        ordering_for_interp_empirical = np.flip(ordering_for_interp_empirical)
                    order = np.argsort(exttemp)
                    rank = np.argsort(order)
                    # GEV 
                    axgev.scatter(risk_empirical[ordering_for_interp_empirical], exttemp[order], marker='+', color=expt_colors[expt])
                    h, = axgev.plot(risks,sevlev_expt.isel(boot=0).to_numpy(),color=expt_colors[expt], label=expt)
                    handles.append(h)
                    axgev.fill_between(risks,*(np.quantile(sevlev_expt.to_numpy(), 0.5+sgn*0.5*confint_width, axis=0) for sgn in [-1,1]), alpha=0.25, color=expt_colors[expt], zorder=-1)
                    if "era5" != expt:
                        # Quantile shift
                        dvar = sevlev_expt.to_numpy() - sevlev.sel(fc_date=fc_date,expt=expt_baseline)
                        axdvar.plot(risks, dvar[0,:], color=expt_colors[expt])
                        axdvar.fill_between(risks, *(np.quantile(dvar, 0.5+sgn*0.5*confint_width, axis=0) for sgn in (-1,1)), alpha=0.5, zorder=-1, color=expt_colors[expt])
                        # empirical quantile shift
                        if (expt_baseline,expt) in expt_pairs:
                            dvar_emp = np.subtract(*(np.sort(da_cgts_extt.isel(lon=i_lon,lat=i_lat).sel(fc_date=fc_date,expt=e).to_numpy().flatten()) for e in [expt,expt_baseline]))
                            axdvar.scatter(risk_empirical, dvar_emp, color=expt_colors[expt], marker='+')
                        # interpolate for relative risks
                        risks_interp = np.apply_along_axis(interp_fun, 1, sevlev_expt.to_numpy())
                        relrisk = risks_interp / risks_interp_baseline
                        # When computing relrisk, truncate appropriately 
                        axrr.plot(np.log10(relrisk[0,:]), sevlev_common, color=expt_colors[expt])
                        axrr.fill_betweenx(sevlev_common, *(np.quantile(np.log10(relrisk), 0.5+0.5*sgn*confint_width, axis=0) for sgn in (-1,1)), color=expt_colors[expt], alpha=0.25, zorder=-1)
                    def truncated_log10(rr):
                        if rr > rrtickvalues[-1]:
                            return (logrrtickvalues[-1]+logrrlims[1])/2
                        if np.isnan(rr) or rr < rrtickvalues[0]:
                            return (logrrlims[0]+logrrtickvalues[0])/2
                        return np.log10(rr)
                    #logrelrisk = np.vectorize(truncated_logrr)(risks_interp_baseline, risks_interp)
                    print(f'{expt = }')
                    rrmids,dvarmids = [gev_sev_risk_var_rr_dvar['rr_dvar'].sel(fc_date=fc_date,quantity=q,boot=0) for q in ['relrisk','dvalatrisk']]
                    # for the relative risk plot, implement the log scale manually 
                    # TODO plot differences on RR and QS plots 
                    expt_pair = (expt_baseline,expt)
                    if expt_pair in expt_pairs:
                        rrmid,dvarmid = [dthing.sel(expt_pair=expt_pair_coordval(expt_pair)).item() for dthing in (rrmids,dvarmids)]
                        logrrmid = truncated_log10(rrmid)
                        logrrlo,logrrhi= [
                                np.quantile(
                                    np.vectorize(truncated_log10)(
                                        gev_sev_risk_var_rr_dvar['rr_dvar'].sel(expt_pair=expt_pair_coordval(expt_pair),fc_date=fc_date,quantity='relrisk',boot=slice(1,None)).to_numpy()),
                                    0.5*(1+sgn*confint_width))
                                for sgn in [-1,1]]
                        dvarlo,dvarhi = [np.quantile(
                            gev_sev_risk_var_rr_dvar['rr_dvar'].sel(expt_pair=expt_pair_coordval(expt_pair),fc_date=fc_date,quantity='dvalatrisk',boot=slice(1,None)), 0.5*(1+sgn*confint_width)) for sgn in [-1,1]]
                        axrrdvar.scatter(logrrmid,dvarmid,marker='o',s=15**2, ec=expt_colors[expt], fc='white', zorder=1)
                        axrrdvar.scatter(logrrmid,dvarmid,marker=gcmstyles[gcm]['marker'], zorder=2)
                        axrrdvar.plot([logrrlo,logrrhi],[dvarmid,dvarmid],color=expt_colors[expt], zorder=-1)
                        axrrdvar.plot([logrrmid,logrrmid],[dvarlo,dvarhi],color=expt_colors[expt], zorder=-1)
                        # draw an arrow toward the other one 
                        for expt_other in expts:
                            expt_pair_other = (expt_baseline,expt_other)
                            if (expt_other != expt) and (expt_pair_other in expt_pairs):
                                ddvar = dvarmids.sel(expt_pair=expt_pair_coordval((expt_baseline,expt_other))) - dvarmid
                                dlogrr = truncated_log10(rrmids.sel(expt_pair=expt_pair_coordval(expt_pair_other)).item()) - logrrmid
                                axrrdvar.plot(
                                        [logrrmid + factor*dlogrr for factor in [0,0.5]],
                                        [dvarmid + factor*ddvar for factor in [0,.5]],
                                        color=expt_colors[expt],
                                        linewidth=6, alpha=0.25, zorder=2)
                    if "era5" == expt:
                        for ax in (axgev,axrr):
                            ax.axhline(exttemp[i_mem_special_era5], color=expt_colors['era5'], linestyle='--')
                        for ax in (axgev,axdvar):
                            ax.axvline(risk_refgivenref.isel(boot=0).item(), color=expt_colors['era5'], linestyle='--')

                fc_date_abbrv = dtlib.datetime.strftime(fc_date,'%Y%m%d')
                fc_date_label = dtlib.datetime.strftime(fc_date,'%Y/%m/%d')
                lonlatstr = utils.lonlatstr(event_region,cgs_level,i_lon,i_lat)

                # Decorations
                ax_title.text(
                        0.5, 0.0, 
                        f'{gcm}, FC {fc_date_label}, {lonlatstr}',
                        ha='center', va='bottom', transform=ax_title.transAxes, fontsize=20
                        )
                #ax_title.text(0.5, 0.0, figtitle_affix, transform=ax_title.transAxes, ha='center', va='bottom', fontsize=20)
                ylims_dvar = (sev_bounds[1]-sev_bounds[0])*np.array([-1/4,1/4])
                ax_title.axis('off')
                axgev.legend(handles=handles)

                axrrdvar.axhline(0, color=expt_colors[expt_baseline])
                axrrdvar.axvline(0, color=expt_colors[expt_baseline])
                axrrdvar.fill_betweenx(ylims_dvar, logrrlims[0], logrrtickvalues[0], ec='none', fc='gray', zorder=-1, alpha=0.3)
                axrrdvar.fill_betweenx(ylims_dvar, logrrtickvalues[-1], logrrlims[1], ec='none', fc='gray', zorder=-1, alpha=0.3)
                axrrdvar.set_xlabel("risk / (free risk)")
                axgev.set_ylabel("severity [K]")
                grk = utils.greekletters()
                axdvar.set_ylabel(f"severity - (free severity) [K]")
                axdvar.set_xlabel("Probability")

                for ax in (axgev, axrr):
                    ax.set_ylim(sev_bounds)
                for ax in (axgev,):
                    ax.set_xscale('log')
                for ax in (axgev, axdvar):
                    #ax.set_xlim([risks[0],risks[-1]])
                    ax.set_xlim(risklim)
                for ax in (axrr,axrrdvar):
                    ax.set_xlim(logrrlims)
                    xticklabels = list(map(
                            lambda rr: "1/%d"%(int(round(1/rr))) if rr<1 else "%d"%rr, 
                            rrtickvalues))
                    xticklabels[0] = "\u003C"+xticklabels[0]
                    xticklabels[-1] = xticklabels[-1]+"\u003E"
                    ax.set_xticks(logrrtickvalues, xticklabels)
                for ax in (axgev,axrr):
                    ax.xaxis.set_tick_params(which='both',labelbottom=False)
                for ax in (axrr,axrrdvar):
                    ax.yaxis.set_tick_params(which='both',labelbottom=False)
                axdvar.set_ylim(ylims_dvar)
                fig.savefig(join(figdir,f'gevsevlev_comp_fc{fc_date_abbrv}_cgs{cgs_level[0]}x{cgs_level[1]}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                plt.close(fig)
            #relrisk_dvalatrisk.to_netcdf(relrisk_dvalatrisk_comp_files[i_cgs_level][i_region])
            #gevsevlev.to_netcdf(gevsevlev_comp_files[i_cgs_level][i_region])

                
    return
        
        


    

def compare_expts(which_ssw, i_gcm):
    todo = dict({
        'plot_gevpar_map_diffs':                    1,
        'compute_valatrisk_comp':                   1,
        'plot_valatrisk_comp_maps':                 1,
        'compute_gevsevlev_comp_select_regions':    1,
        'plot_gevsevlev_comp_select_regions':       1,
        })
    gcms,expts,fc_dates,_,term_date = gcm_multiparams(which_ssw)
    wkf_era5,wkfs,wkf_comp = expt_comparison_workflow(which_ssw,i_gcm)
    if todo['plot_gevpar_map_diffs']:
        plot_gevpar_difference_maps(wkfs,wkf_comp)
    if todo['compute_valatrisk_comp']:
        compute_valatrisk_comp(
                {expt: 
                    [wkfs[expt][i_fc_date]['valatrisk_files'] for i_fc_date in range(len(fc_dates))] 
                    for expt in expts},
                *dtoa(wkf_comp, '''
                valatrisk_comp_files, 
                expts, expt_pairs, cgs_levels, fc_dates
                ''')
                )
    if todo['plot_valatrisk_comp_maps']:
        plot_valatrisk_comp_maps(
                *dtoa(wkf_comp, '''
                valatrisk_comp_files, 
                expt_pairs, event_region, cgs_levels, fc_dates, ext_sign,
                figdir, gcm, onset_date, term_date, ineq_symb, ext_symb,
                ''')
                )
    if todo['compute_gevsevlev_comp_select_regions']:
        compute_gevsevlev_comp_select_regions(
                {expt: 
                    [wkfs[expt][i_fc_date]['gevsevlev_files'] for i_fc_date in range(len(fc_dates))] for expt in expts},
                wkf_era5['gevsevlev_files'],
                *dtoa(wkf_comp,'''
                gevsevlev_comp_files, 
                risk_levels, expts, expt_pairs, cgs_levels, select_regions, fc_dates, n_boot
                ''')
                )
    if todo['plot_gevsevlev_comp_select_regions']:
        # TODO split this task into two subtasks: one to compile together the gevsevlevs into a single netcdf, and anothre to plot them. For now, stack them both into the same function
        plot_gevsevlev_comp_select_regions(
                wkf_comp['gevsevlev_comp_files'],
                wkf_era5['gevsevlev_files'], wkf_era5['event_year'],
                {expt: 
                    [wkfs[expt][i_fc_date]['ens_files_cgts_extt'] for i_fc_date in range(len(fc_dates))] 
                    for expt in expts},
                *dtoa(wkf_era5, 'ens_files_cgts_extt, '),
                *dtoa(wkf_comp, '''
                expts, expt_pairs, expt_baseline, cgs_levels, fc_dates, event_region, select_regions, 
                gcm, onset_date, term_date, 
                ext_sign, confint_width,
                ineq_symb, ext_symb, expt_colors,
                figdir, 
                ''')
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

    if todo['plot_gevsevlev_select_regions']:
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

                # interpolate for relative risk 
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



def reduce_gcm(which_ssw,i_gcm,i_expt,i_init,todoflags=None):
    # One GCM, one forcing (expt), one initialization (init), multiple coarse-grainings in space 
    todokeys = utils.unbag_args('''
    coarse_grain_time,              
    coarse_grain_space,             
    onset_date_sensitivity_analysis,
    compute_severities,             
    plot_sumstats_map,              
    fit_gev,                        
    plot_gevpar_map,                
    compute_risk,                   
    plot_risk_map,                  
    plot_valatrisk_map,             
    fit_gev_select_regions,         
    plot_gevsevlev_select_regions,  
    ''')
    if todoflags is None:
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
            'plot_gevsevlev_select_regions':    1,
            })
    else:
        todo = dict({key: todoflags[i] for (i,key) in enumerate(todokeys)})


    # In this main function, specify only the inputs and outputs as files 
    wkf = gcm_workflow(which_ssw,i_gcm,i_expt,i_init)
    wkf_era5 = pipeline_era5.era5_workflow(which_ssw)
    if todo['coarse_grain_time']:
        coarse_grain_time(
                *(wkf[key.strip()] for key in '''
                raw_mem_files,mem_labels,
                event_region,context_region,
                Nlon_interp, Nlat_interp, Nlon_pad_pre, Nlon_pad_post, Nlat_pad_pre, Nlat_pad_post,  
                fc_date,term_date, ens_file_cgt
                '''.split(',')),
                )

    if todo['coarse_grain_space']:
        pipeline_base.coarse_grain_space(
                *dtoa(wkf,'''
                ens_file_cgt, ens_files_cgts, cgs_levels, landmask_interp_file,
                event_region, context_region, Nlon_interp, Nlat_interp,
                Nlon_pad_pre, Nlon_pad_post, Nlat_pad_pre, Nlat_pad_post,  
                ''')
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
                wkf['event_region'],wkf['context_region'],
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
                overwrite_gevpar=True
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
        pipeline_base.fit_gev_select_regions(
                *dtoa(wkf_era5, '''
                ens_files_cgts_extt, gevsevlev_files, 
                event_year
                '''),
                *dtoa(wkf, '''
                ens_files_cgts_extt, gevsevlev_files, 
                risk_levels, 
                cgs_levels, select_regions,
                ext_sign,
                '''),
                False, n_boot=1000
                )
    if todo['plot_gevsevlev_select_regions']:
        pipeline_base.plot_gevsevlev_select_regions(
                *dtoa(wkf, 'ens_files_cgts, ens_files_cgts_extt, gevsevlev_files'),
                *dtoa(wkf_era5, 'ens_files_cgts, ens_files_cgts_extt, gevsevlev_files, event_year, param_bounds_file'),
                'ERA5',
                *dtoa(wkf, '''
                cgs_levels, daily_stat,
                event_region, select_regions,
                boot_type, confint_width,
                figdir, figfile_tag, figtitle_affix, gcm,
                fc_dates, fc_date, onset_date, term_date, 
                prob_symb, ext_sign, ext_symb, leq_symb, ineq_symb
                '''),
                ref_is_different=True,
                )
    print(psutil.virtual_memory())
    return 

if __name__ == "__main__":
    gcm2institute = all_gcms_institutes()
    gcms = list(gcm2institute.keys())
    gcms2ignore = ["BCC-CSM2-HR","SPEAR","NAVGEM"]
    gcms2ignore += ["GLOBO","GEM-NEMO",]


    all_procedures = ["reduce","compare_expts","compare_gcms","sanity2019"]
    procedures = [sys.argv[1]]
    if not all([procedure in all_procedures for procedure in procedures]):
        raise ValueError("procedures is {procedures} but must be a subset of of {options}".format(procedures=procedures, options=all_procedures))

    # Pass in which procedure to do based on system arguments
    for which_ssw in ["feb2018","jan2019","sep2019"][0:3]:
        if "sep2019" == which_ssw:
            gcms2ignore += ["CanESM5"]
        idx_gcms = [i for i in range(len(gcms)) if ((gcms[i] not in gcms2ignore))]
        for procedure in procedures:
            print(f'{procedure = }, {sys.argv = }')
            if procedure in ["reduce","compare_expts"]:
                for (i_gcm,gcm) in enumerate(gcms):
                    if gcm in gcms2ignore:
                        continue
                    if (len(sys.argv) >= 3) and (i_gcm != int(sys.argv[2])):
                        continue
                    print(f"{gcm = }")
                    #if not (gcm in ["IFS",]): #"IFS" == gcm):
                    #    continue
                    if "reduce" == procedure:
                        for i_fcdate in range(2):
                            print(f"{i_fcdate = }")
                            for i_expt in range(3):
                                print(f"{i_expt = }")
                                reduce_gcm(which_ssw,i_gcm,i_expt,i_fcdate)
                                garbcol.collect()
                    elif "compare_expts" == procedure:
                        compare_expts(which_ssw, i_gcm)
            elif "compare_gcms" == procedure:
                compare_gcms(which_ssw, idx_gcms)
            if "sep2019" == which_ssw and "sanity2019" == procedure:
                for (i_gcm,gcm) in enumerate(gcms):
                    if gcm in gcms2ignore:
                        continue
                    #if not ("IFS" == gcm):
                    #    continue
                    #sanity_check_2019(i_gcm)

