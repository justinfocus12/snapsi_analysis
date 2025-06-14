# The pipeline for calculating the hazard functions for ERA5 data

import numpy as np
import xarray as xr 
from cartopy import crs as ccrs
import netCDF4
from matplotlib import pyplot as plt, rcParams, ticker
import pdb
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


def era5_workflow(which_ssw,verbose=False):
    # 0. Global constants
    fc_dates,onset_date_nominal,term_date = pipeline_base.dates_of_interest(which_ssw)
    onset_date = pipeline_base.least_sensible_onset_date(which_ssw)
    event_region,context_region = pipeline_base.region_of_interest(which_ssw)

    # 1. Raw data
    raw_data_dir = '/gws/nopw/j04/snapsi/processed/wg2/ju26596/era5'

    # 2. Processed data
    analysis_date = '2025-05-20'
    processed_data_dir = '/gws/nopw/j04/snapsi/processed/wg2/ju26596'
    reduced_data_dir = join(processed_data_dir,which_ssw,analysis_date,'era5')
    figdir = join('/home/users/ju26596/snapsi_analysis_figures',which_ssw,analysis_date,'era5')
    makedirs(reduced_data_dir,exist_ok=True)
    makedirs(figdir,exist_ok=True)

    # 3. Files
    landmask_full_file = join(processed_data_dir,'era5','land_sea_mask.nc')
    landmask_interp_file = join(reduced_data_dir,'land_sea_mask_interp.nc')

    daily_stat = 'daily_min'
    years = np.arange(1980,2020,dtype=int)
    year_filegroups = []
    ens_file_cgt = join(reduced_data_dir,'t2m_cgt1day.nc')
    ens_files_cgts = []
    ens_files_cgts_extt = [] # extremized in time 
    gevpar_files = []
    risk_files = []
    valatrisk_files = []
    gevsevlev_files = []
    cgs_levels,select_regions = pipeline_base.analysis_multiparams(which_ssw)
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        ens_files_cgts.append(join(reduced_data_dir,'t2m_cgt1day_cgs%dx%d.nc'%(cgs_level[0],cgs_level[1])))
        ens_files_cgts_extt.append(join(reduced_data_dir,'t2m_cgt1day_cgs%dx%d_extt.nc'%(cgs_level[0],cgs_level[1])))
        gevpar_files.append(join(reduced_data_dir,'gevpar_cgs%dx%d.nc'%(cgs_level[0],cgs_level[1])))
        risk_files.append(join(reduced_data_dir,'risk_cgs%dx%d.nc'%(cgs_level[0],cgs_level[1])))
        valatrisk_files.append(join(reduced_data_dir,'valatrisk_cgs%dx%d.nc'%(cgs_level[0],cgs_level[1])))
        gevsevlev_files.append([])
        for (i_lon,i_lat) in select_regions[i_cgs_level]:
            gevsevlev_files[i_cgs_level].append(join(reduced_data_dir,'gevsevlev_cgs%dx%d_ilon%d_ilat%d.nc'%(cgs_level[0],cgs_level[1],i_lon,i_lat)))

    param_bounds_file = join(reduced_data_dir, 'param_bounds.nc')

    if "feb2018" == which_ssw:
        event_year = 2018
        ext_sign = -1
        for year in years:
            year_filegroups.append(tuple(
                [join(raw_data_dir,f't2m_{(year-1):04}-12.nc')] + 
                [join(raw_data_dir,f't2m_{year:04}-{month:02}.nc') for month in [1,2,3]]
                ))
        Nlon_interp = 80
        Nlat_interp = 16
    elif "jan2019" == which_ssw:
        event_year = 2019
        ext_sign = -1
        for year in years:
            year_filegroups.append(tuple(
                [join(raw_data_dir,f't2m_{(year-1):04}-12.nc')] + 
                [join(raw_data_dir,f't2m_{year:04}-{month:02}.nc') for month in [1,2,3]]
                ))
    elif "sep2019" == which_ssw:
        event_year = 2019
        ext_sign = 1
        for year in years:
            year_filegroups.append(tuple(
                [join(raw_data_dir,f't2m_{year:04}-{month:02}.nc') for month in [9,10,11]]
                ))
    n_boot = 1000
    confint_width = 0.5
    boot_type = 'percentile'

    ext_symb = "max" if 1==ext_sign else "min"
    leq_symb = "\u2264"
    geq_symb = "\u2265"
    ineq_symb = geq_symb if 1==ext_sign else leq_symb
    prob_symb = "\u2119"


    select_points = ()
    risk_levels = np.exp(np.linspace(np.log(0.001),np.log(49/50),30))
    workflow = dict(
            years             =years,
            event_year        =event_year,
            ext_sign          =ext_sign,
            ext_symb = ext_symb,
            ineq_symb = ineq_symb,
            leq_symb = leq_symb,
            prob_symb = prob_symb,
            figtitle_affix = "ERA5",
            figfile_tag = "era5",
            event_region      =event_region,
            context_region    =context_region,
            Nlon_interp = Nlon_interp,
            Nlat_interp = Nlat_interp,
            fc_dates          =fc_dates,
            init_date = fc_dates[0],
            onset_date_nominal=onset_date_nominal,
            onset_date = onset_date,
            term_date         =term_date,
            # files at each stage of processing
            landmask_full_file     =landmask_full_file,
            landmask_interp_file     =landmask_interp_file,
            param_bounds_file = param_bounds_file,
            ens_file_cgt = ens_file_cgt, 
            ens_files_cgts = ens_files_cgts,
            ens_files_cgts_extt = ens_files_cgts_extt,
            gevpar_files = gevpar_files,
            risk_files = risk_files,
            valatrisk_files = valatrisk_files,
            gevsevlev_files = gevsevlev_files,
            year_filegroups   =year_filegroups,
            reduced_data_dir  =reduced_data_dir,
            figdir            =figdir,
            cgs_levels        =cgs_levels,
            select_regions    =select_regions,
            daily_stat        =daily_stat,
            risk_levels       =risk_levels,
            n_boot            =n_boot,
            confint_width     =confint_width,
            boot_type = 'percentile', 
            )
    print(f'Finished setting up workflow')
    return workflow

def interpolate_landmask(landmask_full_file, landmask_interp_file, Nlon_interp, Nlat_interp, event_region):
    landmask_full = (
            utils.rezero_lons(
                xr.open_dataarray(landmask_full_file)
                .isel(time=0,drop=True)
                .isel(latitude=slice(None,None,-1)) # Flip lat to go in increasing order
                .rename(dict(latitude='lat',longitude='lon'))
                )
            #.sel(event_region)
            )
    dlon = (event_region['lon'].stop - event_region['lon'].start)/Nlon_interp
    dlat = (event_region['lat'].stop - event_region['lat'].start)/Nlat_interp
    lons_interp = np.linspace(event_region['lon'].start+dlon/2, event_region['lon'].stop-dlon/2, Nlon_interp)
    lats_interp = np.linspace(event_region['lat'].start+dlat/2, event_region['lat'].stop-dlat/2, Nlat_interp)
    landmask_interp = landmask_full.interp({'lat': lats_interp, 'lon': lons_interp}).sel(event_region)
    assert np.all(np.isfinite(landmask_interp))
    landmask_interp.to_netcdf(landmask_interp_file)
    return 



def coarse_grain_time(years, year_filegroups, region, context_region, Nlon_interp, Nlat_interp, init_date, term_date, ens_file_cgt):
    print(f'Starting to coarse-grain time')
    t2m = []
    duration = term_date - init_date
    # Regrid to an easily divisible size 
    # The physical aspect ratio ranges from 6/1 at the lower boundary to 4/1 at the top, so we settle on a ratio of 5/1

    dlon = (region['lon'].stop - region['lon'].start)/Nlon_interp
    dlat = (region['lat'].stop - region['lat'].start)/Nlat_interp

    lon_interp = np.linspace(region['lon'].start+dlon/2, region['lon'].stop-dlon/2, Nlon_interp)
    lat_interp = np.linspace(region['lat'].start+dlat/2, region['lat'].stop-dlat/2, Nlat_interp)


    for i_year,year in enumerate(years):
        print(f'Ingesting year {year}')
        print(f'{year_filegroups[i_year] = }')
        # Modify the time selection
        t0 = dtlib.datetime.replace(init_date, year=year) #dtlib.datetime(year, init_date.month, init_date.day, t0_ref.hour)
        t1 = t0 + duration
        t2m_year = (
                xr.concat([
                    xr.open_dataarray(yf).rename({'valid_time': 'time'}) 
                    for yf in year_filegroups[i_year]
                    ], dim='time')

                .isel(latitude=slice(None,None,-1)) # Flip lat to go in increasing order
                .rename(longitude='lon',latitude='lat')
                )
        t2m_year = (
                utils.rezero_lons(
                    t2m_year
                    .sel(time=slice(t0,t1))
                    .assign_coords(time=np.arange(init_date,term_date,dtlib.timedelta(hours=6)))
                    )
                .sel(context_region)
                .interp(lon=lon_interp, lat=lat_interp, method="linear")
                .sel(region)
                .expand_dims(member=[year])
                )
        print(f'{t2m_year.shape = }, {t2m_year.dims = }')
        t2m.append(t2m_year)
    t2m = xr.concat(t2m,dim='member') 
    # Take daily mean
    daily_mean = t2m.isel(time=range(0,t2m['time'].size,4))
    daily_min = t2m.isel(time=range(0,t2m['time'].size,4))
    daily_max = t2m.isel(time=range(0,t2m['time'].size,4))
    for i in range(3):
        daily_mean += t2m.isel(time=range(i,t2m['time'].size,4)).assign_coords({'time': daily_mean['time']})
        daily_min = np.minimum(daily_min, t2m.isel(time=range(i,t2m['time'].size,4)).assign_coords({'time': daily_min['time']}))
        daily_max = np.maximum(daily_max, t2m.isel(time=range(i,t2m['time'].size,4)).assign_coords({'time': daily_max['time']}))
    daily_mean /= 4
    t2m_cgt_1xday = xr.concat([daily_mean,daily_min,daily_max], dim='daily_stat').assign_coords(daily_stat=['daily_mean','daily_min','daily_max'])
    ds_cgt = xr.Dataset(data_vars=dict({'1xday': t2m_cgt_1xday, '4xday': t2m.rename({'time': 'time_6h'})}))
    ds_cgt.to_netcdf(ens_file_cgt)
    return 

def reduce_era5(which_ssw):
    todo = dict({
        'interpolate_landmask':             0,
        'coarse_grain_time':                0,
        'coarse_grain_space':               0,
        'set_param_bounds':                 0,
        'onset_date_sensitivity_analysis':  0,
        'compute_severities':               0,
        'plot_sumstats_map':                0,
        'fit_gev':                          0,
        'plot_gevpar_map':                  0,
        'compute_risk':                     0,
        'plot_risk_map':                    0,
        'fit_gev_select_regions':           1,
        'plot_gev_select_regions':          0,
        })
    wkf = era5_workflow(which_ssw)
    if todo['interpolate_landmask']:
        interpolate_landmask(wkf['landmask_full_file'], wkf['landmask_interp_file'], wkf['Nlon_interp'], wkf['Nlat_interp'], wkf['event_region'])
    if todo['coarse_grain_time']:
        coarse_grain_time(wkf['years'], wkf['year_filegroups'], wkf['event_region'], wkf['context_region'], wkf['Nlon_interp'], wkf['Nlat_interp'], wkf['fc_dates'][0], wkf['term_date'], wkf['ens_file_cgt'])
    if todo['coarse_grain_space']:
        pipeline_base.coarse_grain_space(wkf['ens_file_cgt'], wkf['ens_files_cgts'], wkf['cgs_levels'], wkf['landmask_interp_file'], wkf['event_region'])


    if todo['set_param_bounds']:
        pipeline_base.set_param_bounds(
                *dtoa(wkf, '''
                ens_file_cgt, param_bounds_file,landmask_interp_file,daily_stat,onset_date,term_date,ext_sign
                ''')
                )
    # ------------- Sensitivity analysis with respect to onset date ------------
    if todo['onset_date_sensitivity_analysis']:
        pipeline_base.onset_date_sensitivity_analysis(
                *dtoa(wkf, '''
                ens_files_cgts, event_region, cgs_levels, fc_dates,
                init_date, onset_date_nominal, term_date, daily_stat,
                ext_sign, figdir, figtitle_affix, figfile_tag,
                ens_files_cgts, event_year,
                '''),
                idx_cgs_levels=[0,1]
                )
    # ------------ extremize in time -------------------
    if todo['compute_severities']:
        pipeline_base.compute_severity_from_intensity(
                *dtoa(wkf, ''' 
                ens_files_cgts, ens_files_cgts_extt, cgs_levels, 
                ext_sign, onset_date, term_date, daily_stat, 
                landmask_interp_file
                ''')
                )
    # choose an onset date based on this 
    # --------------------------------------------------------------------------
    if todo['plot_sumstats_map']:
        pipeline_base.plot_sumstats_maps_flat(
                *dtoa(wkf, '''
                ens_files_cgts_extt, ens_files_cgts_extt,
                event_year, event_year,
                ext_sign, param_bounds_file, cgs_levels,
                ext_symb, onset_date, term_date,
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
                *dtoa(wkf, '''
                ens_files_cgts_extt, event_year, gevpar_files, gevpar_files, valatrisk_files, cgs_levels, ext_sign
                ''')
                )
        #pipeline_base.compute_risk(
        #        *dtoa(wkf, '''
        #        gevpar_files, risk_files, ens_files_cgts_extt,
        #        event_year,ext_sign,cgs_levels
        #        '''),
        #        )
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
    # Do a more thorough uncertainty quantification at a specific site or region, using bootstrap analysis and goodness-of-fit etc. Maybe parallelize over all sites too
    if todo['fit_gev_select_regions']:
        pipeline_base.fit_gev_select_regions(
                *dtoa(wkf, '''
                ens_files_cgts_extt, event_year, 
                ens_files_cgts_extt, gevsevlev_files, risk_levels, 
                cgs_levels, select_regions,
                ext_sign,
                '''),
                )

    if todo['plot_gev_select_regions']:
        pipeline_base.plot_gevsevlev_select_regions(
                *(wkf[key.strip()] for key in '''
                ens_files_cgts, ens_files_cgts_extt, gevsevlev_files, 
                ens_files_cgts, ens_files_cgts_extt, event_year, param_bounds_file,
                cgs_levels, daily_stat,
                event_region, select_regions, 
                boot_type, confint_width,
                figdir, figfile_tag, figtitle_affix, onset_date, term_date,
                prob_symb, ext_sign, ext_symb, leq_symb, ineq_symb
                '''.split(',')),
                ref_is_different=False,
                )
    return 

if __name__ == '__main__':
    print(f'Starting main')
    for which_ssw in ["feb2018","jan2019","sep2019"][:1]:
        result = reduce_era5(which_ssw)








