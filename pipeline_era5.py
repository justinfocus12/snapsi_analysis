# The pipeline for calculating the hazard functions for ERA5 data

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
from importlib import reload
import utils; reload(utils)
import pipeline_base; reload(pipeline_base)
import stat_functions as stfu; reload(stfu)

def analysis_multiparams():
    # lon/lat ratios are 6/1 at the bottom and 4/1 at the top; stick to 5/1 
    cgs_levels = [(1,1),(5,1),(10,2),(20,4),(40,8),(141,16)]
    return cgs_levels

def era5_workflow(verbose=False):
    print(f'Starting workflow setup')
    raw_data_dir = '/gws/nopw/j04/snapsi/processed/wg2/ju26596/era5'
    years = np.arange(1979,2019,dtype=int)
    year_filegroups = []
    event_time_interval = [datetime.datetime(2018,2,21,0), datetime.datetime(2018,3,8,22)] # for the reference year 
    for year in years:
        year_filegroups.append(tuple(
            join(raw_data_dir,f't2m_nhem_{year:04}-{month:02}.nc')
            for month in [1,2,3]
            ))

    event_region = dict(lat=slice(50,65),lon=slice(-10,130))

    reduced_data_dir = join('/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018/results_2024-05-04','era5')
    makedirs(reduced_data_dir,exist_ok=True)

    # spatial coarse graining (cgs)
    cgs_levels = analysis_multiparams()


    figdir = f'/home/users/ju26596/snapsi_analysis_figures/feb2018/figures_2024-05-04/era5'
    makedirs(figdir,exist_ok=True)

    # Select regions for more detailed GEV analysis (based on visualizing the map)
    # format is (cgs_level, i_lon, i_lat)
    # TODO change this to a list of floating point locations, which will be part of different-sized neighborhoods as a function of resolution 
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
    # TODO specify temperature levels for relative risk 
    workflow = (
            years,event_region,event_time_interval,
            year_filegroups,reduced_data_dir,figdir,
            cgs_levels,select_regions,risk_levels
            )
    print(f'Finished setting up workflow')
    return workflow

def coarse_grain_time(years, year_filegroups, event_region, event_time_interval):
    print(f'Starting to coarse-grain time')
    t2m = []
    t0_ref,t1_ref = event_time_interval
    event_duration = t1_ref - t0_ref
    for i_year,year in enumerate(years):
        print(f'Ingesting year {year}')
        # Modify the time selection
        t0 = datetime.datetime(year, t0_ref.month, t0_ref.day, t0_ref.hour)
        t1 = t0 + event_duration
        t2m_year = (
                xr.concat([xr.open_dataarray(yf) for yf in year_filegroups[i_year]], dim='time')
                .isel(latitude=slice(None,None,-1)) # Flip lat to go in increasing order
                .rename(longitude='lon',latitude='lat')
                )
        t2m_year = (
                utils.rezero_lons(
                    t2m_year
                    .sel(time=slice(t0,t1))
                    .assign_coords(time=np.arange(t0_ref,t1_ref,datetime.timedelta(hours=6)))
                    )
                .sel(event_region)
                .expand_dims(member=[year])
                )
        print(f'{t2m_year.shape = }, {t2m_year.dims = }')
        t2m.append(t2m_year)
    t2m = xr.concat(t2m,dim='member') 
    # Take daily mean
    daily_mean = (
            t2m 
            .coarsen({'time': 4}, side='left', coord_func='min')
            .mean()
            ).compute()
    daily_min = (
            t2m
            .coarsen({'time': 4}, side='left', coord_func='min')
            .min()
            ).compute()
    t2m_cgt = xr.concat([daily_mean,daily_min], dim='daily_stat').assign_coords(daily_stat=['daily_mean','daily_min'])
    print(f'{t2m_cgt.dims = }, {t2m_cgt.shape = }')
    return t2m_cgt

def reduce_era5():
    tododict = dict({
        'coarse_grain_time':           0,
        'plot_t2m_sumstats_map':       0,
        'coarse_grain_space':          0,
        'fit_gev':                     0,
        'plot_statpar_map':            0,
        'fit_gev_select_regions':      1,
        'plot_gev_select_regions':     1,
        })
    years,event_region,event_time_interval,year_filegroups,reduced_data_dir,figdir,cgs_levels,select_regions,risk_levels = era5_workflow()
    ens_file_cgt = join(reduced_data_dir,f't2m_cgt1day.nc')
    if tododict['coarse_grain_time']:
        ds_cgt = coarse_grain_time(years, year_filegroups, event_region, event_time_interval)
        ds_cgt.to_netcdf(ens_file_cgt)
    else:
        ds_cgt = xr.open_dataarray(ens_file_cgt)

    ds_cgt_mint = ds_cgt.min(dim='time')
    if tododict['plot_t2m_sumstats_map']:
        for daily_stat in ['daily_min','daily_mean']:
            fig,axes = pipeline_base.plot_sumstats_map(ds_cgt_mint.sel(daily_stat=daily_stat))
            fig.suptitle(f'ERA5 {daily_stat}')
            fig.savefig(join(figdir,f't2m_sumstats_map_{daily_stat}.png'), **pltkwargs)
            plt.close(fig)

    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        ens_file_cgts = join(reduced_data_dir,f't2m_cgt1day_cgs{cgs_key}.nc')
        if tododict['coarse_grain_space']:
            ds_cgts = pipeline_base.coarse_grain_space(ds_cgt, cgs_level)
            ds_cgts.to_netcdf(ens_file_cgts)
        else:
            ds_cgts = xr.open_dataarray(ens_file_cgts)
        # ---- Minimize in time -----------------------------
        ds_cgts_mint = ds_cgts.min(dim='time')
        print(f'{ds_cgts_mint.dims = }')
        # ----------- Perform GEV fitting (on negative temperature) --------------
        gev_param_file = join(reduced_data_dir,f'gevpar_cgs{cgs_key}.nc')
        if tododict['fit_gev']:
            gevpar = pipeline_base.fit_gev_mintemp(ds_cgts_mint)
            gevpar.to_netcdf(gev_param_file)
        else:
            gevpar = xr.open_dataarray(gev_param_file)
        if tododict['plot_statpar_map'] and min(cgs_level) > 1:
            for daily_stat in ['daily_mean']:
                fig,axes = pipeline_base.plot_statpar_map(ds_cgts_mint.sel(daily_stat=daily_stat), gevpar.sel(daily_stat=daily_stat))
                fig.suptitle(f'ERA5 {daily_stat}')
                fig.savefig(join(figdir,f'statpar_map_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
                plt.close(fig)


        # Do a more thorough uncertainty quantification at a specific site or region, using bootstrap analysis and goodness-of-fit etc. Maybe parallelize over all sites too

        for (i_lon,i_lat) in select_regions[i_cgs_level]:
            mintemp = ds_cgts_mint.sel(daily_stat='daily_mean').isel(lon=i_lon,lat=i_lat).to_numpy()
            if tododict['fit_gev_select_regions']:
                gevpar_reg,mintemp_levels_reg = pipeline_base.fit_gev_mintemp_1d_uq(mintemp,risk_levels)
                gevpar_reg.to_netcdf(join(reduced_data_dir,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                np.save(f'mintemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy', mintemp_levels_reg)
            else:
                mintemp_levels_reg = np.load(f'mintemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy')
                gevpar_reg = xr.open_dataarray(join(reduced_data_dir,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))

            if tododict['plot_gev_select_regions']:
                fig,axes = plt.subplots(ncols=2,figsize=(10,4),sharey=True)
                # Left: probability plot; right: timeseries of minima over years 
                ax = axes[0]
                order = np.argsort(mintemp)
                rank = np.argsort(order)
                risk_empirical = np.arange(1,len(mintemp)+1)/len(mintemp)
                ax.scatter(risk_empirical, mintemp[order], color='black', marker='+')
                # Special marker for 2018
                i_mem_2018 = np.where(ds_cgts_mint.member == 2018)[0][0]

                ax.scatter(risk_empirical[rank[i_mem_2018]], mintemp[i_mem_2018], color='black', marker='o')
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

                ax = axes[1]
                ax.plot(ds_cgts_mint['member'].to_numpy(), mintemp, color='black', marker='+')
                ax.set(xlabel='Year',ylabel='')
                ax.yaxis.set_tick_params(which='both',labelbottom=True)
                fig.savefig(join(figdir,f'riskplot_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                plt.close(fig)
    ds_cgt.close()
    ds_cgt_mint.close()

if __name__ == '__main__':
    print(f'Starting main')
    reduce_era5()








