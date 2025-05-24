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
import utils; reload(utils)
import pipeline_base; reload(pipeline_base)
import stat_functions as stfu; reload(stfu)

def analysis_multiparams(which_ssw):
    # lon/lat ratios are 6/1 at the bottom and 4/1 at the top; stick to 5/1 
    if "feb2018" == which_ssw:
        cgs_levels = [(1,1),(5,1),(10,2),(20,4),(40,8),(80,16)] #,(141,16)]
        select_regions = ( # Indexed by cgs_level
                [(0,0),], # level (1,1)
                [], # level (5,1)
                [(i,j) for i in range(10) for j in range(2)],
                [], # level (20,4)
                [], # level (40,8)
                [], # level (80,16)
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

def era5_workflow(which_ssw,verbose=False):
    print(f'Starting workflow setup')
    analysis_date = '2025-05-20'
    raw_data_dir = '/gws/nopw/j04/snapsi/processed/wg2/ju26596/era5'
    landmask_file = '/gws/nopw/j04/snapsi/processed/wg2/ju26596/era5/land_sea_mask.nc'
    processed_data_dir = '/gws/nopw/j04/snapsi/processed/wg2/ju26596'
    daily_stat = 'daily_min'
    years = np.arange(1980,2020,dtype=int)
    year_filegroups = []
    reduced_data_dir = join(processed_data_dir,which_ssw,analysis_date,'era5')
    figdir = join('/home/users/ju26596/snapsi_analysis_figures',which_ssw,analysis_date,'era5')
    fc_dates,onset_date_nominal,term_date = pipeline_base.dates_of_interest(which_ssw)
    event_region,context_region = pipeline_base.region_of_interest(which_ssw)
    if "feb2018" == which_ssw:
        event_year = 2018
        ext_sign = -1
        for year in years:
            year_filegroups.append(tuple(
                [join(raw_data_dir,f't2m_{(year-1):04}-12.nc')] + 
                [join(raw_data_dir,f't2m_{year:04}-{month:02}.nc') for month in [1,2,3]]
                ))
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
    makedirs(reduced_data_dir,exist_ok=True)
    # spatial coarse graining (cgs)
    cgs_levels,select_regions = analysis_multiparams(which_ssw)
    n_boot = 1000
    confint_width = 0.5

    makedirs(figdir,exist_ok=True)

    select_points = ()
    risk_levels = np.exp(np.linspace(np.log(0.001),np.log(49/50),30))
    workflow = (
            years,
            event_year,ext_sign,
            event_region,context_region,
            fc_dates,onset_date_nominal,term_date,
            landmask_file,year_filegroups,reduced_data_dir,figdir,
            cgs_levels,select_regions,daily_stat,
            risk_levels,n_boot,confint_width
            )
    print(f'Finished setting up workflow')
    return workflow

def coarse_grain_time_fulltime_solo(which_ssw, i_init): 
    years,event_region,event_time_interval,landmask_file,year_filegroups,reduced_data_dir,figdir,cgs_levels,select_regions,risk_levels,n_boot,confint_width = era5_workflow(which_ssw)
    inits,fin = pipeline_base.dates_of_interest(which_ssw)
    event_region = pipeline_base.region_of_interest(which_ssw)
    fcdate = dtlib.datetime.strptime(inits[i_init],'%Y%m%d')
    full_time_interval = [fcdate, event_time_interval[1]]
    ds_cgt_era5,_ = coarse_grain_time(years, year_filegroups, event_region, full_time_interval)
    return ds_cgt_era5


def coarse_grain_time(years, year_filegroups, region, context_region, init_date, term_date):
    print(f'Starting to coarse-grain time')
    t2m = []
    duration = term_date - init_date
    # Regrid to an easily divisible size 
    # The physical aspect ratio ranges from 6/1 at the lower boundary to 4/1 at the top, so we settle on a ratio of 5/1
    Nlon_interp = 80
    Nlat_interp = 16 

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
    ds = xr.Dataset(data_vars=dict({'1xday': t2m_cgt_1xday, '4xday': t2m.rename({'time': 'time_6h'})}))
    return (ds, False)

def reduce_era5(which_ssw):
    todo = dict({
        'coarse_grain_time':                0,
        'coarse_grain_space':               0,
        'onset_date_sensitivity_analysis':  0,
        'plot_sumstats_map':                0,
        'fit_gev':                          0,
        'plot_gevpar_map':                  1,
        'compute_risk':                     0,
        'plot_risk_map':                    0,
        'fit_gev_select_regions':           0,
        'plot_gev_select_regions':          0,
        })
    (
        years,
        event_year, ext_sign, 
        event_region,context_region,
        fc_dates,onset_date_nominal,term_date,
        landmask_file, year_filegroups, reduced_data_dir, figdir,
        cgs_levels,select_regions,daily_stat,
        risk_levels,n_boot,confint_width
    ) = era5_workflow(which_ssw)
    ext_symb = "max" if 1==ext_sign else "min"
    ineq_sign = "geq" if 1==ext_sign else "leq"

    boot_type = 'percentile'
    ens_file_cgt = join(reduced_data_dir,f't2m_cgt1day.nc')
    if todo['coarse_grain_time']:
        ds_cgt,err_flag = coarse_grain_time(years, year_filegroups, event_region, context_region, fc_dates[0], term_date)
        ds_cgt.to_netcdf(ens_file_cgt)

    ds_cgt = xr.open_dataset(ens_file_cgt)
    da_cgt = ds_cgt['1xday'].sel(daily_stat=daily_stat)

    landmask_full = (
            utils.rezero_lons(
                xr.open_dataarray(landmask_file)
                .isel(time=0,drop=True)
                .isel(latitude=slice(None,None,-1)) # Flip lat to go in increasing order
                .rename(dict(latitude='lat',longitude='lon'))
                )
            #.sel(event_region)
            )
    landmask = landmask_full.interp({'lat': da_cgt.coords['lat'].values, 'lon': da_cgt.coords['lon'].values}).sel(event_region)

    assert np.all(np.isfinite(landmask))

    if todo['coarse_grain_space']:
        for i_cgs_level,cgs_level in enumerate(cgs_levels):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            ens_file_cgts = join(reduced_data_dir,f't2m_cgt1day_cgs{cgs_key}.nc')
            if todo['coarse_grain_space']:
                ds_cgts = pipeline_base.coarse_grain_space(ds_cgt, cgs_level, landmask)
                ds_cgts.to_netcdf(ens_file_cgts)

    # ------------- Sensitivity analysis with respect to onset date ------------
    if todo['onset_date_sensitivity_analysis']:
        for i_cgs_level,cgs_level in enumerate(cgs_levels[:2]):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            ens_file_cgts = join(reduced_data_dir,f't2m_cgt1day_cgs{cgs_key}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday'].sel(daily_stat=daily_stat)
            e5min,e5max = np.nanmin(da_cgts),np.nanmax(da_cgts)
            e5mid = 0.5*(e5min+e5max)
            vmin,vmax = e5mid + 1.25*np.array([-1,1])*(e5max-e5min)/2
            onset_date_minsens = pipeline_base.onset_date_sensitivity_analysis(
                    da_cgts,
                    event_region,cgs_level, 
                    fc_dates, fc_dates[0], onset_date_nominal, term_date, ext_sign, 
                    figdir, f'cgs{cgs_key}', "ERA5", mem_special=2018, fc_date_special=None,
                    intensity_lims=[vmin,vmax]
                    )
    # choose an onset date based on this 
    # --------------------------------------------------------------------------
    onset_date = pipeline_base.least_sensible_onset_date(which_ssw)
    da_cgt_extt = ext_sign * (ext_sign*da_cgt.sel(time=slice(onset_date,term_date))).max(dim='time')
    # Set global bounds on plots (sign might flip)
    param_bounds = dict({
        'loc': utils.padded_bounds(ext_sign*da_cgt_extt.where(landmask>0, np.nan).mean(dim='member')),
        'scale': utils.padded_bounds(da_cgt_extt.where(landmask>0, np.nan).std(dim='member')),
        'shape': np.array([-0.5,0.1]),
        })

    if todo['plot_sumstats_map']:
        fmtfun = lambda date: dtlib.datetime.strftime(date, "%m/%d")
        titles = [
                r"ERA5 %s {T2M($t$): %s$\leq t\leq$%s}, %d-%d mean"%(ext_symb, fmtfun(onset_date), fmtfun(term_date), years[0], years[-1]),
                r"ERA5 %s {T2M($t$): %s$\leq t\leq$%s}, %d-%d std. dev."%(ext_symb, fmtfun(onset_date), fmtfun(term_date), years[0], years[-1]),
                r"ERA5 %s {T2M($t$): %s$\leq t\leq$%s}, %d standardized anomaly"%(ext_symb, fmtfun(onset_date), fmtfun(term_date), event_year)
                ]
        fig,axes = pipeline_base.plot_sumstats_maps_flat(
                *((da_cgt_extt,)*2),
                landmask,
                event_year, event_year,
                titles, 
                cgs_levels[2],
                ext_sign=ext_sign,
                param_bounds=param_bounds
                )
        fig.savefig(join(figdir,f'sumstats_map_{daily_stat}.png'), **pltkwargs)
        plt.close(fig)
    if todo['fit_gev']:
        # Also do it for the un-coarsened data
        gevpar_cgt = pipeline_base.fit_gev_exttemp(da_cgt_extt, ext_sign, method='PWM')
        gevpar_file_cgt = join(reduced_data_dir,f'gevpar_cgt1xday.nc')
        gevpar_cgt.to_netcdf(gevpar_file_cgt)

        for i_cgs_level,cgs_level in enumerate(cgs_levels):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            ens_file_cgts = join(reduced_data_dir,f't2m_cgt1day_cgs{cgs_key}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday'].sel(daily_stat=daily_stat)
            da_cgts_extt = ext_sign * (ext_sign * da_cgts.sel(time=slice(onset_date,term_date))).max(dim='time')
            print(f'{da_cgts_extt.dims = }')
            # ----------- Perform GEV fitting (on negative temperature) --------------
            gevpar_file_cgts = join(reduced_data_dir,f'gevpar_cgt1xday_cgs{cgs_key}.nc')
            gevpar_cgts = pipeline_base.fit_gev_exttemp(da_cgts_extt,ext_sign,method='PWM')
            gevpar_cgts.to_netcdf(gevpar_file_cgts)
        # ----------------- Compute risk w.r.t. the event year ---------------
    if todo['plot_gevpar_map']:
        # First the non-coarse-grained version
        for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
            if min(cgs_level) <= 1:
                continue
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            gevpar_file_cgts = join(reduced_data_dir,f'gevpar_cgt1xday_cgs{cgs_key}.nc')
            gevpar_cgts = xr.open_dataarray(gevpar_file_cgts)
            fmtfun = lambda date: dtlib.datetime.strftime(date, "%m/%d")
            titles = [
                    r"ERA5, %s {T2M(t): %s$\leq t\leq$%s}, location $\mu$"%(ext_symb, fmtfun(onset_date), fmtfun(term_date)),
                    r"ERA5, %s {T2M(t): %s$\leq t\leq$%s}, scale $\sigma$"%(ext_symb, fmtfun(onset_date), fmtfun(term_date)),
                    r"ERA5, %s {T2M($t$): %s$\leq t\leq$%s}, shape $\xi$"%(ext_symb, fmtfun(onset_date), fmtfun(term_date))
                    ]
            # TODO plot a map for each level of coarse-graining, not just the full un-coarsened map 
            fig,axes = pipeline_base.plot_gevpar_maps_flat(
                    gevpar_cgts,
                    titles, 
                    cgs_levels[2],
                    ext_sign,
                    landmask=landmask if i_cgs_level==0 else None,
                    param_bounds=param_bounds,
                    )
            fig.savefig(join(figdir,f'gevpar_map_{daily_stat}_cgs{cgs_key}.png'), **pltkwargs)
            plt.close(fig)
    if todo['compute_risk']:
        for i_cgs_level,cgs_level in enumerate(cgs_levels):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            ens_file_cgts = join(reduced_data_dir,f't2m_cgt1day_cgs{cgs_key}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday']
            da_cgts_extt = ext_sign * (ext_sign*da_cgts.sel(time=slice(onset_date,term_date))).max(dim='time')
            gevpar_file_cgts = join(reduced_data_dir,f'gevpar_cgt1xday_cgs{cgs_key}.nc')
            gevpar = xr.open_dataarray(gevpar_file_cgts)
            risk_file = join(reduced_data_dir,f'risk_wrt{event_year}_cgs{cgs_key}.nc')
            dskwargs = dict(daily_stat=daily_stat,drop=True)
            risk = pipeline_base.compute_risk(
                    da_cgts_extt.sel(**dskwargs),
                    da_cgts_extt.sel(member=event_year,**dskwargs),
                    gevpar,
                    gevpar,
                    locsign=ext_sign)
            risk.to_netcdf(risk_file)
    if todo['plot_risk_map']:
        for i_cgs_level,cgs_level in enumerate(cgs_levels):
            if min(cgs_level) < 2:
                continue
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            risk_file = join(reduced_data_dir,f'risk_wrt{event_year}_cgs{cgs_key}.nc')
            risk = xr.open_dataarray(risk_file)
            fig,ax = pipeline_base.plot_risk_map(risk,locsign=ext_sign,projection='mercator')
            ineq_sign = "geq" if ext_sign==1 else "leq"
            ax.set_title(r"$\mathbb{P}_{\mathrm{ERA5}}\{\%s_t\langle T(t)\rangle\%s \%s_t\langle T(t)\rangle_{\mathrm{ERA5,%s}}\}$"%(ext_symb,ineq_sign,ext_symb,event_year))
            fig.savefig(join(figdir,f'risk_map_cgs{cgs_key}_{daily_stat}.png'), **pltkwargs)
            plt.close(fig)
            print(f'Just plotted risk map with {cgs_key = }')



    # Do a more thorough uncertainty quantification at a specific site or region, using bootstrap analysis and goodness-of-fit etc. Maybe parallelize over all sites too
    if todo['fit_gev_select_regions']:
        for i_cgs_level,cgs_level in enumerate(cgs_levels):
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            ens_file_cgts = join(reduced_data_dir,f't2m_cgt1day_cgs{cgs_key}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday']
            da_cgts_extt = ext_sign * (ext_sign*da_cgts.sel(time=slice(onset_date,term_date))).max(dim='time')

            for (i_lon,i_lat) in select_regions[i_cgs_level]:
                exttemp = da_cgts_extt.sel(daily_stat='daily_mean').isel(lon=i_lon,lat=i_lat).to_numpy()
                gevpar_reg,exttemp_levels_reg = pipeline_base.fit_gev_exttemp_1d_uq(exttemp,risk_levels,ext_sign,method='PWM')
                gevpar_reg.to_netcdf(join(reduced_data_dir,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                np.save(join(reduced_data_dir,f'exttemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy'), exttemp_levels_reg)

    if todo['plot_gev_select_regions']:
        for i_cgs_level,cgs_level in enumerate(cgs_levels):
            if max(cgs_level) > 10:
                continue
            cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
            ens_file_cgts = join(reduced_data_dir,f't2m_cgt1day_cgs{cgs_key}.nc')
            da_cgts = xr.open_dataset(ens_file_cgts)['1xday']
            da_cgts_extt = ext_sign * (ext_sign*da_cgts.sel(time=slice(onset_date,term_date))).max(dim='time')
            # Location labeling 
            lon_blocksize,lat_blocksize = ((event_region[d].stop - event_region[d].start)/cgs_level[i_d] for (i_d,d) in enumerate(('lon','lat')))
            for (i_lon,i_lat) in select_regions[i_cgs_level]:
                exttemp = da_cgts_extt.sel(daily_stat='daily_mean').isel(lon=i_lon,lat=i_lat).to_numpy()
                gevpar_reg = xr.open_dataarray(join(reduced_data_dir,f'gevpar_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.nc'))
                exttemp_levels_reg = np.load(join(reduced_data_dir,f'exttemp_levels_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.npy')) 
                center_lon = event_region['lon'].start + (i_lon+0.5)*lon_blocksize
                center_lat = event_region['lat'].start + (i_lat+0.5)*lat_blocksize
                lonlatstr = r'$\lambda=%d\pm%d,\phi=%d\pm%d$'%(center_lon,lon_blocksize/2,center_lat,lat_blocksize/2)
                if max(cgs_level) == 1:
                    lonlatstr = r'%s (whole region)'%(lonlatstr)

                order = np.argsort(exttemp)
                rank = np.argsort(order)
                if ext_sign == -1:
                    risk_empirical = np.arange(1,len(exttemp)+1)/len(exttemp)
                else:
                    risk_empirical = np.arange(len(exttemp),0,-1)/len(exttemp)
                shape,loc,scale = (gevpar_reg.sel(param=p).isel(boot=0) for p in ('shape','loc','scale'))
                if np.all([np.isfinite(p) for p in [shape,loc,scale]]):
                    fig,ax = plt.subplots()
                    ax.scatter(risk_empirical, exttemp[order], color='black', marker='+')
                    param_label = '\n'.join([
                        r'$\mu=%.1f$'%(ext_sign*loc),
                        r'$\sigma=%.1f$'%(scale),
                        r'$\xi=%+.2f$'%(shape)
                        ])
                    # Special marker for the year 
                    i_mem_event_year = np.where(da_cgts_extt.member == event_year)[0][0]

                    ax.scatter(risk_empirical[rank[i_mem_event_year]], exttemp[i_mem_event_year], color='black', marker='o')
                    h, = ax.plot(risk_levels,exttemp_levels_reg[0,:],color='black', label=param_label)
                    boot_quant_lo,boot_quant_hi = (np.quantile(exttemp_levels_reg[1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
                    if boot_type == 'percentile':
                        lo,hi = boot_quant_lo,boot_quant_hi
                    else:
                        lo,hi = 2*risk_ratio[0,:]-boot_quant_hi, 2*risk_ratio[0,:]-boot_quant_lo
                    ax.fill_between(risk_levels, lo, hi, fc='gray', ec='none', alpha=0.3, zorder=-1)
                    ax.set_xscale('log')
                    ax.set_xlabel(r'$\mathbb{P}\{\%s_t\langle T(t)\rangle_{\mathrm{region}}\%s T\}$'%(ext_symb,ineq_sign))
                    ax.set_ylabel(r'$T$')
                    ax.set_title(f'ERA5 at {lonlatstr}')
                    ax.legend(handles=[h])

                    fig.savefig(join(figdir,f'riskplot_reg_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                    plt.close(fig)
    #ds_cgt.close()
    #ds_cgt_extt.close()
    return 

if __name__ == '__main__':
    print(f'Starting main')
    for which_ssw in ["feb2018","jan2019","sep2019"][:1]:
        result = reduce_era5(which_ssw)








