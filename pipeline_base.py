import numpy as np
import xarray as xr 
import datetime as dtlib
from scipy.stats import genextreme as spgex
import pdb
from cartopy import crs as ccrs, feature as cfeature
import netCDF4
from os.path import join, exists
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

from importlib import reload
import utils; reload(utils)
import stat_functions as stfu; reload(stfu)

def least_sensible_onset_date(which_ssw):
    if "feb2018" == which_ssw:
        onset_date = '20180213'
    else:
        raise NotImplementedError(f"Still need to choose onset date for ssw {which_ssw}")
    onset_date_dt = dtlib.datetime.strptime(onset_date,"%Y%m%d")
    return onset_date_dt

def dates_of_interest(which_ssw):
    # onset dates are subject to adjustment 
    if "feb2018" == which_ssw:
        fc_dates = ['20180125','20180208']
        onset_date_nominal = '20180221' 
        term_date = '20180308'
    elif "jan2019" == which_ssw:
        fc_dates = ['20181213','20190108']
        onset_date_nominal = '20190101'
        term_date = '20190131'
    elif "sep2019" == which_ssw:
        fc_dates = ['20190829','20191001']
        onset_date_nominal = '20191001'
        term_date = '20191014'
    # convert to datetime objects 
    fc_dates_dt = [dtlib.datetime.strptime(fc_date, "%Y%m%d").replace(hour=0) for fc_date in fc_dates]
    onset_date_nominal_dt = dtlib.datetime.strptime(onset_date_nominal, "%Y%m%d").replace(hour=0)
    term_date_dt = dtlib.datetime.strptime(term_date, "%Y%m%d").replace(hour=22)

    return fc_dates_dt, onset_date_nominal_dt, term_date_dt

def region_of_interest(which_ssw):
    if "feb2018" == which_ssw:
        lat_min, lat_max, lat_pad = 50, 65, 10
        lon_min, lon_max, lon_pad = -10, 130, 10
    elif "jan2019" == which_ssw:
        lat_min, lat_max, lat_pad = 30, 45, 10
        lon_min, lon_max, lon_pad - -95, -70, 10
    elif "sep2019" == which_ssw:
        lat_min, lat_max, lat_pad = -46, -10, 10
        lon_min, lon_max, lon_pad = 112, 154, 10
    # Of course we can add broader context to the region 
    event_region = dict(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    context_region = dict(lat=slice(lat_min-lat_pad,lat_max+lat_pad),lon=slice(lon_min-lon_pad,lon_max+lon_pad))
    return event_region,context_region

def onset_date_sensitivity_analysis(da, event_region, cgs_level, fc_dates, init_date, onset_date_nominal, term_date, ext_sign, figdir, figfile_suffix, figtitle_prefix, mem_special=None, fc_date_special=None, intensity_lims=None):
    # init_date should coincide with the first entry of da.time, but time conversion headaches...
    # Plot the minimum over the time interval as a function of onset date, across ensemble members. 
    onset_dates = [init_date+dtlib.timedelta(days=dt) for dt in range(1, (term_date-init_date).days+1)]
    severities = xr.DataArray(
            coords = dict(
                **{c: da.sel(event_region).coords[c] for c in ['lon','lat','member',]},
                onset_date=onset_dates,
                ),
            dims = ['lon','lat','member','onset_date',],
            data = np.nan,
            )
    for (i_onset_date,onset_date) in enumerate(onset_dates):
        # for some reason the slicing operation works fine 
        severities[dict(onset_date=i_onset_date)] = ext_sign*(ext_sign*da.sel(time=slice(onset_date,None))).max(dim='time')
    Nlon,Nlat,Nmem = (severities[c].size for c in ['lon','lat','member'])
    if not (Nlon==cgs_level[0] and Nlat==cgs_level[1]):
        pdb.set_trace()
    if intensity_lims is None:
        intensity_lims = [np.nanmin(da), np.nanmax(da)]
    def kwargsofmem(mem):
       if mem_special and mem==mem_special:
           kwargs = dict(color='black', linestyle='--', zorder=1)
       else:
           kwargs = dict(color='gray', alpha=0.5, linestyle='-', zorder=0)
       return kwargs
    for i_lon in range(Nlon):
        for i_lat in range(Nlat):
            fig,axes = plt.subplots(nrows=2,figsize=(6,6+3), sharex=True,height_ratios=[2,1],gridspec_kw=dict(hspace=0.3,))
            axintensity,axseverity = axes
            axnumchange = axintensity.twinx()
            isel = dict(lat=i_lat,lon=i_lon)

            lonlatstr = utils.lonlatstr(event_region,cgs_level,i_lon,i_lat)
            fig.suptitle(r"%s, %s"%(figtitle_prefix,lonlatstr), y=0.93, va='bottom')
            for i_mem,mem in enumerate(da.coords['member']):
                isel['member'] = i_mem
                xr.plot.plot(da.isel(isel), ax=axintensity, x='time',**kwargsofmem(mem))
                xr.plot.plot(severities.isel(isel), ax=axseverity, x='onset_date', **kwargsofmem(mem)) # TODO instead of plotting all the ensemble members, plot the mean
            isel.pop('member')
            xr.plot.plot(
                    (0 != 
                        severities.isel(isel)
                        .diff(dim='onset_date',label='lower')
                    ).sum(dim='member'),
                    ax=axnumchange, x='onset_date', color='red', linewidth=1
                ) 
            axintensity.set_ylim(intensity_lims)
            axseverity.set_ylim(intensity_lims)
            for ax in (axintensity,axseverity):
                ax.axvline(onset_date_nominal, color="black", linestyle="--")
                for i_fc_date,fc_date in enumerate(fc_dates):
                    ax.axvline(fc_date, color="dodgerblue", linestyle="-")
            xticks = fc_dates + [onset_date_nominal,term_date]
            for ax in axes.flat:
                xticklabels = [dtlib.datetime.strftime(date,"%m-%d") for date in xticks]
                ax.set_xticks(xticks,xticklabels,rotation=0)

            # Mark all the nominal dates
            axintensitytitle = "" if fc_date_special is None else "FC %s"%(dtlib.datetime.strftime(fc_date_special, "%Y-%m-%d"))
            axintensity.set_title(axintensitytitle)
            axintensity.set_ylabel("Daily min T2M [K]")
            axintensity.set_xlabel("Date")
            axseverity.set_title("")
            axseverity.set_ylabel("min T2M past onset")
            axseverity.set_xlabel("Onset date")
            axnumchange.set_title("")
            axnumchange.set_ylabel(r"$\#\{\frac{\Delta\text{severity}}{\Delta(\text{onset date})}\neq0\}$", rotation=90, va='bottom', color='red', labelpad=30)
            yticks = range(0,Nmem,1+Nmem//4)
            axnumchange.set_yticks(yticks, list(map(str,yticks)), color='red')
            for ax in axes:
                ax.tick_params(axis='y', which='both',labelleft=True)
                ax.tick_params(axis='x', which='both',labelbottom=True)
            fig.savefig(join(figdir,f'ODSA_{figfile_suffix}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
            plt.close(fig)
    return 

def plot_sumstats_maps_flat(da_cgt_extt, da_cgt_extt_ref, landmask, mem_special, mem_special_ref, titles, cgs_level_2show, ext_sign=1, param_bounds=None):
    # 1. ensemble-mean of time-min
    # 2. ensemble-std of time-min
    # 3. special member's time-min

    # in figure, ERA5 will be on left, and GCM will be in 2nd and 3rd columns for early and late fc_date 



    lons,lats = (da_cgt_extt[c].to_numpy() for c in ('lon','lat'))
    dlon = lons[1]-lons[0]
    dlat = lats[1]-lats[0]
    Nlon = len(lons)
    Nlat = len(lats)
    lon_extent = lons[-1]-lons[0]+dlon
    lat_extent = lats[-1]-lats[0]+dlon
    aspect = lon_extent/lat_extent

    masksea = lambda da: xr.where(landmask>0, da, np.nan)

    da_cgt_extt_ensmean_ref = masksea(da_cgt_extt_ref.mean('member'))
    da_cgt_extt_ensstd_ref = masksea(da_cgt_extt_ref.std('member'))
    da_cgt_extt_special_ref = masksea((da_cgt_extt_ref.sel(member=mem_special_ref, drop=True)-da_cgt_extt_ensmean_ref)/da_cgt_extt_ensstd_ref)

    # Compute the summary stats 
    da_cgt_extt_ensmean = masksea(da_cgt_extt.mean('member'))
    da_cgt_extt_ensstd = masksea(da_cgt_extt.std('member'))
    da_cgt_extt_special = masksea((da_cgt_extt.sel(member=mem_special, drop=True)-da_cgt_extt_ensmean_ref)/da_cgt_extt_ensstd_ref)

    if param_bounds is not None:
        bounds_loc,bounds_scale = (param_bounds[p] for p in ['loc','scale'])
    else:
        bounds_loc,bounds_scale = list(map(utils.padded_bounds, (ext_sign*da_cgt_extt_ensmean_ref,da_cgt_extt_ensstd_ref)))
    bounds_special = utils.padded_bounds(da_cgt_extt_special_ref)

    fig,axes = plt.subplots(figsize=(3*aspect,3*3), nrows=3, gridspec_kw={'hspace': 0.2}, subplot_kw={'projection': ccrs.PlateCarree()})
    axmean,axstd,axspecial = axes

    xr.plot.pcolormesh(
            da_cgt_extt_ensmean,
            cmap=plt.cm.RdYlBu_r,
            vmin = np.min(ext_sign*bounds_loc), vmax=np.max(ext_sign*bounds_loc),
            x='lon', y='lat', ax=axmean,
            cbar_kwargs={'label': '[K]'}
            )
    vmin = np.min(ext_sign*bounds_loc)
    vmax=np.max(ext_sign*bounds_loc)
    xr.plot.pcolormesh(
            da_cgt_extt_ensstd,
            x='lon', y='lat', ax=axstd, 
            cmap=plt.cm.viridis,
            vmin = 0, vmax=np.max(bounds_scale[1]),
            cbar_kwargs={'label': '[K]'}
            )
    xr.plot.pcolormesh(
            da_cgt_extt_special,
            cmap=plt.cm.RdYlBu_r,
            vmin=-np.max(np.abs(bounds_special)), vmax=np.max(np.abs(bounds_special)),
            x='lon', y='lat', ax=axspecial, 
            cbar_kwargs={'label': '[K]'}
            )
    for (i_ax,ax) in enumerate(axes):
        #ax.tick_params(axis='x',which='both',labelbottom=True)
        #ax.set_ylabel("Lat")
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='purple')
        gl = ax.gridlines(draw_labels=True, color="black")
        gl.top_labels = gl.right_labels = False
        gl.xlocator = ticker.FixedLocator(np.linspace(lons[0]-dlon/2, lons[-1]+dlon/2, cgs_level_2show[0]+1).astype(int))
        gl.ylocator = ticker.FixedLocator(np.linspace(lats[0]-dlat/2, lats[-1]+dlat/2, cgs_level_2show[1]+1).astype(int))
        ax.set_title("")
        ax.set_title(titles[i_ax], loc='left')
    return fig,axes

def plot_sumstats_map(ds,loc_vmin,loc_vmax,scale_vmin,scale_vmax):
    # Summary stats for an ensemble 
    clon,clat = (ds.coords[coordname].mean().item() for coordname in ('lon','lat'))
    fig,axes = plt.subplots(figsize=(20,8),nrows=2,subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)},gridspec_kw={'hspace': 0.2, 'wspace': 0.05})
    pcmargs = dict(x='lon',y='lat',cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree(),cbar_kwargs={'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15})
    loc_fields = (ds.mean('member'),)
    #loc_vmin,loc_vmax = (min((da.min().item() for da in loc_fields)), max((da.max().item() for da in loc_fields)))
    loc_titles = [r'Mean',]
    scale_fields = (ds.std('member'),)
    #scale_vmin,scale_vmax = (min((da.min().item() for da in scale_fields)), max((da.max().item() for da in scale_fields)))
    scale_titles = [r'Std. Dev.',]

    fields = loc_fields + scale_fields 
    vmin = [loc_vmin,scale_vmin]
    vmax = [loc_vmax,scale_vmax]
    titles = loc_titles + scale_titles 

    print(f'{vmin = }')
    print(f'{vmax = }')
    for i in range(len(fields)):
        ax = axes.flat[i]
        if fields[i] is None:
            ax.axis('off')
            continue
        pcmargs['cbar_kwargs'].update(ticks=np.linspace(vmin[i],vmax[i],3))
        xr.plot.pcolormesh(fields[i], ax=ax, vmin=vmin[i], vmax=vmax[i], **pcmargs)
        ax.set_title(titles[i])
        ax.coastlines()
        ax.gridlines()
    return fig,axes

def compute_risk(ds_cgts, ds_cgts_ref, gevpar, gevpar_ref, locsign=1):
    Nlon,Nlat = (ds_cgts[d].size for d in ('lon','lat'))
    print(f'{gevpar.dims = }')
    print(f'{gevpar_ref.dims = }')
    print(f'{gevpar.param = }')
    risk = xr.DataArray(
            coords={'lon': ds_cgts_ref.lon, 'lat': ds_cgts_ref.lat},
            dims=['lon','lat'],
            data=np.nan,
            )
    # TODO fill in the rest of this risk array by simply looping over spatial regions 
    for i_lon in range(Nlon):
        for i_lat in range(Nlat):
            thresh = np.array([locsign*ds_cgts_ref.isel(lon=i_lon,lat=i_lat).item()])
            if not np.isfinite(thresh):
                continue
            paramdict = dict({pn: np.array([gevpar.isel(lon=i_lon,lat=i_lat).sel(param=pn)]) for pn in gevpar.coords['param'].values})
            risk[dict(lon=i_lon,lat=i_lat)] = stfu.absolute_risk_parametric('gev', paramdict, thresh=thresh).item()
            # TODO correct for directionality 
    return risk

def compute_valatrisk(ds_cgts, ds_cgts_ref, gevpar, gevpar_ref, locsign=1):
    # Calculate the change in the level corresponding to an exceedance probability ccdf
    risk_valatrisk = xr.DataArray(
            coords={'lon': ds_cgts_ref.lon, 'lat': ds_cgts_ref.lat, 'quantity': ['risk','valatrisk']},
            dims=['quantity'] + [d for d in ds_cgts_ref.dims if d in ['lat','lon']],
            data=np.nan,
            )
    #pdb.set_trace()
    shapes_ref,locs_ref,scales_ref = (gevpar_ref.sel(param=p).to_numpy() for p in ('shape','loc','scale'))
    shapes,locs,scales = (gevpar.sel(param=p).to_numpy() for p in ('shape','loc','scale'))
    risk_valatrisk.loc[dict(quantity='risk')] = spgex.sf(ds_cgts_ref.to_numpy()*locsign, -shapes_ref, loc=locs_ref, scale=scales_ref)
    risk_valatrisk.loc[dict(quantity='valatrisk')] = spgex.isf(risk_valatrisk.sel(quantity='risk').to_numpy(), -shapes, loc=locs, scale=scales)*locsign
    #pdb.set_trace()
    # TODO obtain a whole range of quantile shifts to plot as a function of probability ("value at risk"? )
    return risk_valatrisk


def plot_risk_map(risk, locsign=1, **other_pcmargs):
    # the reference ds_cgts is ERA5, and should only have one year asociated with it 
    clon,clat = (risk.coords[coordname].mean().item() for coordname in ('lon','lat'))
    fig,ax = plt.subplots(subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)})
    pcmargs = dict(
            x='lon',y='lat', transform=ccrs.PlateCarree(),
            cmap=plt.cm.RdYlBu if locsign==-1 else plt.cm.RdYlBu_r,
            vmin=0.0, vmax=1.0,
            #norm=mplcolors.LogNorm(vmin=0.2,vmax=5),
            cbar_kwargs=dict({
                'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 
                'ticks': [0, 0.25, 0.5, 0.75, 1], 
                'format': ticker.FixedFormatter(['0', '0.25', '0.5', '0.75', '1'])
                })
            )
    pcmargs.update(other_pcmargs)
    xr.plot.pcolormesh(risk, **pcmargs, ax=ax)
    ax.coastlines()
    ax.gridlines()
    return fig,ax

def plot_relative_risk_map(risk0, risk1, locsign=1, **other_pcmargs):
    # the reference ds_cgts is ERA5, and should only have one year asociated with it 
    clon,clat = (risk0.coords[coordname].mean().item() for coordname in ('lon','lat'))
    fig,ax = plt.subplots(subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)}, dpi=200.0)
    pcmargs = dict(
            x='lon',y='lat', transform=ccrs.PlateCarree(),
            cmap=plt.cm.RdYlBu,
            norm=mplcolors.LogNorm(vmin=0.2,vmax=5),
            cbar_kwargs=dict({
                'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 
                'ticks': [0.2,0.5,1,2,5], 
                'format': ticker.FixedFormatter(['<0.2', '0.5', '1', '2', '>5'])
                })
            )
    pcmargs.update(other_pcmargs)
    rel_risk = risk1 / risk0
    print(f'About to pcolormesh')
    xr.plot.pcolormesh(rel_risk, **pcmargs, ax=ax)
    # now plot circles for the baseline risk 
    dlon,dlat = (c.values[1]-c.values[0] for c in (risk0.lon,risk0.lat))
    # unit circle
    theta = np.linspace(0,2*np.pi,51)[:-1]
    x_unit_circ = np.cos(theta)
    y_unit_circ = np.sin(theta)

    for (i_lon,lon) in enumerate(risk0.lon.values):
        print(f'{i_lon = } out of {risk0.lon.size}')
        for (i_lat,lat) in enumerate(risk0.lat.values):
            print(f'\t{i_lat = } out of {risk0.lat.size}')
            diam_fracs = [risk.isel(lon=i_lon,lat=i_lat).item() for risk in (risk0,risk1)]
            #ax.scatter(lon, lat, marker='o', s=100, color='black', transform=ccrs.PlateCarree(),)

            ax.plot(lon+dlon*diam_fracs[0]/2*x_unit_circ, lat+dlat*diam_fracs[0]/2*y_unit_circ, linestyle='dotted', transform=ccrs.PlateCarree(), color="black")
            ax.plot(lon+dlon*diam_fracs[1]/2*x_unit_circ, lat+dlat*diam_fracs[1]/2*y_unit_circ, linestyle='solid', transform=ccrs.PlateCarree(), color="black")

    print(f'finished the lon-lat loop')
    ax.coastlines(color='gray')
    print(f'Drew coastlines')
    ax.gridlines()
    print(f'Drew gridlines')
    return fig,ax


def coarse_grain_space(ds_cgt, cgs_level, landmask):
    data_vars = dict()
    Nlon,Nlat = (ds_cgt[d].size for d in ('lon','lat'))        
    dim = {'lon': Nlon//cgs_level[0], 'lat': Nlat//cgs_level[1]}
    trim_kwargs = dict(lon=slice(None,cgs_level[0]*dim['lon']),lat=slice(None,cgs_level[1]*dim['lat']))
    ds_cgt_trimmed = ds_cgt.isel(**trim_kwargs)
    landmask_trimmed = landmask.isel(**trim_kwargs) #* xr.ones_like(ds_cgt_trimmed)
    coslat = np.cos(np.deg2rad(ds_cgt_trimmed['lat'])) * xr.ones_like(ds_cgt_trimmed)
    # trim land mask the same way 
    coarsen_kwargs = dict(dim=dim, boundary='trim', coord_func={'lon': 'mean', 'lat': 'mean'})
    try:
        numerator = (ds_cgt_trimmed * landmask_trimmed * coslat).coarsen(**coarsen_kwargs).sum() 
    except ValueError:
        pdb.set_trace()
    denominator = (landmask_trimmed * coslat).coarsen(**coarsen_kwargs).sum() 
    land_frac = denominator / (coslat.coarsen(**coarsen_kwargs)).sum() 
    ds_cgts = numerator / denominator 
    if cgs_level[0] > 1 or cgs_level[1] > 1:
        ds_cgts = ds_cgts.where(np.isfinite(ds_cgts)*(land_frac > 0.0), np.nan)
    if not (ds_cgts.lon.size == cgs_level[0] and ds_cgts.lat.size == cgs_level[1]):
        pdb.set_trace()
    return ds_cgts # awkward to put into a single dataset because of differing lon/lat coordinates between coarsening levels


def plot_gevpar_maps_flat(gevpar, titles, cgs_level_2show, ext_sign, landmask=None, gevpar_ref=None, param_bounds=None):
    lons,lats = (gevpar[c].to_numpy() for c in ('lon','lat'))
    dlon = lons[1]-lons[0]
    dlat = lats[1]-lats[0]
    Nlon = len(lons)
    Nlat = len(lats)
    lon_extent = lons[-1]-lons[0]+dlon
    lat_extent = lats[-1]-lats[0]+dlon
    aspect = lon_extent/lat_extent

    def masksea(da):
        if landmask is None:
            return da
        return xr.where(landmask>0, da, np.nan)


    if param_bounds is not None:
        bounds_loc,bounds_scale,bounds_shape = (param_bounds[p] for p in ['loc','scale','shape'])
    elif gevpar_ref is not None:
        bounds_loc,bounds_scale,bounds_shape = list(map(utils.padded_bounds, (gevpar_ref.sel(param=p) for p in ['loc','scale','shape'])))
    else:
        bounds_loc,bounds_scale,bounds_shape = list(map(utils.padded_bounds, (gevpar.sel(param=p) for p in ['loc','scale','shape'])))

    fig,axes = plt.subplots(figsize=(3*aspect,3*3), nrows=3, gridspec_kw={'hspace': 0.2}, subplot_kw={'projection': ccrs.PlateCarree()})
    axloc,axscale,axshape = axes

    xr.plot.pcolormesh(
            masksea(ext_sign*gevpar.sel(param='loc')),
            cmap=plt.cm.RdYlBu_r,
            vmin = np.min(ext_sign*bounds_loc), vmax=np.max(ext_sign*bounds_loc),
            x='lon', y='lat', ax=axloc, 
            cbar_kwargs={'label': '[K]'}
            )
    xr.plot.pcolormesh(
            masksea(gevpar.sel(param='scale')),
            x='lon', y='lat', ax=axscale,
            cmap=plt.cm.viridis,
            vmin = 0, vmax=np.max(bounds_scale[1]), 
            cbar_kwargs={'label': '[K]'}
            )
    xr.plot.pcolormesh(
            masksea(gevpar.sel(param='shape')),
            cmap=plt.cm.RdYlBu_r,
            vmin=-np.max(np.abs(bounds_shape)), vmax=np.max(np.abs(bounds_shape)),
            x='lon', y='lat', ax=axshape,
            cbar_kwargs={'label': ''}
            )
    for (i_ax,ax) in enumerate(axes):
        #ax.tick_params(axis='x',which='both',labelbottom=True)
        #ax.set_ylabel("Lat")
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='purple')
        gl = ax.gridlines(draw_labels=True, color="black")
        gl.top_labels = gl.right_labels = False
        gl.xlocator = ticker.FixedLocator(np.linspace(lons[0]-dlon/2, lons[-1]+dlon/2, cgs_level_2show[0]+1).astype(int))
        gl.ylocator = ticker.FixedLocator(np.linspace(lats[0]-dlat/2, lats[-1]+dlat/2, cgs_level_2show[1]+1).astype(int))
        ax.set_title("")
        ax.set_title(titles[i_ax], loc='left')
    return fig,axes

def plot_statpar_map(ds_cgts,gevpar,locsign=1):
    # Essentially Gaussian parameters next to GEV parameters
    lons,lats = (ds_cgts.coords[coordname].to_numpy() for coordname in ('lon','lat'))
    dlon,dlat = np.abs(lons[1]-lons[0]),np.abs(lats[1]-lats[0])
    lonmin,lonmax = np.min(lons)-dlon/2,np.max(lons)+dlon/2
    latmin,latmax = np.min(np.abs(lats))-dlat/2,np.max(np.abs(lats))+dlat/2

    clon,clat = (np.mean(lons),np.mean(lats))

    # to get aspect ratio, find arc-length between corners
    length_x = utils.great_circle_distance(lonmin,latmin,lonmax,latmin)
    length_y = (latmax - latmin)*np.pi/180 + (1 - np.cos(np.deg2rad((lonmax-lonmin)/2)))*np.cos(np.deg2rad(latmax))
    #pdb.set_trace()
    cmap_loc = plt.cm.coolwarm
    cmap_scale = plt.cm.summer_r
    cmap_shape = plt.cm.RdYlBu_r if locsign==1 else plt.cm.RdYlBu
    pcmargs = dict(x='lon',y='lat',transform=ccrs.PlateCarree(),cbar_kwargs={'location': 'right', 'label': ''}) #, 'shrink': 0.75, 'aspect': 20, 'fraction': 0.1})
    loc_fields = (ds_cgts.mean('member'), locsign*gevpar.sel(param='loc'))
    loc_min,loc_max = (min((da.min().item() for da in loc_fields)), max((da.max().item() for da in loc_fields)))
    if locsign == 1:
        loc_vmin = 2*loc_min - loc_max
        loc_vmax = loc_max
    else:
        loc_vmin = loc_min
        loc_vmax = 2*loc_max - loc_min
    scale_fields = (ds_cgts.std('member'), gevpar.sel(param='scale'))
    scale_vmin,scale_vmax = (min((da.min().item() for da in scale_fields)), max((da.max().item() for da in scale_fields)))
    shape_fields = (None,gevpar.sel(param='shape'))
    shape_vmax = max((np.abs(da).max().item() for da in shape_fields if da is not None))
    shape_vmin = -shape_vmax

    # Gaussian figure (simple mean and std)
    fig_gaussian,axes_gaussian = plt.subplots(figsize=(10,8*2*length_y/length_x),nrows=2,subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)},gridspec_kw={'hspace': 0.2,})
    ax = axes_gaussian[0]
    pcmargs['cbar_kwargs']['ticks'] = np.linspace(loc_vmin,loc_vmax,3)
    xr.plot.pcolormesh(loc_fields[0], ax=ax, vmin=loc_vmin, vmax=loc_vmax, **pcmargs, cmap=cmap_loc)
    ax.set_title("Mean")
    ax.coastlines()
    ax.gridlines()
    ax = axes_gaussian[1]
    pcmargs['cbar_kwargs']['ticks'] = np.linspace(scale_vmin,scale_vmax,3)
    xr.plot.pcolormesh(scale_fields[0], ax=ax, vmin=scale_vmin, vmax=scale_vmax, **pcmargs, cmap=cmap_scale)
    ax.set_title("Std. Dev.")
    ax.coastlines()
    ax.gridlines()

    # GEV figure 
    fig_gev,axes_gev = plt.subplots(figsize=(10,8*3*length_y/length_x),nrows=3,subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)},gridspec_kw={'hspace': 0.2,})
    ax = axes_gev[0]
    pcmargs['cbar_kwargs']['ticks'] = np.linspace(loc_vmin,loc_vmax,3)
    xr.plot.pcolormesh(loc_fields[1], ax=ax, vmin=loc_vmin, vmax=loc_vmax, **pcmargs, cmap=cmap_loc)
    ax.set_title("GEV location")
    ax.coastlines()
    ax.gridlines()
    ax = axes_gev[1]
    pcmargs['cbar_kwargs']['ticks'] = np.linspace(scale_vmin,scale_vmax,3)
    xr.plot.pcolormesh(scale_fields[1], ax=ax, vmin=scale_vmin, vmax=scale_vmax, **pcmargs, cmap=cmap_scale)
    ax.set_title("GEV scale")
    ax.coastlines()
    ax.gridlines()
    ax = axes_gev[2]
    pcmargs['cbar_kwargs']['ticks'] = np.linspace(shape_vmin,shape_vmax,3)
    xr.plot.pcolormesh(shape_fields[1], ax=ax, vmin=shape_vmin, vmax=shape_vmax, **pcmargs, cmap=cmap_shape)
    ax.set_title("GEV shape")
    ax.coastlines()
    ax.gridlines()
    return fig_gaussian,axes_gaussian,fig_gev,axes_gev


def plot_statpar_map_difference(ds_cgts_0,ds_cgts_1,gevpar_0,gevpar_1,locsign=1):
    # NOTE this is only for mintemp where we care about NEGATIVE extremes
    def dsdiff(ds0,ds1):
        #diff_interp = ds1.interp_like(ds0,bounds_error=False) - ds0
        #print(f'{diff_interp.shape = }')
        #return diff_interp
        return ds1.assign_coords(coords=ds0.coords) - ds0
    # Essentially Gaussian parameters next to GEV parameters
    clon,clat = (ds_cgts_0.coords[coordname].mean().item() for coordname in ('lon','lat'))
    pcmargs = dict(x='lon',y='lat',cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree(),cbar_kwargs={'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 'ticks': ticker.LinearLocator(numticks=3)})
    loc_fields = (dsdiff(ds_cgts_0.mean('member'),ds_cgts_1.mean('member')), dsdiff(locsign*gevpar_0.sel(param='loc'), locsign*gevpar_1.sel(param='loc')))
    loc_vmax = tuple(np.abs(da).max().item() for da in loc_fields)
    loc_titles = [r'$\Delta$(mean)',r'$\Delta$(GEV location)']
    scale_fields = (dsdiff(ds_cgts_0.std('member'),ds_cgts_1.std('member')), dsdiff(gevpar_0.sel(param='scale'), gevpar_1.sel(param='scale')))
    scale_vmax = tuple(np.abs(da).max().item() for da in scale_fields)
    scale_titles = [r'$\Delta$(std. dev.)',r'$\Delta$(GEV scale)']
    shape_fields = (None,dsdiff(gevpar_0.sel(param='shape'),gevpar_1.sel(param='shape')))
    shape_vmax = tuple((np.abs(da).max().item() if da is not None else np.nan) for da in shape_fields)
    shape_titles = [None,r'$\Delta$(GEV shape)']

    fields = loc_fields + scale_fields + shape_fields
    vmax = loc_vmax + scale_vmax + shape_vmax
    vmin = tuple(-vm for vm in vmax)
    titles = loc_titles + scale_titles + shape_titles

    print(f'{vmin = }')
    print(f'{vmax = }')
    fig_gaussian,axes_gaussian = plt.subplots(figsize=(10,16/3),nrows=2,ncols=1,subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)},gridspec_kw={'hspace':0.2})
    fig_gev,axes_gev = plt.subplots(figsize=(10,8),nrows=3,ncols=1,subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)},gridspec_kw={'hspace':0.2})
    for (i,ax) in enumerate((axes_gaussian[0],axes_gev[0],axes_gaussian[1],axes_gev[1],None,axes_gev[2])):
        if ax is None: continue
        pcmargs['cbar_kwargs'].update(ticks=np.linspace(vmin[i],vmax[i],3))
        xr.plot.pcolormesh(fields[i], ax=ax, vmin=vmin[i], vmax=vmax[i], **pcmargs)
        ax.set_title(titles[i])
        ax.coastlines()
        ax.gridlines()
    return fig_gaussian,axes_gaussian,fig_gev,axes_gev

def fit_gev_exttemp(ds_cgts_extt,ext_sign,method='MLE'):
    # ext_sign = 1 means hot; -1 means cold
    # Take care of negative signs appropriately 
    memdim = ds_cgts_extt.dims.index('member')
    print(f'{memdim = }')
    func = lambda X: stfu.fit_gev_single(X, method=method)
    gevpar_array = np.apply_along_axis(func, memdim, ext_sign*ds_cgts_extt.to_numpy())
    gevpar_dims = list(ds_cgts_extt.dims).copy()
    gevpar_dims[memdim] = 'param'
    gevpar_coords = dict(ds_cgts_extt.coords).copy()
    gevpar_coords.pop('member')
    gevpar_coords['param'] = ['shape','loc','scale']
    gevpar = xr.DataArray(
            coords=gevpar_coords,
            dims=gevpar_dims,
            data=gevpar_array)
    return gevpar

def fit_gev_exttemp_1d_uq(exttemp, risk_levels, ext_sign, method='MLE', n_boot=1000):
    # do bootstrapping to get confidence intervals on return levels, etc. 
    gevpar_dict = stfu.fit_statistical_model(ext_sign*exttemp, 'gev', n_boot=n_boot, method=method)
    gevpar = xr.DataArray(coords={'param': ['shape','loc','scale'], 'boot': np.arange(n_boot+1)}, data=np.array([gevpar_dict[p] for p in ['shape','loc','scale']]))
    # Compute quantiles corresponding to risk levels 
    levels = ext_sign*stfu.complementary_quantile_parametric('gev', gevpar_dict, risk_levels)
    # levels should get progressively less eextreme as risk_levels increases, because less-extreme levels have a higher risk of being exceeded

    return gevpar, levels



