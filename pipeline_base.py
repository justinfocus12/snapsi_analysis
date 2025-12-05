import numpy as np
import xarray as xr 
import datetime as dtlib
from scipy.stats import genextreme as spgex
import pdb
from cartopy import crs as ccrs, feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import netCDF4
from os.path import join, exists
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

from importlib import reload
import utils; reload(utils)
import stat_functions as stfu; reload(stfu)

def least_sensible_onset_date(which_ssw):
    if "feb2018" == which_ssw:
        onset_date = '20180221'
    elif "jan2019" == which_ssw:
        onset_date = '20190115'
    else:
        onset_date = '20191002'
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
        onset_date_nominal = '20190115'
        term_date = '20190127' # restricted to the last day from GCMs
    elif "sep2019" == which_ssw:
        fc_dates = ['20190829','20191001']
        onset_date_nominal = '20191002'
        term_date = '20191115' # '20191014' #
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
        lat_min, lat_max, lat_pad = 40, 60, 10
        lon_min, lon_max, lon_pad = -102, -55, 10
    elif "sep2019" == which_ssw:
        lat_min, lat_max, lat_pad = -46, -10, 10
        lon_min, lon_max, lon_pad = 112, 154, 10
    # Of course we can add broader context to the region 
    event_region = dict(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    context_region = dict(lat=slice(lat_min-lat_pad,lat_max+lat_pad),lon=slice(lon_min-lon_pad,lon_max+lon_pad))
    return event_region,context_region

def analysis_multiparams(which_ssw):
    # lon/lat ratios are 6/1 at the bottom and 4/1 at the top; stick to 5/1 
    if "feb2018" == which_ssw:
        cgs_levels = [(1,1),(5,1),(10,2),(20,4),(40,8),(80,16)] #,(141,16)]
        select_regions = ( # Indexed by cgs_level
                [(0,0),], # level (1,1)
                [], # level (5,1)
                [(2*i,2*i//5) for i in range(5)],
                [], # level (20,4)
                [], # level (40,8)
                [], # level (80,16)
                )
    elif "jan2019" == which_ssw:
        cgs_levels = [(1,1),(9,6),(15,10),(30,20)]
        select_regions = ( # Indexed by cgs_level
                [(0,0),], # level (1,1)
                [], # level (2,1)
                [(i,i) for i in range(4)],
                [],
                )
    elif "sep2019" == which_ssw:
        cgs_levels = [(1,1),(4,4),(12,12),(36,36)]
        select_regions = ( # Indexed by cgs_level
                [(0,0),], # level (1,1)
                [], # level (2,1)
                [(4,4),(8,4),(4,8),(8,8)],
                [],
                )
    return cgs_levels,select_regions


def onset_date_sensitivity_analysis(ens_files_cgts, event_region, cgs_levels, fc_dates, init_date, onset_date_nominal, term_date, daily_stat, ext_sign, figdir, figtitle_prefix, figfile_tag, ens_files_cgts_ref=None, mem_special_ref=None, fc_date_special=None, idx_cgs_levels=None):
    # init_date should coincide with the first entry of da.time, but time conversion headaches...
    # Plot the minimum over the time interval as a function of onset date, across ensemble members. 
    if idx_cgs_levels is None:
        idx_cgs_levels = range(len(cgs_levels))

    for i_cgs_level in idx_cgs_levels:
        cgs_level = cgs_levels[i_cgs_level]
        cgs_key = r'%dx%d'%(cgs_level[0],cgs_level[1])
        da_cgts = xr.open_dataset(ens_files_cgts[i_cgs_level])['1xday'].sel(daily_stat=daily_stat)
        intensity_lims = utils.padded_bounds(da_cgts)
        onset_dates = [init_date+dtlib.timedelta(days=dt) for dt in range(1, (term_date-init_date).days-1)]
        severities = xr.DataArray(
                coords = dict(
                    **{c: da_cgts.sel(event_region).coords[c] for c in ['lon','lat','member',]},
                    onset_date=onset_dates,
                    ),
                dims = ['lon','lat','member','onset_date',],
                data = np.nan,
                )
        if ens_files_cgts_ref is not None:
            da_cgts_ref = xr.open_dataset(ens_files_cgts_ref[i_cgs_level])['1xday'].sel(daily_stat=daily_stat)
            severities_ref = xr.DataArray(
                    coords = dict(
                        **{c: da_cgts_ref.sel(event_region).coords[c] for c in ['lon','lat','member',]},
                        onset_date=onset_dates,
                        ),
                    dims = ['lon','lat','member','onset_date',],
                    data = np.nan,
                    )
            for (i_onset_date,onset_date) in enumerate(onset_dates):
                # for some reason the slicing operation works fine 
                severities[dict(onset_date=i_onset_date)] = ext_sign*(ext_sign*da_cgts.sel(time=slice(onset_date,None))).max(dim='time')
                severities_ref[dict(onset_date=i_onset_date)] = ext_sign*(ext_sign*da_cgts_ref.sel(time=slice(onset_date,None))).max(dim='time')
        Nlon,Nlat,Nmem = (severities[c].size for c in ['lon','lat','member'])
        if not (Nlon==cgs_level[0] and Nlat==cgs_level[1]):
            pdb.set_trace()
        def kwargsofmem(mem):
            if mem_special_ref and mem==mem_special_ref:
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
                for i_mem,mem in enumerate(da_cgts.coords['member']):
                    isel['member'] = i_mem
                    xr.plot.plot(da_cgts.isel(isel), ax=axintensity, x='time',**kwargsofmem(mem))
                    xr.plot.plot(severities.isel(isel), ax=axseverity, x='onset_date', **kwargsofmem(mem)) # TODO instead of plotting all the ensemble members, plot the mean
                isel.pop('member')
                xr.plot.plot(
                        (0 != 
                            severities.isel(isel)
                            .diff(dim='onset_date',label='lower')
                        ).sum(dim='member'),
                        ax=axnumchange, x='onset_date', color='red', linewidth=1
                    ) 
                if ens_files_cgts_ref is not None and mem_special_ref is not None:
                    xr.plot.plot(da_cgts_ref.isel(isel).sel(member=mem_special_ref), ax=axintensity, x='time',**kwargsofmem(mem_special_ref))
                    xr.plot.plot(severities_ref.isel(isel).sel(member=mem_special_ref), ax=axseverity, x='onset_date', **kwargsofmem(mem_special_ref)) # TODO instead of plotting all the ensemble members, plot the mean
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
                fig.savefig(join(figdir,f'ODSA_{figfile_tag}_cgs{cgs_key}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
                plt.close(fig)
    return 

def set_param_bounds(ens_file_cgt, param_bounds_file, landmask_file, daily_stat, onset_date, term_date, ext_sign):
    landmask = xr.open_dataarray(landmask_file)
    da_cgt = xr.open_dataset(ens_file_cgt)['1xday'].sel(daily_stat=daily_stat)
    da_cgt_extt = ext_sign * (ext_sign*da_cgt.sel(time=slice(onset_date,term_date))).max(dim='time')
    # Set global bounds on plots (sign might flip)
    param_bounds = xr.DataArray(coords={'param': ['loc','scale','shape','allvalues'], 'side': ['lo','hi']}, dims=['param','side'], data=np.nan)
    param_bounds.loc[dict(param='loc')][:] = utils.padded_bounds(ext_sign*da_cgt_extt.where(landmask>0, np.nan).mean(dim='member'))
    param_bounds.loc[dict(param='allvalues')][:] = utils.padded_bounds(da_cgt_extt.where(landmask>0, np.nan)) #.mean(dim='member'))
    param_bounds.loc[dict(param='scale')][:] = utils.padded_bounds(da_cgt_extt.where(landmask>0, np.nan).std(dim='member'), 0.1)
    param_bounds.loc[dict(param='shape')][:] = np.array([-0.5, 0.1])
    param_bounds.to_netcdf(param_bounds_file)
    return

def compute_severity_from_intensity(ens_files_cgts, ens_files_cgts_extt, cgs_levels, ext_sign, onset_date, term_date, daily_stat, landmask_file):
    # First get parameter bounds for all plots 

    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        da_cgts = xr.open_dataset(ens_files_cgts[i_cgs_level])['1xday'].sel(daily_stat=daily_stat)
        da_cgts_extt = ext_sign * (ext_sign * da_cgts.sel(time=slice(onset_date,term_date))).max(dim='time')
        da_cgts_extt.to_netcdf(ens_files_cgts_extt[i_cgs_level])
    i_cgs_level = len(cgs_levels)
    da_cgts_context = xr.open_dataset(ens_files_cgts[i_cgs_level])['1xday'].sel(daily_stat=daily_stat)
    da_cgts_context_extt = ext_sign * (ext_sign * da_cgts_context.sel(time=slice(onset_date,term_date))).max(dim='time')
    da_cgts_context_extt.to_netcdf(ens_files_cgts_extt[i_cgs_level])
    return

def plot_sumstats_maps_flat(
        event_region, context_region,
        ens_files_cgts_extt, ens_files_cgts_extt_ref, 
        mem_special, mem_special_ref, 
        ext_sign, param_bounds_file, cgs_levels, 
        ext_symb,onset_date,term_date,figdir,figfile_tag,
        title_prefix='',subplot_prefixes=None,
        ):
    # 1. ensemble-mean of time-min
    # 2. ensemble-std of time-min
    # 3. special member's time-min


    param_bounds = xr.open_dataarray(param_bounds_file)
    # TODO expand the below loop to include the context region map 
    for i_cgs_level in range(len(cgs_levels)+1):
    #for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        if i_cgs_level < len(cgs_levels) and min(cgs_levels[i_cgs_level]) <= 1:
            continue
        da_cgts_extt = xr.open_dataarray(ens_files_cgts_extt[i_cgs_level])
        da_cgts_extt_ref = xr.open_dataarray(ens_files_cgts_extt_ref[i_cgs_level])
        lons,lats = (da_cgts_extt[c].to_numpy() for c in ('lon','lat'))
        dlon = lons[1]-lons[0]
        dlat = lats[1]-lats[0]
        Nlon = len(lons)
        Nlat = len(lats)
        lonmin = lons[0]-dlon/2
        lonmax = lons[-1]+dlon/2
        latmin = lats[0]-dlat/2
        latmax = lats[-1]+dlat/2
        lon_extent = lonmax-lonmin
        lat_extent = latmax-latmin
        aspect = lon_extent/lat_extent * np.cos(np.deg2rad((lats[0]+lats[-1])/2))

        masksea = lambda da: da # we probably don't even need this #xr.where(landmask>0, da, np.nan)
        da_cgts_extt_ensmean_ref = masksea(da_cgts_extt_ref.mean('member'))
        da_cgts_extt_ensstd_ref = masksea(da_cgts_extt_ref.std('member'))
        da_cgts_extt_special_anom_ref = masksea((da_cgts_extt_ref.sel(member=mem_special_ref, drop=True)-da_cgts_extt_ensmean_ref)/da_cgts_extt_ensstd_ref)

        # Compute the summary stats 
        da_cgts_extt_ensmean = masksea(da_cgts_extt.mean('member'))
        da_cgts_extt_ensstd = masksea(da_cgts_extt.std('member'))
        da_cgts_extt_special_anom = masksea((da_cgts_extt.sel(member=mem_special, drop=True)-da_cgts_extt_ensmean_ref)/da_cgts_extt_ensstd_ref)

        bounds_special_anom = utils.padded_bounds(da_cgts_extt_special_anom_ref, 0.05)

        fig,axes = plt.subplots(figsize=(3*aspect,3*3), nrows=3, gridspec_kw={'hspace': 0.3}, subplot_kw={'projection': ccrs.Mercator(central_longitude=(lons[0]+lons[-1])/2)})
        axmean,axstd,axanomspecial = axes

        pcmargs = dict(
                x='lon',y='lat', transform=ccrs.PlateCarree(),
                add_labels=False,
                #norm=mplcolors.LogNorm(vmin=0.2,vmax=5),
                cbar_kwargs=dict({
                    'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 
                    })
                )

        pcmargs['cbar_kwargs']['label'] = '[K]'
        xr.plot.pcolormesh(
                da_cgts_extt_ensmean,
                cmap=plt.cm.RdYlBu_r,
                vmin = np.min(ext_sign*param_bounds.sel(param='loc')).item(), vmax=np.max(ext_sign*param_bounds.sel(param='loc')).item(),
                ax=axmean,
                **pcmargs,
                )
        xr.plot.pcolormesh(
                da_cgts_extt_ensstd,
                ax=axstd, 
                cmap=plt.cm.viridis,
                vmin = 0, vmax=param_bounds.sel(param='scale',side='hi').item(),
                **pcmargs,
                )
        pcmargs['cbar_kwargs']['label'] = ''
        xr.plot.pcolormesh(
                da_cgts_extt_special_anom,
                cmap=plt.cm.RdYlBu_r,
                vmin=-np.max(np.abs(bounds_special_anom)), vmax=np.max(np.abs(bounds_special_anom)),
                ax=axanomspecial, 
                **pcmargs,
                )

        # Bounding box around event region
        if i_cgs_level == len(cgs_levels):
            for ax in [axmean,axstd,axanomspecial]:
                ax.plot(
                        [event_region['lon'].start, event_region['lon'].stop, event_region['lon'].stop, event_region['lon'].start, event_region['lon'].start],
                        [event_region['lat'].start, event_region['lat'].start, event_region['lat'].stop, event_region['lat'].stop, event_region['lat'].start],
                        transform=ccrs.PlateCarree(),
                        color='black', linewidth=2, linestyle='--'
                        )

        fmtfun = lambda date: dtlib.datetime.strftime(date, "%m/%d")
        suptitle = "%s\n%s{T2M(t):\n%s\u2264t\u2264%s}"%(title_prefix, ext_symb, fmtfun(onset_date), fmtfun(term_date))
        titles = [
                "Mean",
                "Std. dev.",
                "%s norm. anom."%(mem_special),
                ]
        titles = list(map(lambda titlepre,title: titlepre+title, subplot_prefixes, titles))
        for (i_ax,ax) in enumerate(axes):
            decorate_mercator_axis(ax, lonmin, lonmax, latmin, latmax)
            ax.set_title(titles[i_ax], loc='left')
        fig.suptitle(suptitle, x=axes[0].get_position().x0, y=axes[0].get_position().y1+0.05, ha='left', va='bottom')
        cgs_suffix = r'%dx%d'%(cgs_levels[i_cgs_level][0],cgs_levels[i_cgs_level][1]) if i_cgs_level<len(cgs_levels) else 'context'
        fig.savefig(join(figdir,'sumstats_map_%s_cgs%s.png'%(figfile_tag,cgs_suffix)), **pltkwargs)
        plt.close(fig)
    return 

def decorate_mercator_axis(ax, lonmin, lonmax, latmin, latmax):
    ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='black')
    ax.coastlines(color='black')
    gl = ax.gridlines(draw_labels=True, color="black")
    gl.top_labels = gl.right_labels = False
    gl_xlocs,gl_ylocs = format_mercator_gridlines(lonmin,lonmax,latmin,latmax)
    gl.xlocator = ticker.FixedLocator(gl_xlocs)
    gl.ylocator = ticker.FixedLocator(gl_ylocs)
    gl.xformatter = LongitudeFormatter(number_format='.0f')
    gl.yformatter = LatitudeFormatter(number_format='.0f')
    return

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

def compute_risk(gevpar_files, risk_files, ens_files_cgts_extt, mem_special, ext_sign, cgs_levels):
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        if min(cgs_level) <= 1:
            continue
        gevpar = xr.open_dataarray(gevpar_files[i_cgs_level])
        da_cgts_extt = xr.open_dataarray(ens_files_cgts_extt[i_cgs_level]).sel(member=mem_special)
        risk = xr.DataArray(
                coords={'lon': da_cgts_extt.lon, 'lat': da_cgts_extt.lat},
                dims=['lon','lat'],
                data=np.nan,
                )
        # TODO fill in the rest of this risk array by simply looping over spatial regions 
        Nlon,Nlat = (da_cgts_extt.coords[c].size for c in ('lon','lat'))
        for i_lon in range(Nlon):
            for i_lat in range(Nlat):
                thresh = np.array([ext_sign*da_cgts_extt.isel(lon=i_lon,lat=i_lat).item()])
                if not np.isfinite(thresh):
                    continue
                paramdict = dict({pn: np.array([gevpar.isel(lon=i_lon,lat=i_lat).sel(param=pn)]) for pn in gevpar.coords['param'].values})
                risk[dict(lon=i_lon,lat=i_lat)] = stfu.absolute_risk_parametric('gev', paramdict, thresh=thresh).item()
                # TODO correct for directionality 
        risk.to_netcdf(risk_files[i_cgs_level])
    return 

def compute_valatrisk(ens_files_cgts_extt_ref, mem_special_ref, gevpar_ref_files, gevpar_files, valatrisk_files, cgs_levels, ext_sign):
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        gevpar = xr.open_dataarray(gevpar_files[i_cgs_level])
        gevpar_ref = xr.open_dataarray(gevpar_ref_files[i_cgs_level])
        da_cgts_extt_ref = xr.open_dataarray(ens_files_cgts_extt_ref[i_cgs_level]).sel(member=mem_special_ref)
        lons,lats = (da_cgts_extt_ref.coords[c].to_numpy() for c in ['lon','lat'])
        valatrisk = xr.DataArray(
                coords={'lon': lons, 'lat': lats, 'quantity': ['risk_refgivenref','risk_refgivenexpt','valatrisk']},
                dims=['quantity'] + [d for d in da_cgts_extt_ref.dims if d in ['lat','lon']],
                data=np.nan,
                )
        shapes_ref,locs_ref,scales_ref = (gevpar_ref.sel(param=p).to_numpy() for p in ('shape','loc','scale'))
        shapes,locs,scales = (gevpar.sel(param=p).to_numpy() for p in ('shape','loc','scale'))
        valatrisk.loc[dict(quantity='risk_refgivenref')] = spgex.sf(da_cgts_extt_ref.to_numpy()*ext_sign, -shapes_ref, loc=locs_ref, scale=scales_ref)
        valatrisk.loc[dict(quantity='risk_refgivenexpt')] = spgex.sf(da_cgts_extt_ref.to_numpy()*ext_sign, -shapes, loc=locs, scale=scales)
        valatrisk.loc[dict(quantity='valatrisk')] = spgex.isf(valatrisk.sel(quantity='risk_refgivenref').to_numpy(), -shapes, loc=locs, scale=scales)*ext_sign
        valatrisk.to_netcdf(valatrisk_files[i_cgs_level])
    return 

def plot_risk_or_valatrisk_map(
        riskandvar_files, cgs_levels, ext_sign, 
        onset_date, term_date, 
        prob_symb, ext_symb, leq_symb, ineq_symb, title_infix, event_year, 
        figdir, figfile_tag, is_risk
        ):
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        if min(cgs_level) <= 1:
            continue

        riskandvar = xr.open_dataarray(riskandvar_files[i_cgs_level])
        lons,lats = (riskandvar[c].to_numpy() for c in ('lon','lat'))
        clon,clat = np.mean(lons),np.mean(lats)
        dlon = lons[1]-lons[0]
        dlat = lats[1]-lats[0]
        Nlon = len(lons)
        Nlat = len(lats)
        lon_extent = lons[-1]-lons[0]+dlon
        lat_extent = lats[-1]-lats[0]+dlat
        minlon = 1.5*lons[0]-0.5*lons[1]
        maxlon = 1.5*lons[-1]-0.5*lons[-2]
        minlat = 1.5*lats[0]-0.5*lats[1]
        maxlat = 1.5*lats[-1]-0.5*lats[-2]
        aspect = lon_extent/lat_extent * np.cos(np.deg2rad((lats[0]+lats[-1])/2))
        # the reference ds_cgts is ERA5, and should only have one year asociated with it 
        subplot_kw = {'projection': ccrs.Mercator(central_longitude=clon,min_latitude=minlat,max_latitude=maxlat)}
        pcmargs = dict(
                x='lon',y='lat', transform=ccrs.PlateCarree(),
                add_labels=False,
                cbar_kwargs=dict({
                    'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 
                    })
                )
        if is_risk:
            pcmargs.update(dict(
                cmap=plt.cm.RdYlBu if ext_sign==-1 else plt.cm.RdYlBu_r,
                vmin=0.0, vmax=1.0,
                #norm=mplcolors.LogNorm(vmin=0.2,vmax=5),
                ))
            pcmargs['cbar_kwargs'].update(dict({
                'ticks': [0, 0.25, 0.5, 0.75, 1], 
                'format': ticker.FixedFormatter(['0', '0.25', '0.5', '0.75', '1'])
                }))
        else:
            pcmargs.update(dict(
                cmap=plt.cm.RdYlBu_r if ext_sign==-1 else plt.cm.RdYlBu,
                ))
        fig,ax = plt.subplots(figsize=(3*aspect,3), subplot_kw=subplot_kw)
        quantity2plot = 'risk_refgivenexpt' if is_risk else 'valatrisk'
        xr.plot.pcolormesh(riskandvar.sel(quantity=quantity2plot), **pcmargs, ax=ax)
        decorate_mercator_axis(ax, minlon, maxlon, minlat, maxlat)
        fmtfun = lambda dt: dtlib.datetime.strftime(dt, "%m/%d")
        if is_risk:
            ax.set_title(f'{prob_symb}{{{ext_symb}{{T2M(t): {fmtfun(onset_date)} {leq_symb} t {leq_symb} {fmtfun(term_date)}}} {ineq_symb} (ERA5 {event_year} value)}} \naccording to {title_infix}', loc='left')
        else:
            ax.set_title(f'{ext_symb}{{T2M(t): {fmtfun(onset_date)} {leq_symb} t {leq_symb} {fmtfun(term_date)}}} value at {prob_symb}={prob_symb}[ERA5]\naccording to {title_infix}', loc='left')
            #ax.set_title(f'%s[%s]{%s{T2M(t): %s %s t %s %s'%(prob_symb,title_infix,ext_symb, fmtfun(onset_date), leq_symb, leq_symb, fmtfun(term_date)))
        fig.savefig(join(figdir, "%s_map_%s_cgs%dx%d"%(quantity2plot, figfile_tag,cgs_level[0],cgs_level[1])))
        plt.close(fig)
    return 

def plot_relative_risk_map_flat(risk0, risk1, event_region, ext_sign, plot_contour_ratio=True, ):
    lons,lats = (risk0[c].to_numpy() for c in ('lon','lat'))
    dlon = lons[1]-lons[0]
    dlat = lats[1]-lats[0]
    lonmin = lons[0]-dlon/2
    lonmax = lons[-1]+dlon/2
    latmin = lats[0]-dlat/2
    latmax = lats[-1]+dlat/2
    Nlon = len(lons)
    Nlat = len(lats)
    lon_extent = lons[-1]-lons[0]+dlon
    lat_extent = lats[-1]-lats[0]+dlat
    aspect = lon_extent/lat_extent * np.cos(np.deg2rad((lats[0]+lats[-1])/2))

    # the reference ds_cgts is ERA5, and should only have one year asociated with it 
    clon,clat = (risk0.coords[coordname].mean().item() for coordname in ('lon','lat'))
    fig,ax = plt.subplots(figsize=(3*aspect,3), gridspec_kw={'hspace': 0.3}, subplot_kw={'projection': ccrs.Mercator(central_longitude=(lons[0]+lons[-1])/2)})
    pcmargs_relrisk = dict(
            x='lon',y='lat', transform=ccrs.PlateCarree(),
            add_labels=False,
            cmap=plt.cm.RdYlBu_r if ext_sign==1 else plt.cm.RdYlBu,
            norm=mplcolors.LogNorm(vmin=0.25,vmax=4),
            cbar_kwargs=dict({
                'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 
                'ticks': [0.25,0.5,1,2,4], 
                "format": ticker.FixedFormatter(["\u2264 0.25", "0.5", "1", "2", "\u2265 4"])
                })
            )
    pcmargs_absrisk = dict(
            x='lon',y='lat', transform=ccrs.PlateCarree(),
            cmap=plt.cm.RdYlBu_r if ext_sign==1 else plt.cm.RdYlBu,
            vmin=0, vmax=1,
            cbar_kwargs=dict({
                'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 
                'ticks': [0, 1/4, 1/2, 3/4, 1], 
                "format": ticker.FixedFormatter(["0", "1/4", "1/2", "3/4", "1"]),
                })
            )
    rel_risk = risk1 / risk0
    print(f'About to pcolormesh')
    xr.plot.pcolormesh(rel_risk, **pcmargs_relrisk, ax=ax)
    if plot_contour_ratio:
        # contour plot for baseline risk  
        contour_levels_absrisk = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        contour_linestyles_absrisk = list(map(lambda lev: 'dotted' if lev<0.5 else ('dashed' if lev==0.5 else 'solid'), contour_levels_absrisk))
        contour_levels_relrisk = np.array([1/2, 3/4, 1.0, 4/3, 2])
        contour_linestyles_relrisk = list(map(lambda lev: 'dotted' if lev<1.0 else ('dashed' if lev==1.0 else 'solid'), contour_levels_relrisk))
        contour_linewidths_relrisk = np.array([2, 1, 1, 1, 2])
        xr.plot.contour(rel_risk, x="lon", y="lat", ax=ax, transform=ccrs.PlateCarree(), levels=contour_levels_relrisk, linestyles=contour_linestyles_relrisk, colors='black', linewidths=contour_linewidths_relrisk, add_labels=False,)

    decorate_mercator_axis(ax, lonmin, lonmax, latmin, latmax)
    return fig,ax

def format_mercator_gridlines(lonmin,lonmax,latmin,latmax):
    aspect = (lonmax-lonmin)/(latmax-latmin) * np.cos(np.deg2rad((latmin+latmax)/2))
    if aspect > 1:
        Nlat = 2
        Nlon = int(round(aspect*2))
    else:
        Nlon = 2
        Nlat = int(round(aspect*2))
    fraclon = 1 / (2 + Nlon)
    fraclat = 1 / (2 + Nlat)
    gridlon = np.linspace((1-fraclon)*lonmin + fraclon*lonmax, fraclon*lonmin+(1-fraclon)*lonmax, Nlon)
    gridlat = np.linspace((1-fraclat)*latmin + fraclat*latmax, fraclat*latmin+(1-fraclat)*latmax, Nlat)
    return gridlon,gridlat


def plot_relative_risk_map(risk0, risk1, locsign=1, **other_pcmargs):
    # the reference ds_cgts is ERA5, and should only have one year asociated with it 
    clon,clat = (risk0.coords[coordname].mean().item() for coordname in ('lon','lat'))
    fig,ax = plt.subplots(subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)}, dpi=200.0)
    pcmargs = dict(
            x='lon',y='lat', transform=ccrs.PlateCarree(),
            cmap=plt.cm.RdYlBu,
            norm=mplcolors.LogNorm(vmin=0.25,vmax=4),
            cbar_kwargs=dict({
                'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 
                'ticks': [0.25,0.5,1,2,4], 
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

def interpolate_landmask(landmask_file_full, landmask_file_interp, lons_ref, lats_ref, event_region):
    landmask_full = (
            utils.rezero_lons(
                xr.open_dataarray(landmask_file_full)
                .isel(time=0,drop=True)
                .isel(latitude=slice(None,None,-1)) # Flip lat to go in increasing order
                .rename(dict(latitude='lat',longitude='lon'))
                )
            #.sel(event_region)
            )
    landmask = landmask_full.interp({'lat': ds_cgt.coords['lat'].values, 'lon': ds_cgt.coords['lon'].values}).sel(event_region)
    assert np.all(np.isfinite(landmask))
    landmask.to_netcdf(landmask_file_interp)
    return


def coarse_grain_space(ens_file_cgt, ens_files_cgts, cgs_levels, landmask_interp_file, event_region, context_region, Nlon, Nlat, Nlon_pad_pre, Nlon_pad_post, Nlat_pad_pre, Nlat_pad_post):
    # Add the context region as one additional cgs_level

    ds_cgt = xr.open_dataset(ens_file_cgt)
    landmask = xr.open_dataarray(landmask_interp_file)
    data_vars = dict()
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        dim = {'lon': Nlon//cgs_level[0], 'lat': Nlat//cgs_level[1]}
        trim_kwargs = dict(lon=slice(Nlon_pad_pre,Nlon_pad_pre+cgs_level[0]*dim['lon']),lat=slice(Nlat_pad_pre,Nlat_pad_pre+cgs_level[1]*dim['lat']))
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
        #pdb.set_trace()
        if cgs_level[0] > 1 or cgs_level[1] > 1:
            ds_cgts = ds_cgts.where(np.isfinite(ds_cgts)*(land_frac > 0.0), np.nan)
        if not (ds_cgts.lon.size == cgs_level[0] and ds_cgts.lat.size == cgs_level[1]):
            pdb.set_trace()
        ds_cgts.to_netcdf(ens_files_cgts[i_cgs_level])
    # Now for the context region 
    coslat = np.cos(np.deg2rad(ds_cgt['lat']) * xr.ones_like(ds_cgt))
    dim = {'lon': Nlon//cgs_level[0], 'lat': Nlat//cgs_level[1]} # same level of trimming as finest cgslevel
    coarsen_kwargs = dict(dim=dim, boundary='trim', coord_func={'lon': 'mean', 'lat': 'mean'})
    try:
        numerator = (ds_cgt * landmask * coslat).coarsen(**coarsen_kwargs).sum() 
    except ValueError:
        pdb.set_trace()
    denominator = (landmask * coslat).coarsen(**coarsen_kwargs).sum() 
    land_frac = denominator / (coslat.coarsen(**coarsen_kwargs)).sum() 
    ds_cgts_context = numerator / denominator 
    #ds_cgts_context = (ds_cgt * landmask * coslat) / (landmask * coslat).sum()
    ds_cgts_context.to_netcdf(ens_files_cgts[len(cgs_levels)])
    return # awkward to put into a single dataset because of differing lon/lat coordinates between coarsening levels


def plot_gevpar_difference_maps_flat(gevpar_files_0, expt0, gevpar_files_1, expt1, param_bounds_file, ext_sign, cgs_levels, event_region, figdir, gcm, fc_date, fc_date_abbrv, ):
    param_bounds = xr.open_dataarray(param_bounds_file)
    max_dloc,max_dscale,max_dshape = (0.5*np.abs(param_bounds.sel(param=p,side='hi')-param_bounds.sel(param=p,side='lo')).item() for p in ['loc','scale','shape'])
    print(f"{max_dloc = }")
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        if min(cgs_level) <= 1:
            continue
        gevpar0,gevpar1 = (xr.open_dataarray(gpf[i_cgs_level]) for gpf in [gevpar_files_0, gevpar_files_1])
        gevpar_diff = gevpar1 - gevpar0 

        lons,lats = (gevpar0[c].to_numpy() for c in ('lon','lat'))
        dlon = lons[1]-lons[0]
        dlat = lats[1]-lats[0]
        Nlon = len(lons)
        Nlat = len(lats)
        lonmin = lons[0]-dlon/2
        lonmax = lons[-1]+dlon/2
        latmin = lats[0]-dlat/2
        latmax = lats[-1]+dlat/2
        lon_extent = lonmax-lonmin
        lat_extent = latmax-latmin
        aspect = lon_extent/lat_extent * np.cos(np.deg2rad((lats[0]+lats[-1])/2))

        gl = utils.greekletters()

        fig,axes = plt.subplots(figsize=(3*aspect,3*3), nrows=3, gridspec_kw={'hspace': 0.3}, subplot_kw={'projection': ccrs.Mercator(central_longitude=(lons[0]+lons[-1])/2)})
        axdloc,axdscale,axdshape = axes
        fc_date_label = dtlib.datetime.strftime(fc_date, "%Y/%m/%d")

        pcmargs = dict(
                x='lon',y='lat', transform=ccrs.PlateCarree(),
                add_labels=False,
                #norm=mplcolors.LogNorm(vmin=0.2,vmax=5),
                cbar_kwargs=dict({
                    'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 
                    })
                )
        pcmargs['cbar_kwargs']['label'] = '[K]'
        fig.suptitle(f"{gcm}, FC {fc_date_label}, {expt0}\u2192{expt1}", x=0.2, ha='left')
        xr.plot.pcolormesh(
                ext_sign*gevpar_diff.sel(param='loc'),
                cmap=plt.cm.RdYlBu if ext_sign==1 else plt.cm.RdYlBu_r,
                #vmin=-max_dloc, vmax=max_dloc,
                ax=axdloc, 
                **pcmargs,
                )
        axdloc.set_title(f"Location shift {gl['Delta']}{gl['mu']}",loc='left')
        xr.plot.pcolormesh(
                gevpar_diff.sel(param='scale'),
                ax=axdscale,
                cmap=plt.cm.RdYlBu_r if ext_sign==1 else plt.cm.RdYlBu,
                vmin=-max_dscale, vmax=max_dscale,
                **pcmargs,
                )
        axdscale.set_title(f"Scale shift {gl['Delta']}{gl['sigma']}",loc='left')
        pcmargs['cbar_kwargs']['label'] = ''
        xr.plot.pcolormesh(
                gevpar_diff.sel(param='shape'),
                cmap=plt.cm.RdYlBu_r if ext_sign==1 else plt.cm.RdYlBu,
                vmin=-max_dshape, vmax=max_dshape,
                ax=axdshape,
                **pcmargs,
                )
        axdshape.set_title(f"Shape shift {gl['Delta']}{gl['xi']}",loc='left')
        for (i_ax,ax) in enumerate(axes):
            decorate_mercator_axis(ax, lonmin, lonmax, latmin, latmax)
        fig.savefig(join(figdir, f'gevpar_diff_map_e{expt0}to{expt1}_i{fc_date_abbrv}_cgs{cgs_level[0]}x{cgs_level[1]}.png'), **pltkwargs)
        plt.close(fig)
        gevpar0.close()
        gevpar1.close()
    param_bounds.close()

    return 

def plot_gevpar_maps_flat(gevpar_files, ext_sign, cgs_levels, param_bounds_file, figdir, figfile_tag, title_affix):
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        print(f"{cgs_level = }")
        if min(cgs_level) <= 1:
            continue
        gevpar = xr.open_dataarray(gevpar_files[i_cgs_level])
        lons,lats = (gevpar[c].to_numpy() for c in ('lon','lat'))
        dlon = lons[1]-lons[0]
        dlat = lats[1]-lats[0]
        Nlon = len(lons)
        Nlat = len(lats)
        lonmin,lonmax,latmin,latmax = lons[0]-dlon/2, lons[-1]+dlon/2, lats[0]-dlat/2,lats[-1]+dlat/2
        lon_extent = lonmax-lonmin
        lat_extent = latmax-latmin
        aspect = lon_extent/lat_extent * np.cos(np.deg2rad((lats[0]+lats[-1])/2))
    
        param_bounds = xr.open_dataarray(param_bounds_file)
        bounds_loc,bounds_scale,bounds_shape = (param_bounds.sel(param=p).to_numpy().flatten() for p in ['loc','scale','shape'])

        fig,axes = plt.subplots(figsize=(3*aspect,3*3), nrows=3, gridspec_kw={'hspace': 0.3}, subplot_kw={'projection': ccrs.Mercator(central_longitude=(lons[0]+lons[-1])/2)})
        axloc,axscale,axshape = axes

        pcmargs = dict(
                x='lon',y='lat', transform=ccrs.PlateCarree(),
                add_labels=False,
                #norm=mplcolors.LogNorm(vmin=0.2,vmax=5),
                cbar_kwargs=dict({
                    'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 
                    })
                )

        pcmargs['cbar_kwargs']['label'] = '[K]'
        xr.plot.pcolormesh(
                ext_sign*gevpar.sel(param='loc'),
                cmap=plt.cm.RdYlBu_r,
                vmin = np.min(ext_sign*bounds_loc), vmax=np.max(ext_sign*bounds_loc),
                ax=axloc, 
                **pcmargs,
                )
        axloc.set_title(r"Location $\mu$", loc='left')
        xr.plot.pcolormesh(
                gevpar.sel(param='scale'),
                ax=axscale,
                cmap=plt.cm.viridis,
                vmin = 0, vmax=np.max(bounds_scale[1]), 
                **pcmargs,
                )
        axscale.set_title(r"Scale $\sigma$", loc='left')
        pcmargs['cbar_kwargs']['label'] = ''
        xr.plot.pcolormesh(
                gevpar.sel(param='shape'),
                cmap=plt.cm.RdYlBu_r,
                vmin=-np.max(np.abs(bounds_shape)), vmax=np.max(np.abs(bounds_shape)),
                ax=axshape,
                **pcmargs,
                )
        axshape.set_title(r"Shape $\xi$", loc='left')
        for (i_ax,ax) in enumerate(axes):
            decorate_mercator_axis(ax, lonmin, lonmax, latmin, latmax)
        fig.suptitle(title_affix, x=0.5, y=axes[0].get_position().y1+0.05, ha='center', va='bottom')
        fig.savefig(join(figdir,"gevpar_map_%s_cgs%dx%d.png"%(figfile_tag,cgs_level[0],cgs_level[1])), **pltkwargs)
        plt.close(fig)
    return 

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


def plot_statpar_map_difference(da_cgts_0,da_cgts_1,gevpar_0,gevpar_1,locsign=1):
    # TODO turn this into a flat view (or with flexible projection) and make the GEV / Gaussian parameters diferent.
    # NOTE this is only for mintemp where we care about NEGATIVE extremes
    def dsdiff(ds0,ds1):
        return ds1 - ds0 #ds1.assign_coords(coords=ds0.coords) - ds0
    # Essentially Gaussian parameters next to GEV parameters
    clon,clat = (da_cgts_0.coords[coordname].mean().item() for coordname in ('lon','lat'))
    pcmargs = dict(x='lon',y='lat',cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree(),cbar_kwargs={'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 'ticks': ticker.LinearLocator(numticks=3)})
    loc_fields = (dsdiff(da_cgts_0.mean('member'),da_cgts_1.mean('member')), dsdiff(locsign*gevpar_0.sel(param='loc'), locsign*gevpar_1.sel(param='loc')))
    loc_vmax = tuple(np.abs(da).max().item() for da in loc_fields)
    loc_titles = [r'$\Delta$(mean)',r'$\Delta$(GEV location)']
    scale_fields = (dsdiff(da_cgts_0.std('member'),da_cgts_1.std('member')), dsdiff(gevpar_0.sel(param='scale'), gevpar_1.sel(param='scale')))
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

def fit_gev_exttemp(ens_files_cgts_extt,gevpar_files,ext_sign,cgs_levels,method='PWM'):
    # ext_sign = 1 means hot; -1 means cold
    # Take care of negative signs appropriately 
    for i_cgs_level,cgs_level in enumerate(cgs_levels):
        da_cgts_extt = xr.open_dataarray(ens_files_cgts_extt[i_cgs_level])
        memdim = da_cgts_extt.dims.index('member')
        print(f'{memdim = }')
        func = lambda X: stfu.fit_gev_single(X, method=method)
        gevpar_array = np.apply_along_axis(func, memdim, ext_sign*da_cgts_extt.to_numpy())
        gevpar_dims = list(da_cgts_extt.dims).copy()
        gevpar_dims[memdim] = 'param'
        gevpar_coords = dict(da_cgts_extt.coords).copy()
        gevpar_coords.pop('member')
        gevpar_coords['param'] = ['shape','loc','scale']
        gevpar = xr.DataArray(
                coords=gevpar_coords,
                dims=gevpar_dims,
                data=gevpar_array)
        gevpar.to_netcdf(gevpar_files[i_cgs_level])
    return 

def fit_gev_exttemp_1d_uq(exttemp, risks, ext_sign, method='PWM', n_boot=1000):
    # do bootstrapping to get confidence intervals on return levels, etc. 
    gevpar_dict = stfu.fit_statistical_model(ext_sign*exttemp, 'gev', n_boot=n_boot, method=method)
    gevpar = xr.DataArray(coords={'param': ['shape','loc','scale'], 'boot': np.arange(n_boot+1)}, data=np.array([gevpar_dict[p] for p in ['shape','loc','scale']]))
    # Compute quantiles corresponding to risk levels 
    levels = ext_sign*stfu.complementary_quantile_parametric('gev', gevpar_dict, risks)
    sevlev = xr.DataArray(coords={'boot': np.arange(n_boot+1), 'risk': risks}, dims=['boot','risk'], data=levels) # for severity levels
    #gevsevlev = xr.Dataset(data_vars={'gevpar': gevpar, 'sevlev': sevlev, 'relrisk': relrisk, 'dvalatrisk': dvalatrisk})
    # levels should get progressively less extreme as risk_levels increases, because less-extreme levels have a higher risk of being exceeded

    return gevpar, sevlev #gevsevlev


def fit_gev_select_regions(
        ens_files_cgts_extt_ref, gevsevlev_files_ref, 
        mem_special_ref, 
        ens_files_cgts_extt, gevsevlev_files, 
        risk_levels, cgs_levels, select_regions, ext_sign, 
        expt_equals_ref: bool,
        n_boot=1000
        ):
    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        da_cgts_extt = xr.open_dataarray(ens_files_cgts_extt[i_cgs_level])
        da_cgts_extt_ref = xr.open_dataarray(ens_files_cgts_extt_ref[i_cgs_level])
        Nmem_ref = da_cgts_extt_ref.coords['member'].size
        i_mem_special_ref = np.argmax([mem==mem_special_ref for mem in da_cgts_extt_ref.coords['member'].values])
        for (i_region,(i_lon,i_lat)) in enumerate(select_regions[i_cgs_level]):
            gevpar,sevlev = fit_gev_exttemp_1d_uq(
                    da_cgts_extt.isel(lon=i_lon,lat=i_lat).to_numpy().flatten(),
                    risk_levels, ext_sign, method='PWM', n_boot=n_boot
                    )
            extt_ref = da_cgts_extt_ref.isel(lon=i_lon,lat=i_lat).to_numpy().flatten()
            # Additionally, here should calculate risk and value at risk  
            risk_refgivenexpt = spgex.sf(ext_sign*extt_ref[i_mem_special_ref], -gevpar.sel(param="shape").to_numpy(), gevpar.sel(param="loc").to_numpy(), gevpar.sel(param="scale").to_numpy())
            if expt_equals_ref:
                risk_refgivenref = risk_refgivenexpt
            else:
                risk_refgivenref = xr.open_dataset(gevsevlev_files_ref[i_cgs_level][i_region])['risk_refgivenexpt']
            # ---------- VESTIGE -------------
            #rank_special_ref = np.argsort(np.argsort(extt_ref))[i_mem_special_ref]
            #prob_greater = (Nmem_ref - rank_special_ref + 0.5) / Nmem_ref
            #prob_lesser = (rank_special_ref + 0.5) / Nmem_ref 
            #risk_empirical_special_ref = prob_greater if 1==ext_sign else prob_lesser
            valatrisk_refgivenexpt = ext_sign*spgex.isf(risk_refgivenref[0], -gevpar.sel(param="shape").to_numpy(), gevpar.sel(param="loc").to_numpy(), gevpar.sel(param="scale").to_numpy())
            gev_sev_risk_var = xr.Dataset(data_vars={
                'gevpar': gevpar, 
                'sevlev': sevlev, 
                'risk_refgivenexpt': xr.DataArray(coords={'boot': np.arange(n_boot+1),},data=risk_refgivenexpt),
                'valatrisk_refgivenexpt': xr.DataArray(coords={'boot': np.arange(n_boot+1),}, data=valatrisk_refgivenexpt)
                })
            gev_sev_risk_var.to_netcdf(gevsevlev_files[i_cgs_level][i_region])
    return

def plot_gevsevlev_select_regions(
        ens_files_cgts, ens_files_cgts_extt, gevsevlev_files, 
        ens_files_cgts_ref, ens_files_cgts_extt_ref, gevsevlev_files_ref, 
        mem_special_ref, param_bounds_file, ref_label,
        cgs_levels, daily_stat,
        event_region, select_regions, 
        boot_type, confint_width, 
        figdir, figfile_tag, figtitle_affix, gcm_label,
        fc_dates, fc_date, onset_date, term_date, 
        prob_symb, ext_sign, ext_symb, leq_symb, ineq_symb,
        ref_is_different=False, # whether the reference ensemble is different from the main one 
        ):
    # TODO add lines to indicate the reference values etc 
    param_bounds = xr.open_dataarray(param_bounds_file)

    for (i_cgs_level,cgs_level) in enumerate(cgs_levels):
        da_cgts = xr.open_dataset(ens_files_cgts[i_cgs_level])['1xday'].sel(daily_stat=daily_stat)
        da_cgts_ref = xr.open_dataset(ens_files_cgts_ref[i_cgs_level])['1xday'].sel(daily_stat=daily_stat)
        da_cgts_extt = xr.open_dataarray(ens_files_cgts_extt[i_cgs_level])
        da_cgts_extt_ref = xr.open_dataarray(ens_files_cgts_extt_ref[i_cgs_level])
        lon_blocksize,lat_blocksize = ((event_region[d].stop - event_region[d].start)/cgs_level[i_d] for (i_d,d) in enumerate(('lon','lat')))
        idx_mem_special_ref = np.argmax([mem_special_ref == mem for mem in da_cgts_extt_ref['member'].values])
        for (i_region,(i_lon,i_lat)) in enumerate(select_regions[i_cgs_level]):
            gevsevlev = xr.open_dataset(gevsevlev_files[i_cgs_level][i_region])
            gevsevlev_ref = xr.open_dataset(gevsevlev_files_ref[i_cgs_level][i_region])
            sevlev = gevsevlev['sevlev'].to_numpy() # n_boot x n_lev
            sevlev_ref = gevsevlev_ref['sevlev'].to_numpy() # n_boot x n_lev
            exttemp = da_cgts_extt.isel(lon=i_lon,lat=i_lat).to_numpy().flatten()
            exttemp_ref = da_cgts_extt_ref.isel(lon=i_lon,lat=i_lat).to_numpy().flatten()
            temp_bounds = utils.padded_bounds(da_cgts_ref.isel(lon=i_lon,lat=i_lat), inflation=0.2)
            exttemp_ref_special = exttemp_ref[idx_mem_special_ref]
            risk_levels = gevsevlev.coords['risk'].to_numpy() # increasing 
            risk_levels_ref = gevsevlev_ref.coords['risk'].to_numpy() # increasing 
            gevpar = gevsevlev['gevpar']
            gevpar_ref = gevsevlev_ref['gevpar']
            risk_refgivenexpt = gevsevlev['risk_refgivenexpt']
            valatrisk_refgivenexpt = gevsevlev['valatrisk_refgivenexpt']
            risk_refgivenref = gevsevlev_ref['risk_refgivenexpt']
            valatrisk_refgivenref = gevsevlev_ref['valatrisk_refgivenexpt']

            center_lon = event_region['lon'].start + (i_lon+0.5)*lon_blocksize
            center_lat = event_region['lat'].start + (i_lat+0.5)*lat_blocksize
            lonlatstr = r'$\lambda=%d\pm%d,\phi=%d\pm%d$'%(center_lon,lon_blocksize/2,center_lat,lat_blocksize/2)
            if max(cgs_level) == 1:
                lonlatstr = r'%s (whole region)'%(lonlatstr)

            order = np.argsort(exttemp)
            rank = np.argsort(order)
            order_ref = np.argsort(exttemp_ref)
            rank_ref = np.argsort(order_ref)
            if ext_sign == -1:
                risk_empirical = np.arange(1,len(exttemp)+1)/len(exttemp)
                risk_empirical_ref = np.arange(1,len(exttemp_ref)+1)/len(exttemp_ref)
            else:
                risk_empirical = np.arange(len(exttemp),0,-1)/len(exttemp)
                risk_empirical_ref = np.arange(len(exttemp_ref),0,-1)/len(exttemp_ref)
            shape,loc,scale = (gevpar.sel(param=p).isel(boot=0) for p in ('shape','loc','scale'))
            shape_ref,loc_ref,scale_ref = (gevpar_ref.sel(param=p).isel(boot=0) for p in ('shape','loc','scale'))
            if not np.all([np.isfinite(p) for p in [shape,loc,scale]]):
                pdb.set_trace()
            # --------------- Figure: left GEV, right timeseries ---------
            fig = plt.figure(figsize=(12,8))
            gs = gridspec.GridSpec(figure=fig, nrows=2, ncols=2, height_ratios=[1,18], hspace=0.0)
            ax_title = fig.add_subplot(gs[0,0:2])
            ax_gev = fig.add_subplot(gs[1,0])
            ax_timeseries = fig.add_subplot(gs[1,1], sharey=ax_gev)

            # GEV & sevlev
            grk = utils.greekletters()
            def param_label_fun(loc,scale,shape):
                param_label = '\n'.join([
                    '%s=%.1f'%(grk["mu"],ext_sign*loc),
                    '%s=%.1f'%(grk["sigma"],scale),
                    '%s=%+.2f'%(grk["xi"],shape)
                    ])
                return param_label
            ax = ax_gev
            xlim = [min(np.min(risk_empirical),np.min(risk_empirical_ref)), 1.01]
            ax.set_xlim(xlim)
            handles = []
            # non-ref
            ax.scatter(risk_empirical, exttemp[order], color='red', marker='+')
            h, = ax.plot(risk_levels,sevlev[0,:],color='red', label=gcm_label+'\n'+param_label_fun(loc,scale,shape))
            handles.append(h)
            boot_quant_lo,boot_quant_hi = (np.quantile(sevlev[1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
            if boot_type == 'percentile':
                lo,hi = boot_quant_lo,boot_quant_hi
            else:
                lo,hi = 2*sevlev[0,:]-boot_quant_hi, 2*sevlev[0,:]-boot_quant_lo
            ax.fill_between(risk_levels, lo, hi, fc='red', ec='none', alpha=0.3, zorder=-1)
            # Plot the ref risk value to make sure it's consistent
            ax.plot(
                    [risk_refgivenexpt.isel(boot=0).item()] + [np.quantile(risk_refgivenexpt.isel(boot=slice(1,None)), 0.5*(1+sgn*confint_width)).item() for sgn in [-1,1]], 
                    exttemp_ref_special*np.ones(3), 
                    color='purple', linewidth=3, zorder=2
                    )
            ax.plot(
                    #risk_empirical_ref[rank_ref[idx_mem_special_ref]]*np.ones(3), 
                    risk_refgivenref.isel(boot=0).item()*np.ones(3),
                    [valatrisk_refgivenexpt.isel(boot=0).item()] + [np.quantile(valatrisk_refgivenexpt[1:], 0.5*(1+sgn*confint_width)).item() for sgn in [-1,1]],
                    color='purple', linewidth=3, zorder=2
                    )
            # Dropping lines to axes to illustrate absolute risk and value-at-risk 
            # 1. Horizontal line from era5 to gcm curve 
            ax.plot([risk_empirical_ref[rank_ref[idx_mem_special_ref]], risk_refgivenexpt.isel(boot=0).item()], exttemp_ref_special*np.ones(2), color='black', linestyle='--', linewidth=1.5)
            # 2. vertical line from gcm curve to risk axis, with error bars
            ax.plot(risk_refgivenexpt.isel(boot=0).item()*np.ones(2), [temp_bounds[0], exttemp_ref_special], color='red', linestyle='--', linewidth=1.5)
            ax.fill_betweenx([temp_bounds[0], exttemp_ref_special], *[np.quantile(risk_refgivenexpt.isel(boot=slice(1,None)), 0.5*(1+sgn*confint_width)).item() for sgn in [-1,1]], fc='red', ec='none', zorder=-1, alpha=0.3)
            # 3. Vertical line from era5 curve (at ERA5 risk) to gcm curve
            ax.plot(risk_refgivenref.isel(boot=0).item()*np.ones(2), [exttemp_ref_special, valatrisk_refgivenexpt.isel(boot=0).item()], color='black', linestyle='--', linewidth=1.5) 
            # 4. Horizontal line from gcm curve (at valatrisk) to severity axis
            ax.plot([risk_refgivenref.isel(boot=0).item(), xlim[1]], valatrisk_refgivenexpt.isel(boot=0).item()*np.ones(2), color='red', linestyle='--', linewidth=1.5)
            ax.fill_between([risk_refgivenref.isel(boot=0).item(), xlim[1]], *[np.quantile(valatrisk_refgivenexpt.isel(boot=slice(1,None)), 0.5*(1+sgn*confint_width)).item() for sgn in [-1,1]], fc='red', ec='none', zorder=-1, alpha=0.3)


            # ref
            ax.scatter(risk_empirical_ref, exttemp_ref[order_ref], color='black', marker='+')
            h, = ax.plot(risk_levels_ref,sevlev_ref[0,:],color='black', label=ref_label+'\n'+param_label_fun(loc_ref,scale_ref,shape_ref))
            handles.append(h)
            boot_quant_lo,boot_quant_hi = (np.quantile(sevlev_ref[1:], 0.5*(1+sgn*confint_width), axis=0) for sgn in (-1,1))
            if boot_type == 'percentile':
                lo,hi = boot_quant_lo,boot_quant_hi
            else:
                lo,hi = 2*sevlev_ref[0,:]-boot_quant_hi, 2*sevlev_ref[0,:]-boot_quant_lo
            ax.fill_between(risk_levels_ref, lo, hi, fc='gray', ec='none', alpha=0.3, zorder=-1)

            # Special marker for the event year itself 
            ax.scatter(risk_empirical_ref[rank_ref[idx_mem_special_ref]], exttemp_ref_special, color='black', marker='o', s=18)

            # Decorations 
            ax.set_xscale('log')

            fmtfun = lambda dt: dtlib.datetime.strftime(dt, "%m/%d")
            ax.set_xlabel(f'{prob_symb}{{{ext_symb}{{T2M(t): {fmtfun(onset_date)} {leq_symb} t {leq_symb} {fmtfun(term_date)}}} {ineq_symb} T}}')
            ax.set_ylabel(r'T [K]')
            ax.set_ylim(temp_bounds)
            ax.set_title('')
            ax.legend(handles=handles)

            # Timeseries
            argmaxwindow = lambda da: da.sel(time=slice(onset_date,term_date))
            ax = ax_timeseries
            ts_argmax = []
            Ts_argmax = []
            for i_mem in range(da_cgts.member.size):
                temp_gcm = da_cgts.isel(lat=i_lat,lon=i_lon,member=i_mem)
                #print(f'{temp_gcm.time = }')
                xr.plot.plot(temp_gcm, x='time', ax=ax, color='red', alpha=0.25)
                i_t_argmax = argmaxwindow(ext_sign*temp_gcm).argmax(dim='time').item()
                t_argmax = onset_date + dtlib.timedelta(i_t_argmax)
                T_argmax = argmaxwindow(temp_gcm).isel(time=i_t_argmax).item()
                ts_argmax.append(t_argmax)
                Ts_argmax.append(T_argmax)
            ax.scatter(ts_argmax, Ts_argmax,color='red', marker='+')
            # collect all the min points 
            ts_argmax = []
            Ts_argmax = []
            for (i_mem,mem) in enumerate(da_cgts_ref.member.to_numpy()):
                temp_ref = da_cgts_ref.isel(lat=i_lat,lon=i_lon,member=i_mem)
                xr.plot.plot(temp_ref, x='time', ax=ax, color='gray', alpha=0.25) 
                if mem == mem_special_ref:
                    xr.plot.plot(temp_ref, x='time', ax=ax, color='black', linestyle='--', linewidth=2) 
                i_t_argmax = argmaxwindow(ext_sign*temp_ref).argmax(dim='time').item()
                t_argmax = onset_date + dtlib.timedelta(i_t_argmax)
                T_argmax = argmaxwindow(temp_ref).isel(time=i_t_argmax).item()
                ts_argmax.append(t_argmax)
                Ts_argmax.append(T_argmax)
            ax.scatter(ts_argmax, Ts_argmax,color='black', marker='+')
            ax.axvline(onset_date, color='dodgerblue', zorder=-1, alpha=0.5)
            ax.axvline(term_date, color='dodgerblue', zorder=-1, alpha=0.5)
            ax.set_title('')
            ax.axvline(fc_date, color='red', linestyle='--', zorder=-1)
            ax.set_xlabel('')
            xtickvalues = [fc_date] + [onset_date, term_date]
            fmtfun = lambda date: dtlib.datetime.strftime(date, "%Y/%m/%d")
            xticklabels = list(map(fmtfun, xtickvalues))
            ax.set_xticks(xtickvalues, xticklabels, rotation=37.5, ha='right')
            ax.yaxis.set_tick_params(which='both', labelbottom=True)
            ax.set_ylabel('')

            shape,loc,scale = (gevpar.sel(param=p).isel(boot=0) for p in ('shape','loc','scale'))
            shape_ref,loc_ref,scale_ref = (gevpar_ref.sel(param=p).isel(boot=0) for p in ('shape','loc','scale'))
            # Title
            ax_title.text(0.5, 0.0, figtitle_affix, transform=ax_title.transAxes, ha='center', va='bottom', fontsize=20)
            ax_title.axis('off')

            fig.savefig(join(figdir,f'gevsevlev_{figfile_tag}_cgs{cgs_level[0]}x{cgs_level[1]}_ilon{i_lon}_ilat{i_lat}.png'), **pltkwargs)
            plt.close(fig)
            

            





