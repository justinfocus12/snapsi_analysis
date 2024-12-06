import numpy as np
import xarray as xr 
from scipy.stats import genextreme as spgex
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

from importlib import reload
import stat_functions as stfu; reload(stfu)



def plot_sumstats_map(ds):
    # Summary stats for an ensemble 
    clon,clat = (ds.coords[coordname].mean().item() for coordname in ('lon','lat'))
    fig,axes = plt.subplots(figsize=(20,8),nrows=2,subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)},gridspec_kw={'hspace': 0.2, 'wspace': 0.05})
    pcmargs = dict(x='lon',y='lat',cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree(),cbar_kwargs={'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15})
    loc_fields = (ds.mean('member'),)
    loc_vmin,loc_vmax = (min((da.min().item() for da in loc_fields)), max((da.max().item() for da in loc_fields)))
    loc_titles = [r'Mean',]
    scale_fields = (ds.std('member'),)
    scale_vmin,scale_vmax = (min((da.min().item() for da in scale_fields)), max((da.max().item() for da in scale_fields)))
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

            ax.plot(lon+dlon*diam_fracs[0]/2*x_unit_circ, lat+dlat*diam_fracs[0]/2*y_unit_circ, linestyle='dotted', transform=ccrs.PlateCarree())
            ax.plot(lon+dlon*diam_fracs[1]/2*x_unit_circ, lat+dlat*diam_fracs[1]/2*y_unit_circ, linestyle='solid', transform=ccrs.PlateCarree())

    print(f'finished the lon-lat loop')
    ax.coastlines(color='gray')
    print(f'Drew coastlines')
    ax.gridlines()
    print(f'Drew gridlines')
    return fig,ax


def coarse_grain_space(ds_cgt, cgs_level, landmask):
    data_vars = dict()
    Nlon,Nlat = (ds_cgt[d].size for d in ('lon','lat'))
    dim = {'lon': int(Nlon/cgs_level[0]), 'lat': int(Nlat/cgs_level[1])}
    print(f'{dim = }')
    trim_kwargs = dict(lon=slice(None,cgs_level[0]*dim['lon']),lat=slice(None,cgs_level[1]*dim['lat']))
    ds_cgt_trimmed = ds_cgt.isel(**trim_kwargs)
    landmask_trimmed = landmask.isel(**trim_kwargs) #* xr.ones_like(ds_cgt_trimmed)
    print(f'{landmask_trimmed.coords = }')
    coslat = np.cos(np.deg2rad(ds_cgt_trimmed['lat'])) * xr.ones_like(ds_cgt_trimmed)
    # trim land mask the same way 
    coarsen_kwargs = dict(dim=dim, boundary='trim', coord_func={'lon': 'mean', 'lat': 'mean'})
    numerator = (ds_cgt_trimmed * landmask_trimmed * coslat).coarsen(**coarsen_kwargs).sum() 
    denominator = (landmask_trimmed * coslat).coarsen(**coarsen_kwargs).sum() 
    land_frac = denominator / (coslat.coarsen(**coarsen_kwargs)).sum() 
    ds_cgts = numerator / denominator 
    if cgs_level[0] > 1 or cgs_level[1] > 1:
        ds_cgts = ds_cgts.where(np.isfinite(ds_cgts)*(land_frac >= 0.5), np.nan)
    print(f'{ds_cgts.shape = }')
    assert ds_cgts.lon.size == cgs_level[0] and ds_cgts.lat.size == cgs_level[1]
    return ds_cgts # awkward to put into a single dataset because of differing lon/lat coordinates between coarsening levels

def plot_statpar_map(ds_cgts,gevpar,locsign=1):
    # Essentially Gaussian parameters next to GEV parameters
    clon,clat = (ds_cgts.coords[coordname].mean().item() for coordname in ('lon','lat'))
    fig,axes = plt.subplots(figsize=(20,8),nrows=3,ncols=2,subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)},gridspec_kw={'hspace': 0.2, 'wspace': 0.05})
    pcmargs = dict(x='lon',y='lat',cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree(),cbar_kwargs={'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15})
    loc_fields = (ds_cgts.mean('member'), locsign*gevpar.sel(param='loc'))
    loc_vmin,loc_vmax = (min((da.min().item() for da in loc_fields)), max((da.max().item() for da in loc_fields)))
    loc_titles = [r'Mean',r'GEV location']
    scale_fields = (ds_cgts.std('member'), gevpar.sel(param='scale'))
    scale_vmin,scale_vmax = (min((da.min().item() for da in scale_fields)), max((da.max().item() for da in scale_fields)))
    scale_titles = [r'Std. Dev.',r'GEV scale']
    shape_fields = (None,gevpar.sel(param='shape'))
    shape_vmax = max((np.abs(da).max().item() for da in shape_fields if da is not None))
    shape_vmin = -shape_vmax
    shape_titles = [None,r'GEV shape']

    fields = loc_fields + scale_fields + shape_fields
    vmin = [loc_vmin]*2 + [scale_vmin]*2 + [shape_vmin]*2
    vmax = [loc_vmax]*2 + [scale_vmax]*2 + [shape_vmax]*2
    titles = loc_titles + scale_titles + shape_titles

    print(f'{vmin = }')
    print(f'{vmax = }')
    for i in range(6):
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


def plot_statpar_map_difference(ds_cgts_0,ds_cgts_1,gevpar_0,gevpar_1,locsign=1):
    # NOTE this is only for mintemp where we care about NEGATIVE extremes
    def dsdiff(ds0,ds1):
        #diff_interp = ds1.interp_like(ds0,bounds_error=False) - ds0
        #print(f'{diff_interp.shape = }')
        #return diff_interp
        return ds1.assign_coords(coords=ds0.coords) - ds0
    # Essentially Gaussian parameters next to GEV parameters
    clon,clat = (ds_cgts_0.coords[coordname].mean().item() for coordname in ('lon','lat'))
    fig,axes = plt.subplots(figsize=(20,8),nrows=3,ncols=2,subplot_kw={'projection': ccrs.Orthographic(central_longitude=clon,central_latitude=clat)},gridspec_kw={'hspace': 0.2, 'wspace': 0.05})
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
    for i in range(6):
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



