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



def plot_summary_stats_map(ds):
    # Summary stats for an ensemble 
    fig,axes = plt.subplots(figsize=(20,8),nrows=2,subplot_kw={'projection': ccrs.Orthographic(60,58)},gridspec_kw={'hspace': 0.2, 'wspace': 0.05})
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

def coarse_grain_space(ds_cgt, cgs_level):
    data_vars = dict()
    Nlon,Nlat = (ds_cgt[dim].size for dim in ('lon','lat'))
    print(f'{ds_cgt.dims = }')
    print(f'{ds_cgt.shape = }')
    dim = {'lon': int(round(Nlon/cgs_level[0])), 'lat': int(round(Nlat/cgs_level[1]))}
    print(f'{dim = }')
    coslat = np.cos(np.deg2rad(ds_cgt['lat'])) * xr.ones_like(ds_cgt)
    coarsen_kwargs = dict(dim=dim, boundary='pad', coord_func={'lon': 'mean', 'lat': 'mean'})
    ds_cgts = (ds_cgt * coslat).coarsen(**coarsen_kwargs).sum() / coslat.coarsen(**coarsen_kwargs).sum()
    print(f'{ds_cgts.shape = }')
    return ds_cgts # awkward to put into a single dataset because of differing lon/lat coordinates between coarsening levels

def plot_ensstats_map(ds_cgts,gevpar):
    # Essentially Gaussian parameters next to GEV parameters
    fig,axes = plt.subplots(figsize=(20,8),nrows=3,ncols=2,subplot_kw={'projection': ccrs.Orthographic(60,58)},gridspec_kw={'hspace': 0.2, 'wspace': 0.05})
    pcmargs = dict(x='lon',y='lat',cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree(),cbar_kwargs={'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15})
    loc_fields = (ds_cgts.mean('member'), -gevpar.sel(param='loc'))
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

def plot_ensstats_map_difference(ds_cgts_0,ds_cgts_1,gevpar_0,gevpar_1):
    # Essentially Gaussian parameters next to GEV parameters
    fig,axes = plt.subplots(figsize=(20,8),nrows=3,ncols=2,subplot_kw={'projection': ccrs.Orthographic(60,58)},gridspec_kw={'hspace': 0.2, 'wspace': 0.05})
    pcmargs = dict(x='lon',y='lat',cmap=plt.cm.coolwarm,transform=ccrs.PlateCarree(),cbar_kwargs={'orientation': 'vertical', 'label': '', 'shrink': 0.75, 'pad': 0.04, 'aspect': 15, 'ticks': ticker.LinearLocator(numticks=3)})
    loc_fields = (ds_cgts_1.mean('member')-ds_cgts_0.mean('member'), -gevpar_1.sel(param='loc')+gevpar_0.sel(param='loc'))
    loc_vmax = tuple(np.abs(da).max().item() for da in loc_fields)
    loc_titles = [r'$\Delta$(mean)',r'$\Delta$(GEV location)']
    scale_fields = (ds_cgts_1.std('member')-ds_cgts_0.std('member'), gevpar_1.sel(param='scale')-gevpar_0.sel(param='scale'))
    scale_vmax = tuple(np.abs(da).max().item() for da in scale_fields)
    scale_titles = [r'$\Delta$(std. dev.)',r'$\Delta$(GEV scale)']
    shape_fields = (None,gevpar_1.sel(param='shape')-gevpar_0.sel(param='shape'))
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

def fit_gev_mintemp(ds_cgts_mint):
    # Take care of negative signs appropriately 
    memdim = ds_cgts_mint.dims.index('member')
    print(f'{memdim = }')
    gevpar_array = np.apply_along_axis(spgex.fit, memdim, -ds_cgts_mint.to_numpy())
    gevpar_dims = list(ds_cgts_mint.dims).copy()
    gevpar_dims[memdim] = 'param'
    gevpar_coords = dict(ds_cgts_mint.coords).copy()
    gevpar_coords.pop('year')
    gevpar_coords['param'] = ['shape','loc','scale']
    gevpar = xr.DataArray(
            coords=gevpar_coords,
            dims=gevpar_dims,
            data=gevpar_array)
    gevpar.loc[dict(param='shape')] *= -1
    return gevpar


