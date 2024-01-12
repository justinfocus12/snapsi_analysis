# Show SSW composites for the date range of interest
import yaml
import numpy as np
import datetime
import xarray as xr
import pandas as pd
from os.path import join,exists
from matplotlib import pyplot as plt, patches as mpatches
from cartopy import crs as ccrs
pltsvargs = {"bbox_inches": "tight", "pad_inches": 0.2}

dirinfo = yaml.safe_load(open("data_sources.yaml","r"))

tododict = dict({
    "compute_era5_anomaly":     1,
    "plot_era5_maps":           1,
    })


if tododict["compute_era5_anomaly"]:
    date_range = [datetime.datetime(2018,2,21),datetime.datetime(2018,3,7)]
    era5_clim = xr.open_dataarray(join(dirinfo["era5"]["directory"],dirinfo["era5"]["t2m"]["climatology"]))
    era5_anom = xr.open_dataset(join(dirinfo["era5"]["directory"],dirinfo["era5"]["t2m"]["full"]))["t2m"].sel(time=slice(date_range[0],date_range[1]+datetime.timedelta(days=1)))
    era5_anom_dayofyear = pd.to_datetime(era5_anom["time"].to_numpy()).dayofyear
    # WARNING this kills the resources
    #era5_anom = era5_anom - era5_clim.sel(dayofyear=np.minimum(era5_anom_dayofyear, 365)).to_numpy()
    for i in range(era5_anom.time.size):
        if i % 100 == 0: print(f"{i = }, {era5_anom_dayofyear[i] = }")
        era5_anom[dict(time=i)] = era5_anom.isel(time=i) - era5_clim.sel(dayofyear=min(era5_anom_dayofyear[i], 365),drop=True)
    # Single out the dates of interest
    era5_anom.to_netcdf("era5_t2m_anom_%sto%s.nc"%(date_range[0].strftime("%Y%m%d"), date_range[1].strftime("%Y%m%d")))

# Regions of interest
rois = dict({
    "eurasia": [-10,130,50,65],
    })

if tododict["plot_era5_maps"]:
    era5_anom = xr.open_dataarray("era5_anom_20180221-20180308.nc")
    fig,ax = plt.subplots(subplot_kw={"projection":  ccrs.Orthographic(central_latitude=90.0,central_longitude=0.0)})
    im = xr.plot.pcolormesh(era5_anom.sel(lat=slice(20,None)).mean(dim="time"), x="lon", y="lat", cmap="seismic", ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs={"label": "T2m anomaly [K]"})
    ax.coastlines()
    ax.gridlines()
    # Add bounding boxes for Europe
    roi = rois["eurasia"]
    ax.add_patch(mpatches.Rectangle(xy=[roi[0],roi[2]], width=roi[1]-roi[0], height=roi[3]-roi[2], facecolor="none", edgecolor="darkorange", linewidth=3, transform=ccrs.PlateCarree()))
    ax.set_title(r"ERA5 T2m anomaly, %s to %s"%(date_range[0].strftime("%Y-%m-%d"), date_range[1].strftime("%Y-%m-%d")))
    fig.savefig("t2m_anom_map.png")
    plt.close(fig)

 

