import numpy as np
import datetime
import xarray as xr
import dask
from os.path import join,exists
import glob
import matplotlib.pyplot as plt
pltsvargs = {"bbox_inches": "tight", "pad_inches": 0.2}

dir_prepro = "/gws/nopw/j04/snapsi/processed/wg2/area_averages"
eurasia_fc = xr.open_dataarray(join(dir_prepro,"Eurasia_area_avg_T2m.nc")) # forecasts
eurasia_ra = xr.open_dataarray(join(dir_prepro,"ERA5_regional_avg_Eurasia.nc")) # reanalysis
centers = [c.item() for c in eurasia_fc.center]

cao_begin = datetime.datetime(2018,2,21)
cao_end = datetime.datetime(2018,3,8)

ra_anom = eurasia_ra.sel(time=slice(cao_begin,cao_end)).mean().item()

num_extreme = (
        1*(eurasia_fc.min(dim="time") < ra_anom)
        .sum(dim="ens")
        )

rel_risk_wrt_free = num_extreme.sel(exp="nudged",drop=True)/num_extreme.sel(exp="free",drop=True)
rel_risk_wrt_ctrl = num_extreme.sel(exp="nudged",drop=True)/num_extreme.sel(exp="control",drop=True)

fig,ax = plt.subplots()
inits2colors = ["red","blue"]
for i_init in range(2):
    ax.scatter(rel_risk_wrt_free.isel(init=i_init),np.arange(len(centers)),color=inits2colors[i_init],marker="o")
ax.set_yticks(np.arange(len(centers)))
ax.set_yticklabels(centers)
ax.axvline(0,color="black",linestyle="--")
fig.savefig("rel_risk_wrt_free.png",**pltsvargs)
plt.close(fig)

fig,ax = plt.subplots()
inits2colors = ["red","blue"]
for i_init in range(2):
    ax.scatter(rel_risk_wrt_ctrl.isel(init=i_init),np.arange(len(centers)),color=inits2colors[i_init],marker="o")
ax.set_yticks(np.arange(len(centers)))
ax.set_yticklabels(centers)
ax.axvline(0,color="black",linestyle="--")
fig.savefig("rel_risk_wrt_ctrl.png",**pltsvargs)
plt.close(fig)
        
    





