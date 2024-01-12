import numpy as np
import datetime
import xarray as xr
from os.path import join,exists
import matplotlib.pyplot as plt
pltsvargs = {"bbox_inches": "tight", "pad_inches": 0.2}

# Folders of interest
base_dir = "/gws/nopw/j04/snapsi/processed/"

all_models = ["BCC-CSM2-HR","GLOBO","GEM-NEMO","CANESM5","IFS","SPEAR","GRIMs","GloSea6-GC32","CNRM-CM61","CESM2-CAM6","NAVGEM","GloSea6"]
models = [all_models[i] for i in [9,11]]

expts = ["free","nudged","control","nudged-full","control-full"]
dates = [datetime.datetime(2018,1,25),datetime.datetime(2018,2,8)]
date_abbrvs = ["s20180125","s20180208"]
date_labels = [d.strftime('%Y-%m-%d') for d in dates]



colors = ["red","dodgerblue"]
vbl = "u1060"

# Plot the ensemble of u1060 over time
fig,axes = plt.subplots(nrows=len(expts),ncols=len(models), figsize=(5*len(models),4*len(expts)), sharex=True, sharey=True)
for i_model,model in enumerate(models):
    for i_expt,expt in enumerate(expts):
        print(f"{model = }, {expt = }")
        ax = axes[i_expt,i_model]
        handles = []
        for i_date,date_abbrv in enumerate(date_abbrvs):
            date_label = date_labels[i_date]
            filename = join(base_dir,model,expt,date_abbrv,vbl,f"{model}_{expt}_{date_abbrv}_{vbl}.nc")
            if exists(filename):
                u = xr.open_dataset(filename)["u"]
                print(f"{u.coords = }")
                u = u.assign_coords(time=np.arange(u.time.size)+(dates[i_date]-dates[0]).days)
                for mem in u["member_id"]:
                    h, = xr.plot.plot(u.sel(member_id=mem), x="time", ax=ax, color=colors[i_date], label=date_abbrv)
                handles.append(h)
                if vbl == "u1060":
                    ax.axhline(0,color="black",linestyle="--")

        # TODO add ERA5 to the plot
        ax.set_xlabel(f"Time [days since {date_label}]")
        ax.set_ylabel(vbl)
        ax.set_title(f"{model}, {expt}")
        ax.legend(handles=handles)
fig.savefig(f"vis_{vbl}", **pltsvargs)
plt.close(fig)





