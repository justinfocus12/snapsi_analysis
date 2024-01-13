import numpy as np
from scipy.stats import norm as spnorm
import pandas as pd
import datetime
import pickle
import xarray as xr
import dask
from os.path import join,exists,basename
import glob
import yaml
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'monospace',
    'font.size': 20,
    })
pltsvargs = {"bbox_inches": "tight", "pad_inches": 0.2}
import data_retrieval as datre
import utils

def reduce_clim(qoidict, stagefiles):
    # Compute daily and areally-averaged temperatures for climatology and ERA5
    avgD = (
            datre.rezero_lons(xr.open_dataarray(datre.get_clim_filename(qoidict['var_name'])))
            .sel(qoidict['spacesel'])
            .sel(qoidict['timesel_maxposs']))
    avgD.to_netcdf(stagefiles['avgD'])
    avgDA = datre.area_average(avgD)
    avgDA.to_netcdf(stagefiles['avgDA'])
    return

def reduce_era5(qoidict, stagefiles):
    # Compute daily and areally-averaged temperatures for climatology and ERA5
    avgD = (
            datre.rezero_lons(
                xr.open_dataset(datre.get_era5_filename(qoidict['var_name']))[qoidict['var_name']]
                .sel(time=slice(period_begin,period_end)))
            .sel(qoidict['spacesel'])
            .sel(qoidict['timesel_maxposs']))
    avgD.to_netcdf(stagefiles['avgD'])
    avgDA = datre.area_average(avgD)
    avgDA.to_netcdf(stagefiles['avgDA'])
    return

def reduce_gcm(qoidict, stagefiles, tododict):
    if tododict['avgD']:
        ds = []
        for i_fcdate,fcdate in enumerate(qoidict['fcdates']):
            preprocess = lambda dsmem: datre.preprocess_gcm_6hrPt(dsmem,qoidict['var_name'],fcdate,qoidict['timesel_maxposs'],qoidict['spacesel'])
            mem_filenames = dict()
            mem_labels = dict()
            ens_sizes = []
            for expt in qoidict['expts']:
                print(f"{expt = }")
                mem_filenames[expt],mem_labels[expt] = datre.get_gcm_6hrPt_filenames(qoidict['model'], qoidict['var_name'], expt, fcdate)
                print(f"{len(mem_filenames[expt]) = }")
                ens_sizes.append(len(mem_filenames[expt]))
            # Before loading data, verify there are the same number of files in each
            if ens_sizes[0] > 0 and len(np.unique(ens_sizes)) == 1:
                ds_fc = []
                for expt in qoidit['expts']:
                    ds_fc.append(xr.open_mfdataset(mem_filenames[expt], preprocess=preprocess, combine='nested', concat_dim='member').assign_coords(member=mem_labels[expt]))
                ds_fc = xr.concat(ds_fc, dim="expt").assign_coords(expt=qoidict['expts'])
                print(f"{ds_fc[0]['time'].to_numpy() = }")
                print(f"Expts concatenated successfully")
                #ds_fc = ds_fc.compute()
                ds.append(ds_fc)
        if len(ds) == 2:
            ds = xr.concat(ds, dim='init').assign_coords(init=qoidict['fcdates'])
            ds.to_netcdf(stagefiles['avgD'])
    if tododict['avgDA']:
        # Don't anomalize, because absolute minimum temperature might be more relevant than minimum anomaly. Maybe we should subtract the min over the time period from the min over the time period from ERA5
        ds_avgD = xr.open_dataset(stagefiles['avgD'])
        ds_avgDA = datre.area_average(ds_avgD)
        ds_avgDA.to_netcdf(stagefiles['avgDA'])
    return

def plot_avgDA_timeseries(qoidict, stagefiles):
    # Only for 1 model
    avgDA_era5 = xr.open_dataarray(stagefiles['era5']['avgDA'])
    avgDA_clim = xr.open_dataarray(stagefiles['clim']['avgDA'])
    avgDA_gcm = xr.open_dataarray(stagefiles['gcm']['avgDA'])
    fig,axes = plt.subplots(nrows=2,ncols=len(qoidict['expts']),figsize=(6*len(qoidict['expts']),6),sharex=True,sharey=True)
    for i_expt,expt in enumerate(qoidict['expts']):
        for i_fcdate,fcdate in enumerate(anom_aa_gcm_justin.fcdate.values):
            ax = axes[i_fcdate,i_expt]
            for member in anom_aa_gcm_justin.member.values:
                hgcm, = xr.plot.plot(avgDA_gcm.sel(expt=expt,fcdate=fcdate,member=member),x="time",color="red", ax=ax, label=model, alpha=0.3)
            hera, = xr.plot.plot(avgDA_era5,x="time",color="black",linewidth=3,linestyle="--",ax=ax,label="ERA5")
            hclim, = xr.plot.plot(avgDA_clim,x="time",color="gray",linewidth=4,linestyle="-",alpha=0.5,ax=ax,label="Climatology")
            hgcm_mean, = xr.plot.plot(anom_aa_da_gcm_justin.sel(expt=expt,fcdate=fcdate).mean("member"),x="time",color="red",linewidth=3,linestyle="--", ax=ax, label="Ens. mean")

            ax.set_title(r"Init %s, %s"%(fcdates[i_fcdate].strftime("%Y-%m-%d"),expt))
            ax.set_ylabel(f"{qoidict['var_name']} [K]")
            ax.set_xlabel("")
            ax.legend(handles=[hclim,hera,hgcm,hgcm_mean])
    fig.savefig(stagefiles['avgDA_plot'],**pltsvargs)
    plt.close(fig)
    

def risk_calc_pipeline_1model(qoidict, tododict, stagefiles):
    # qoidict: quantities of interest
    # tododict: to-do items
    # avgD: daily averaged
    # avgDA: daily and area-averaged
    if tododict['avgD'] or tododict['avgDA']:
        reduce_gcm(qoidict, stagefiles, tododict)
    if tododict['plot_avgDA_timeseries']:
        ds_avgDA = xr.open_dataset(stagefiles['avgDA']) # Do we need to further subset by the variable? Probably not because didn't subtract off ERA5
    if tododict['compute_severity']:
        ds_avgDA = xr.open_dataset(stagefiles['avgDA']) # Do we need to further subset by the variable? Probably not because didn't subtract off ERA5
        severity = severity_fun_avgDA(ds_avgDA.sel(qoidict['timesel_event']), qoidict) # dims: (fcdate,expt,member)
        severity.to_netcdf(stagefiles['severity'])
    if tododict['compute_risk']:
        severity = xr.open_dataset(stagefiles['severity'])
        print(f'{severity.dims = }, {severity.shape = }')

    return
        
        



# -------------------- Severity functions -------------------
def severity_fun_avgDA(avgDA, qoidict):
    # measures of severity for daily- and areally-averaged timeseries data
    if qoidict['metric'] == 'mintemp': 
        severity = -avgDA.min(dim='time')
    elif qoidict['metric'] == 'meantemp':
        severity = -avgDA.mean(dim='time')
    elif qoidict['metric'] == 'durationcold':
        # Calculate the maximum duration with temps < some threshold
        pass
    return severity

# -----------------------------------------------------------

tododict = dict({
    "compute_clim":                  0,
    "compute_era5":                  0,
    "compute_model":                 0,
    "compute_area_averages":         0,
    "compare_to_will":               1,
    "compute_risks":                 1,
    "plot_risks":                    1,
    "plot_relative_risk":            1,
    })

# Anomalize?
anomalize_flag = False
anom_str = "_anom_" if anomalize_flag else "_nom_"
# Overwrite? 
overwrite_flag = False

# ------- Bounds of interest in time, variable, etc. ---------
model2institute,vbl2key,base_dirs = datre.get_dirinfo()
models = list(model2institute.keys())
rois = dict({
    "eurasia": dict(lat=slice(50,65),lon=slice(-10,130)),
    })
expts = ["control","free","nudged"]
expt_pairs = dict({"n2f": ["nudged","free"], "f2c": ["free","control"], "n2c": ["nudged","control"]})
fcdates = [datetime.datetime(2018,1,25),datetime.datetime(2018,2,8)]
vbl = "t2m"
timesel_event_maxposs = dict(time=slice(datetime.datetime(2018,2,12),datetime.datetime(2018,3,11))) # maximum possible time selection
timesel_event = dict(time=slice(datetime.datetime(2018,2,21),datetime.datetime(2018,3,8))) # actual time selection
# ------------------------------------------------------------

# ------------------------ Repackage data into area- and time-resolved -------------
savedir = "/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018"

clim_filename = join(savedir,f"{vbl}_clim.nc")
if tododict["compute_clim"]:
    print(f"Starting climatology computation")
    # Climatology
    ds_clim = (
            datre.rezero_lons(xr.open_dataarray(datre.get_clim_filename(vbl)))
            .sel(rois["eurasia"]))
    ds_clim.to_netcdf(clim_filename)
ds_clim = xr.open_dataarray(clim_filename)
# TODO change around how climatology is subtracted

era5_filename = join(savedir,f"{vbl}{anom_str}era5.nc")
if tododict["compute_era5"]:
    print(f"Starting era5 computation")
    # Reanalysis
    ds = (
            datre.rezero_lons(
                xr.open_dataset(datre.get_era5_filename(vbl))["t2m"]
                .sel(time=slice(period_begin,period_end)))
            .sel(rois["eurasia"]))
    if anomalize_flag:
        ds_dayofyear = pd.to_datetime(ds["time"].to_numpy()).dayofyear
        ds = ds - (
                ds_clim.sel(dayofyear=ds_dayofyear)
                .assign_coords(dayofyear=ds["time"].to_numpy())
                .rename(dayofyear="time"))
    ds = ds.compute()
    ds.to_netcdf(era5_filename)

# Model results
if tododict["compute_model"]:
    for i_model,model in enumerate(models):
        print(f"Starting {model} computation")
        model_filename = join(savedir,f"{vbl}{anom_str}{model}.nc")
        print(f"{model_filename = }\n\texists? {exists(model_filename) = }")
        if (not exists(model_filename)) or overwrite_flag:
            ds = []
            for i_fcdate,fcdate in enumerate(fcdates):
                preprocess = lambda dsmem: datre.preprocess_gcm_6hrPt(dsmem,vbl,fcdate,period_begin,period_end,rois["eurasia"])
                mem_filenames = dict()
                mem_labels = dict()
                ens_sizes = []
                for expt in expts:
                    print(f"{expt = }")
                    mem_filenames[expt],mem_labels[expt] = datre.get_gcm_6hrPt_filenames(model, vbl, expt, fcdate)
                    print(f"{len(mem_filenames[expt]) = }")
                    ens_sizes.append(len(mem_filenames[expt]))
                # Before loading data, verify there are the same number of files in each
                if ens_sizes[0] > 0 and len(np.unique(ens_sizes)) == 1:
                    ds_fc = []
                    for expt in expts:
                        ds_fc.append(xr.open_mfdataset(mem_filenames[expt], preprocess=preprocess, combine='nested', concat_dim='member').assign_coords(member=mem_labels[expt]))
                    ds_fc = xr.concat(ds_fc, dim="expt").assign_coords(expt=expts)
                    print(f"{ds_fc[0]['time'].to_numpy() = }")
                    print(f"Expts concatenated successfully")
                    #ds_fc = ds_fc.compute()
                    ds.append(ds_fc)
            if len(ds) == 2:
                ds = xr.concat(ds, dim='init').assign_coords(init=fcdates)
                ds.to_netcdf(model_filename)

    


if tododict["compute_area_averages"]:
    # climatology
    aa_clim = datre.area_average(ds_clim)
    # era5
    ds = xr.open_dataset(era5_filename)
    ds_dayofyear = pd.to_datetime(ds["time"].to_numpy()).dayofyear
    aa = datre.area_average(ds) - aa_clim.sel(dayofyear=ds_dayofyear).assign_coords(dayofyear=ds["time"].to_numpy()).rename(dayofyear="time")
    aa.to_netcdf(join(savedir,f"{vbl}_anom_aa_era5.nc"))

    # models
    for i_model,model in enumerate(models):
        print(f"Starting {model} averaging")
        model_filename = join(savedir,f"{vbl}{anom_str}{model}.nc")
        if exists(model_filename):
            ds = xr.open_dataset(model_filename)
            print(f'opened the file')
            # Take area average
            ds_dayofyear = pd.to_datetime(ds["time"].to_numpy()).dayofyear
            print(f'found the day of year')
            aa = datre.area_average(ds) - aa_clim.sel(dayofyear=ds_dayofyear).assign_coords(dayofyear=ds["time"].to_numpy()).rename(dayofyear="time")
            aa.to_netcdf(join(savedir,f"{vbl}_anom_aa_{model}.nc"))

if tododict["compare_to_will"]:
    anom_aa_gcm_will = xr.open_dataarray('/gws/nopw/j04/snapsi/processed/wg2/area_averages/Eurasia_area_avg_T2m.nc')
    anom_aa_era5_will = xr.open_dataarray('/gws/nopw/j04/snapsi/processed/wg2/area_averages/ERA5_regional_avg_Eurasia.nc')
    anom_aa_era5_justin = xr.open_dataset(join(savedir,f'{vbl}_anom_aa_era5.nc'))[vbl]
    for model in ["GloSea6-GC32"]:
        print(f"{model = }")

        # Justin's results
        model_filename = join(savedir,f"{vbl}{anom_str}{model}.nc")
        if exists(model_filename):
            anom_aa_gcm_justin = xr.open_dataset(join(savedir,f"{vbl}_anom_aa_{model}.nc"))[vbl2key[vbl]]
            anom_aa_da_gcm_justin = anom_aa_gcm_justin.rolling(time=4,center=False).mean()
            fig,axes = plt.subplots(nrows=2,ncols=len(expts),figsize=(6*len(expts),6),sharex=True,sharey=True)
            for i_expt,expt in enumerate(expts):
                for i_init,init in enumerate(anom_aa_gcm_justin.init.values):
                    ax = axes[i_init,i_expt]
                    for member in anom_aa_gcm_justin.member.values:
                        hgcm, = xr.plot.plot(anom_aa_da_gcm_justin.sel(expt=expt,init=init,member=member),x="time",color="red", ax=ax, label=model, alpha=0.3)
                    hera, = xr.plot.plot(anom_aa_era5_justin,x="time",color="black",linewidth=3,linestyle="--",ax=ax,label="ERA5")
                    hgcm_mean, = xr.plot.plot(anom_aa_da_gcm_justin.sel(expt=expt,init=init).mean("member"),x="time",color="red",linewidth=3,linestyle="--", ax=ax, label="Ens. mean")

                    ax.set_title(r"init %s, %s"%(fcdates[i_init].strftime("%Y-%m-%d"),expt))
                    ax.set_ylabel(f"{vbl} anomaly [K]")
                    ax.set_xlabel("")
                    ax.legend(handles=[hera,hgcm,hgcm_mean])
            fig.savefig(join('figures','timeseries',f"plot_aa_da_{model}_justin"),**pltsvargs)
            plt.close(fig)

        # Will's results
        if model in list(anom_aa_gcm_will.center):
            fig,axes = plt.subplots(nrows=2,ncols=len(expts),figsize=(6*len(expts),6),sharex=True,sharey=True)
            for i_expt,expt in enumerate(expts):
                for i_init,init in enumerate(anom_aa_gcm_will.init.values):
                    ax = axes[i_init,i_expt]
                    for ens in anom_aa_gcm_will.ens.values:
                        hgcm, = xr.plot.plot(anom_aa_gcm_will.sel(center=model,exp=expt,init=init,ens=ens),x="time",color="red", ax=ax, label=model, alpha=0.3)
                    hgcm_mean, = xr.plot.plot(anom_aa_gcm_will.sel(center=model,exp=expt,init=init).mean(dim="ens"),x="time",color="red",ax=ax,linewidth=3,linestyle="--",label="Ens. mean")
                    hera, = xr.plot.plot(anom_aa_era5_will,x="time",color="black",linewidth=3,linestyle="--",ax=ax,label="ERA5")
                    ax.set_title(r"init %s, %s"%(fcdates[i_init].strftime("%Y-%m-%d"),expt))
                    ax.legend(handles=[hera,hgcm,hgcm_mean])
                    ax.set_ylabel(f"{vbl} anomaly [K]")
                    ax.set_xlabel("")
            fig.savefig(join('figures','timeseries',f"plot_aa_da_{model}_will"),**pltsvargs)
            plt.close(fig)

if tododict["compute_risks"]:
    # Do the full pipeline for each model (so no need to have a ragged array iwth different ensemble eisze)
    timesel = slice(cao_begin,cao_end)
    # Define a few metrics for extremity. By convention, POSITIVE means extreme.
    def extremity_fun(anom_aa, metric):
        if metric == "mean_neg":
            extremity = -anom_aa.sel(time=timesel).mean(dim='time')
        elif metric == "min_neg":
            extremity = -anom_aa.sel(time=timesel).min(dim='time')
        return extremity
    metrics = ['min_neg','mean_neg']
    aa_era5 = xr.open_dataset(join(savedir,f"{vbl}_anom_aa_era5.nc"))
    aa_extremity_era5 = dict({metric: extremity_fun(aa_era5[vbl], metric).item() for metric in metrics}) #aa_era5[vbl].sel(time=timesel).min('time').item(), 'mean': aa_era5[vbl].sel(time=timesel).mean('time').item()})
    risk = xr.DataArray(
            coords={"model": models, "expt": expts, "init": fcdates, 'method': ['binomial','normal','gev'], "metric": ["mean_neg","min_neg"],},
            dims = ['model','expt','init','metric','method',],
            data = np.nan)
    risk_ci = xr.DataArray(
            coords={"model": models, "expt": expts, "init": fcdates, "metric": ["mean_neg","min_neg"], "confint": [0.5,0.95], "side": ["lo","hi"], "citype": ["normal","delta","percentile_bootstrap","basic_bootstrap","wilson"]},
            dims = ['model','expt','init','metric','confint','side','citype'],
            data=np.nan)
    relative_risk = xr.DataArray(
            coords={"model": models, "expt_pair": list(expt_pairs.keys()), "init": fcdates, 'method': ['binomial','normal','gev'], "metric": ["mean_neg","min_neg"]},
            dims = ['model','expt_pair','init','metric','method'],
            data = np.nan)
    relative_risk_ci = xr.DataArray(
            coords={"model": models, "expt_pair": list(expt_pairs.keys()), "init": fcdates, "metric": ["mean_neg","min_neg"], "confint": [0.5,0.95], "side": ["lo","hi"], "citype": ["normal","delta","percentile_bootstrap","basic_bootstrap","koopman"]},
            dims = ['model','expt_pair','init','metric','confint','side','citype'],
            data = np.nan)
    for i_model,model in enumerate(models):
        model_filename = join(savedir,f"{vbl}_anom_aa_{model}.nc")
        print(f"Starting RR calculation for {model}")
        if exists(model_filename):
            aa = xr.open_dataset(model_filename)
            n_mem = aa.member.size
            print(f"ensemble size = {aa.member.size}") 
            for i_metric,metric in enumerate(metrics):
                sspe = dict(model=model, metric=metric) # the sub-selection for point estimatesthat we're modifying
                ssci = dict(model=model, metric=metric) # the sub-selection for confidence intervals that we're modifying
                aa_extremity = extremity_fun(aa[vbl2key[vbl]], metric) # dims: ('init','expt','member'
                # ====================== Absolute risk ===========================
                # *** point estimates ***
                sspe.update(method='binomial')
                nsucc = (aa_extremity > aa_extremity_era5[metric]).sum(dim="member")
                nfail = (aa_extremity <= aa_extremity_era5[metric]).sum(dim="member")
                phat = nsucc / (nsucc + nfail)
                risk.loc[sspe] = phat.transpose("expt","init")
                sspe.update(method='normal')
                mean_exceedance = (aa_extremity - aa_extremity_era5[metric]).mean(dim='member').transpose('expt','init')
                std_exceedance = (aa_extremity - aa_extremity_era5[metric]).std(dim='member').transpose('expt','init')
                risk.loc[sspe] = spnorm.sf(0.0, loc=mean_exceedance, scale=std_exceedance)
                # *** confidence intervals ***
                # --------- normal ----------
                print(f'...normal')
                sspe.update(method='normal')
                ssci.update(citype='normal')
                for ci in risk_ci.confint.values:
                    ssci.update(confint=ci)
                    ssci.update(side='lo')
                    risk_ci.loc[ssci] = spnorm.sf(spnorm.ppf(0.5-0.5*ci) + mean_exceedance/std_exceedance)
                    ssci.update(side='hi')
                    risk_ci.loc[ssci] = spnorm.sf(spnorm.ppf(0.5+0.5*ci) + mean_exceedance/std_exceedance)

                # -------- delta -------------
                print(f"...delta")
                sspe.update(method='binomial')
                ssci.update(citype='delta')
                stderr = np.sqrt(phat * (1-phat) / (nsucc + nfail))
                for ci in risk_ci.confint.values:
                    ssci.update(confint=ci)
                    z = spnorm.ppf(0.5 + 0.5*ci)
                    for (side,sign) in [('lo',-1),('hi',1)]:
                        ssci.update(side=side)
                        risk_ci.loc[ssci] = risk.sel(sspe) + sign*z*stderr

                # -------- bootstrap  --------
                print(f"...bootstrap")
                sspe.update(method='binomial')
                n_boot = 1000
                rng = np.random.default_rng(2718)
                members_boot = rng.choice(np.arange(n_mem), size=(n_boot,n_mem), replace=True)
                risk_boot = xr.DataArray(
                        coords={"expt": expts, "init": fcdates, 'boot': np.arange(n_boot)},
                        dims=['expt','init','boot'],
                        data=np.nan
                        )
                for i_boot in range(n_boot):
                    risk_boot.loc[dict(boot=i_boot)] = (aa_extremity.isel(member=members_boot[i_boot,:]) > aa_extremity_era5[metric]).mean(dim='member').transpose('expt','init')
                for ci in risk_ci.confint.to_numpy():
                    ssci.update(confint=ci)
                    for (side,sign) in [('lo',-1),('hi',1)]:
                        ssci.update(side=side)
                        ssci.update(citype='basic_bootstrap')
                        risk_ci.loc[ssci] = 2*risk.sel(sspe) - sign * risk_boot.quantile(0.5 - sign*0.5*ci, dim="boot")
                        ssci.update(citype='percentile_bootstrap')
                        risk_ci.loc[ssci] = risk_boot.quantile(0.5 + sign*0.5*ci, dim="boot")
                # ------------ wilson ------------
                print(f"...wilson")
                sspe.update(method='binomial')
                ssci.update(citype='wilson')
                for ci in risk_ci.confint.to_numpy():
                    ssci.update(confint=ci)
                    wilson_interval = utils.wilson_score_interval(nsucc,nfail,ci)
                    for i_side,side in enumerate(['lo','hi']):
                        ssci.update(side=side)
                        risk_ci.loc[ssci] = wilson_interval[i_side].transpose("expt","init")

                # ////////////////  Relative risk /////////////////////////////
                sspe = dict(model=model,metric=metric) # the sub-selection for point estimatesthat we're modifying
                ssci = dict(model=model,metric=metric) # the sub-selection for confidence intervals that we're modifying
                for epk in list(expt_pairs.keys()):
                    sspe.update(expt_pair=epk)
                    ssci.update(expt_pair=epk)
                    p1hat = phat.sel(expt=expt_pairs[epk][0],drop=True)
                    p2hat = phat.sel(expt=expt_pairs[epk][1],drop=True)
                    nsucc1_epk = nsucc.sel(expt=expt_pairs[epk][0],drop=True)
                    nfail1_epk = nfail.sel(expt=expt_pairs[epk][0],drop=True)
                    nsucc2_epk = nsucc.sel(expt=expt_pairs[epk][1],drop=True)
                    nfail2_epk = nfail.sel(expt=expt_pairs[epk][1],drop=True)
                    # *** point estimates ***
                    sspe.update(method='binomial')
                    relative_risk.loc[sspe] = p1hat/p2hat #xr.where(p2hat > 0, p1hat/p2hat, np.inf)
                    sspe.update(method='normal')
                    relative_risk.loc[sspe] = risk.sel(dict(model=model,metric=metric,method='normal',expt=expt_pairs[epk][0])) / risk.sel(dict(model=model,metric=metric,method='normal',expt=expt_pairs[epk][1]))
                    # *** confidence intervals ***
                    # ------------ delta ------------
                    sspe.update(method='binomial')
                    ssci.update(citype='delta')
                    stderr_log_rr = np.sqrt((1 - p1hat)/(p1hat * (nsucc1_epk + nfail1_epk)) + (1 - p2hat)/(p2hat * (nsucc2_epk + nfail2_epk))) # dims: (init,expt)
                    for ci in risk_ci.confint.values:
                        ssci.update(confint=ci)
                        z = spnorm.ppf(0.5 + 0.5*ci)
                        for (side,sign) in [('lo',-1),('hi',1)]:
                            relative_risk_ci.loc[ssci] = relative_risk.sel(sspe) * np.exp(sign * stderr_log_rr * z)
                    # ------------ bootstrap -----------
                    sspe.update(method='binomial')
                    rr_boot = xr.DataArray(
                            coords={"init": fcdates, 'boot': np.arange(n_boot)},
                            dims=['init','boot'],
                            data=np.nan
                            )
                    for i_boot in range(n_boot):
                        i1_boot,i2_boot = rng.choice(np.arange(n_boot),size=2,replace=True) # just in case of correlations
                        rr_boot.loc[dict(boot=i_boot)] = risk_boot.sel(expt=expt_pairs[epk][0],boot=i1_boot) / risk_boot.sel(expt=expt_pairs[epk][1],boot=i2_boot)
                    for ci in risk_ci.confint.to_numpy():
                        ssci.update(confint=ci)
                        for (side,sign) in [('lo',-1),('hi',1)]:
                            ssci.update(side=side)
                            ssci.update(citype='basic_bootstrap')
                            relative_risk_ci.loc[ssci] = 2*relative_risk.sel(sspe) - sign * rr_boot.quantile(0.5 + 0.5*ci, dim="boot")
                            ssci.update(citype='percentile_bootstrap')
                            relative_risk_ci.loc[ssci] = rr_boot.quantile(0.5 + sign * 0.5*ci, dim="boot")
                    
                  
                    
    risks = dict({"r": risk, "rci": risk_ci, "rr": relative_risk, "rrci": relative_risk_ci})
    pickle.dump(risks,open(join(savedir,"risks.pickle"),"wb"))

if tododict["plot_risks"]:
    risks = pickle.load(open(join(savedir,'risks.pickle'),'rb'))
    expt_pair_labels = dict({'n2f': r'$P(A|V^-)/P(A)$', 'f2c': r'$P(A)/P(A|V^0)$', 'n2c': r'$P(A|V^-)/P(A|V^0)$'})
    expt_labels = dict({'free': r'$P(A)$', 'control': r'$P(A|V^0)$', 'nudged': r'$P(A|V^-)$'})
    metric_labels = dict({
        'min_neg': r'$A=\{\mathrm{min}_t\ \mathrm{mean}_{x,y}\ T(x,y,t)$ more extreme than ERA5$\}$',
        'mean_neg': r'$A=\{\mathrm{mean}_t\ \mathrm{mean}_{x,y}\ T(x,y,t)$ more extreme than ERA5$\}$',
        })
    citype_labels = dict({
        'percentile_bootstrap': 'CI: percentile bootstrap',
        'basic_bootstrap': 'CI: basic bootstrap',
        'wilson': 'CI: Wilson',
        'delta': 'CI: Delta method',
        'normal': 'CI: Gaussian model',
        })
    method_labels = dict({
        'binomial': 'Binomial model',
        'normal': 'Gaussian model',
        })
    vscale = 1.5
    for (method,citype) in [('binomial','basic_bootstrap'),('binomial','percentile_bootstrap'),('binomial','wilson'),('binomial','delta'),('normal','normal'),]:
        for metric in ['mean_neg','min_neg']:
            sspe = dict(metric=metric,method=method)
            ssci = dict(metric=metric,citype=citype)
            # Plot absolute risk
            fig,axes = plt.subplots(ncols=len(expts), figsize=(6*len(expts),6), sharex=True, sharey=True)
            for i_expt,expt in enumerate(list(risks['r'].expt)):
                sspe.update(expt=expt)
                ssci.update(expt=expt)
                ax = axes[i_expt]
                fccolors = ["red","blue"]
                vert_offsets = [-0.2,0.2]
                handles = []
                for i_fcdate,fcdate in enumerate(fcdates):
                    sspe.update(init=fcdate)
                    ssci.update(init=fcdate)
                    h = ax.scatter(risks['r'].sel(sspe).values, vscale*np.arange(len(models))+vert_offsets[i_fcdate], color=fccolors[i_fcdate], marker='o', s=100, label=r"Init %s"%(fcdate.strftime("%Y-%m-%d")))
                    handles.append(h)
                    for i_model,model in enumerate(models):
                        ssci.update(model=model)
                        for (ci,linewidth) in [(0.5,4),(0.95,1)]:
                            ssci.update(confint=ci)
                            ax.plot(risks['rci'].sel(ssci).values, (vscale*i_model+vert_offsets[i_fcdate])*np.ones(2), color=fccolors[i_fcdate], linewidth=linewidth,)
                ax.axvline(1, linestyle="--", color='black')
                for i_model,model in enumerate(models):
                    ax.axhline(vscale*(i_model-0.5),color='gray',alpha=0.5,zorder=-1)
                    ax.axhline(vscale*(i_model+0.5),color='gray',alpha=0.5,zorder=-1)
                ax.set_xlabel(expt_labels[expt.item()])
                ax.set_ylabel("")
                ax.xaxis.set_tick_params(which="both",labelbottom=True)
                ax.yaxis.set_tick_params(which="both",labelbottom=False)
    
            for i_ax in range(len(axes)):
                axes[i_ax].set_xlim([-0.1,1.1])
            axes[0].set_yticks(vscale*np.arange(len(models)))
            axes[0].set_yticklabels(models)
            axes[0].yaxis.set_tick_params(which="both",labelbottom=True)
            axes[0].legend(handles=handles,loc=(0,1.05))
            axes[1].text(0.0, 1.05, f'{metric_labels[metric]}\n{citype_labels[citype]}', horizontalalignment='left', verticalalignment='bottom', transform=axes[1].transAxes)
            fig.savefig(join('figures','risk_analysis',f"AR_MT{metric}_PE{method}_CI{citype}"),**pltsvargs)
            plt.close(fig)

    # Plot relative risk
    for (method,citype) in [('binomial','basic_bootstrap'),('binomial','percentile_bootstrap'),('binomial','delta'),('normal','normal'),]:
        for metric in ['mean_neg','min_neg']:
            sspe = dict(metric=metric,method=method)
            ssci = dict(metric=metric,citype=citype)
            fig,axes = plt.subplots(ncols=len(risks['rr'].expt_pair), figsize=(6*len(risks['rr'].expt_pair),6), sharex=True, sharey=True)
            for i_epk,epk in enumerate(list(risks['rr'].expt_pair)):
                sspe.update(expt_pair=epk)
                ssci.update(expt_pair=epk)
                ax = axes[i_epk]
                fccolors = ["red","blue"]
                vert_offsets = [-0.2,0.2]
                handles = []
                for i_fcdate,fcdate in enumerate(fcdates):
                    sspe.update(init=fcdate)
                    ssci.update(init=fcdate)
                    h = ax.scatter(risks['rr'].sel(sspe).values, vscale*np.arange(len(models))+vert_offsets[i_fcdate], color=fccolors[i_fcdate], marker='o', s=100, label=r"Init %s"%(fcdate.strftime("%Y-%m-%d")))
                    handles.append(h)
                    for i_model,model in enumerate(models):
                        ssci.update(model=model)
                        for (ci,linewidth) in [(0.5,4),(0.95,1)]:
                            ssci.update(confint=ci)
                            ax.plot(risks['rrci'].sel(ssci).values, (vscale*i_model+vert_offsets[i_fcdate])*np.ones(2), color=fccolors[i_fcdate], linewidth=linewidth,)
                ax.axvline(1, linestyle="--", color='black')
                for i_model,model in enumerate(models):
                    ax.axhline(vscale*(i_model-0.5),color='gray',alpha=0.5,zorder=-1)
                    ax.axhline(vscale*(i_model+0.5),color='gray',alpha=0.5,zorder=-1)
                ax.set_xlabel(expt_pair_labels[epk.item()])
                ax.set_ylabel("")
                ax.xaxis.set_tick_params(which="both",labelbottom=True)
                ax.yaxis.set_tick_params(which="both",labelbottom=False)
    
            for i_ax in range(len(axes)):
                axes[i_ax].set_xlim([0.0,10.0])
            axes[0].set_yticks(vscale*np.arange(len(models)))
            axes[0].set_yticklabels(models)
            axes[0].yaxis.set_tick_params(which="both",labelbottom=True)
            axes[0].legend(handles=handles,loc=(0,1.05))
            axes[1].text(0.0, 1.08, f'{metric_labels[metric]}\n{method_labels[method]}\n{citype_labels[citype]}', horizontalalignment='left', verticalalignment='bottom', transform=axes[1].transAxes)
            fig.savefig(join('figures','risk_analysis',f"RR_MT{metric}_PE{method}_CI{citype}"),**pltsvargs)
            plt.close(fig)







if False and tododict["compute_relative_risk"]:
    # For each model and each ensemble member, calculate a few scalar metrics for extremity 
    timesel = slice(cao_begin,cao_end)
    extremity_differences_ensmean = xr.DataArray(
            coords={"model": models, "expt": expts, "init": fcdates, "statistic": ["mean","min"], "confint": [0.0,0.5,0.95], "side": ["lo","hi"], "citype": ["bootstrap","wilson","koopman"]},
            dims = ['model','expt','init','statistic','confint','side','citype'],
            data = 0.0)
    extreme_counts = xr.DataArray(
            coords={"model": models, "expt": expts, "init": fcdates, "category": [0,1], "statistic": ["mean","min"]},
            dims = ['model','expt','init','category','statistic','confint','side','citype'],
            data = 0)
    aa_era5 = xr.open_dataset(join(savedir,f"{vbl}_anom_aa_era5.nc"))
    aa_sumstat_era5 = dict({"min": aa_era5[vbl].sel(time=timesel).min('time').item(), 'mean': aa_era5[vbl].sel(time=timesel).mean('time').item()})
    for i_model,model in enumerate(models):
        model_filename = join(savedir,f"{vbl}_anom_aa_{model}.nc")
        if exists(model_filename):
            aa = xr.open_dataset(model_filename)
            # mind the signs, since negatives are the extremes
            # Mean
            extremity_differences_ensmean.loc[dict(model=model,statistic='mean')] = (
                    -aa[vbl2key[vbl]].sel(time=timesel).mean(dim=["time","member"]) + aa_sumstat_era5['mean']).transpose("expt","init")
            extreme_counts.loc[dict(model=model,category=0,statistic='mean')] = (
                    aa[vbl2key[vbl]].sel(time=timesel).mean(dim=["time"]) >= aa_sumstat_era5['mean']
                    ).sum(dim="member").transpose('expt','init')
            extreme_counts.loc[dict(model=model,category=1,statistic='mean')] = (
                    aa[vbl2key[vbl]].sel(time=timesel).mean(dim=["time"]) < aa_sumstat_era5['mean']
                    ).sum(dim="member").transpose('expt','init')
            # TODO compute confidence intervals
            # Min
            extremity_differences_ensmean.loc[dict(model=model,statistic='min')] = (
                    -aa[vbl2key[vbl]].sel(time=timesel).min(dim="time").mean("member") + aa_sumstat_era5['min']).transpose("expt","init")
            extreme_counts.loc[dict(model=model,category=0,statistic='min')] = (
                    aa[vbl2key[vbl]].sel(time=timesel).min(dim="time") >= aa_sumstat_era5['min']
                    ).sum(dim="member").transpose('expt','init')
            extreme_counts.loc[dict(model=model,category=1,statistic='min')] = (
                    aa[vbl2key[vbl]].sel(time=timesel).min(dim=["time"]) < aa_sumstat_era5['min']
                    ).sum(dim="member").transpose('expt','init')
    extreme_counts.to_netcdf(join(savedir,"extreme_counts.nc"))
    extremity_differences_ensmean.to_netcdf(join(savedir,"extremity_differences_ensmean.nc"))

if False:
    extreme_counts = xr.open_dataarray(join(savedir,"extreme_counts.nc"))
    extremity_differences_ensmean = xr.open_dataarray(join(savedir,"extremity_differences_ensmean.nc"))
    if tododict["plot_relative_risk"]:
        totals = extreme_counts.sum(dim="category")
        totals = totals.where(totals>0, np.nan)
        extreme_probs = extreme_counts.sel(category=1,drop=True)/totals
        exptpairs = ["n2f","f2c","n2c"]
        exptpair_labels_rr = [r"$P(A|V^-)/P(A)$",r"$P(A)/P(A|V^0)$",r"$P(A|V^-)/P(A|V^0)$"]
        exptpair_labels_ed = [r"$E[-\Delta T|V^-] - E[-\Delta T]$",r"$E[-\Delta T] - E[-\Delta T|V^0]$",r"$E[-\Delta T|V^-] - E[-\Delta T|V^0]$"]
    
        # Relative risk
        rr = xr.DataArray(
                coords={"model": models, "exptpair": ["n2f","f2c","n2c"], "init": fcdates, 'statistic': ['mean','min'], "confint": [0.0, 0.5, 0.95], "side": ["lo","hi"]},
                dims=["model","exptpair","init",'statistic',"confint","side"],
                data=np.nan)
    
        rr.loc[dict(exptpair="n2f",confint=0,side="lo")] = extreme_probs.sel(expt="nudged",drop=True)/extreme_probs.sel(expt="free")
        rr.loc[dict(exptpair="f2c",confint=0,side="lo")] = extreme_probs.sel(expt="free",drop=True)/extreme_probs.sel(expt="control")
        rr.loc[dict(exptpair="n2c",confint=0,side="lo")] = extreme_probs.sel(expt="nudged",drop=True)/extreme_probs.sel(expt="control")
        # TODO calculate Wilson and other confidence intervals for the binomial proportions
    
        # Extremity difference
        ed = xr.DataArray(
                coords={"model": models, "exptpair": ["n2f","f2c","n2c"], "init": fcdates, 'statistic': ['mean','min'], "confint": [0.0, 0.5, 0.95], "side": ["lo","hi"]},
                dims=["model","exptpair","init",'statistic',"confint","side"],
                data=np.nan)
    
        ed.loc[dict(exptpair="n2f",confint=0,side="lo")] = extremity_differences_ensmean.sel(expt="nudged",drop=True) - extremity_differences_ensmean.sel(expt="free",drop=True)
        ed.loc[dict(exptpair="f2c",confint=0,side="lo")] = extremity_differences_ensmean.sel(expt="free",drop=True) - extremity_differences_ensmean.sel(expt="control",drop=True)
        ed.loc[dict(exptpair="n2c",confint=0,side="lo")] = extremity_differences_ensmean.sel(expt="nudged",drop=True) - extremity_differences_ensmean.sel(expt="control",drop=True)
    
    
        for statistic in ['mean','min']:
            # Relative risk
            fig,axes = plt.subplots(ncols=rr.exptpair.size, figsize=(6*rr.exptpair.size,6), sharex=True, sharey=True)
            for i_exptpair,exptpair in enumerate(list(rr.exptpair)):
                ax = axes[i_exptpair]
                fccolors = ["red","blue"]
                handles = []
                for i_fcdate,fcdate in enumerate(fcdates):
                    h = ax.scatter(rr.sel(exptpair=exptpair,init=fcdate,statistic=statistic,confint=0,side="lo").values, np.arange(len(models)), color=fccolors[i_fcdate], marker='o', s=100, label=r"Init %s"%(fcdate.strftime("%Y-%m-%d")))
                    handles.append(h)
                ax.axvline(1, linestyle="--", color='black')
                for i_model,model in enumerate(models):
                    ax.axhline(i_model,color='gray',alpha=0.5,zorder=-1)
                ax.set_xlabel(exptpair_labels_rr[i_exptpair])
                ax.set_ylabel("")
                ax.xaxis.set_tick_params(which="both",labelbottom=True)
                ax.yaxis.set_tick_params(which="both",labelbottom=False)
    
            axes[0].set_yticks(np.arange(len(models)))
            axes[0].set_yticklabels(models)
            axes[0].yaxis.set_tick_params(which="both",labelbottom=True)
            axes[0].legend(handles=handles,loc=(0,1.05))
            fig.savefig(f"plot_relative_risk_{statistic}",**pltsvargs)
            plt.close(fig)
    
            # Extremity differences
            fig,axes = plt.subplots(ncols=ed.exptpair.size, figsize=(6*ed.exptpair.size,6), sharex=True, sharey=True)
            for i_exptpair,exptpair in enumerate(list(ed.exptpair)):
                ax = axes[i_exptpair]
                fccolors = ["red","blue"]
                handles = []
                for i_fcdate,fcdate in enumerate(fcdates):
                    h = ax.scatter(ed.sel(exptpair=exptpair,init=fcdate,statistic=statistic,confint=0,side="lo").values, np.arange(len(models)), color=fccolors[i_fcdate], marker='o', s=100, label=r"Init %s"%(fcdate.strftime("%Y-%m-%d")))
                    handles.append(h)
                ax.axvline(0, linestyle="--", color='black')
                for i_model,model in enumerate(models):
                    ax.axhline(i_model,color='gray',alpha=0.5,zorder=-1)
                ax.set_xlabel(exptpair_labels_ed[i_exptpair])
                ax.set_ylabel("")
                ax.xaxis.set_tick_params(which="both",labelbottom=True)
                ax.yaxis.set_tick_params(which="both",labelbottom=False)
    
            axes[0].set_yticks(np.arange(len(models)))
            axes[0].set_yticklabels(models)
            axes[0].yaxis.set_tick_params(which="both",labelbottom=True)
            axes[0].legend(handles=handles,loc=(0,1.05))
            fig.savefig(f"plot_extremity_difference_{statistic}",**pltsvargs)
            plt.close(fig)
    




    

    

    

        
    
