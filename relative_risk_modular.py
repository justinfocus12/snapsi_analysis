import numpy as np
import pandas as pd
import datetime
import pickle
import xarray as xr
import dask
from os import makedirs
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
import stat_functions as stfu

def reduce_clim(qoidict, tododict, stagefiles):
    # Compute daily and areally-averaged temperatures for climatology 
    if tododict['avgD']:
        avgD = (
                datre.rezero_lons(xr.open_dataarray(datre.get_clim_filename(qoidict['var_name'])))
                .sel(qoidict['spacesel'])
                #.sel(qoidict['timesel_maxposs'])
                )
        avgD.to_netcdf(stagefiles['avgD'])
    if tododict['avgDA']:
        avgD = xr.open_dataarray(stagefiles['avgD'])
        avgDA = datre.area_average(avgD)
        avgDA.to_netcdf(stagefiles['avgDA'])
    return

def reduce_era5(qoidict, tododict, stagefiles):
    # Compute daily and areally-averaged temperatures for ERA5
    if tododict['avgD']:
        avgD = (
                datre.rezero_lons(
                    xr.open_dataset(datre.get_era5_filename(qoidict['var_name']))[qoidict['var_name']]
                    .sel(qoidict['timesel_maxposs']))
                .sel(qoidict['spacesel'])
                )
        avgD.to_netcdf(stagefiles['avgD'])
    if tododict['avgDA']:
        avgD = xr.open_dataarray(stagefiles['avgD'])
        avgDA = datre.area_average(avgD)
        avgDA.to_netcdf(stagefiles['avgDA'])
    return

def reduce_gcm(qoidict, stagefiles, tododict):
    if tododict['avgD'] and (tododict['overwrite'] or (not exists(stagefiles['avgD']))):
        ds = []
        for i_fcdate,fcdate in enumerate(qoidict['fcdates']):
            preprocess = lambda dsmem: datre.preprocess_gcm_6hrPt(dsmem,qoidict['var_name'],fcdate,qoidict['timesel_maxposs'],qoidict['spacesel'])
            mem_filenames = dict()
            mem_labels = dict()
            ens_sizes = []
            for expt in qoidict['expts']:
                print(f"{expt = }")
                mem_labels[expt],mem_filenames[expt] = list(stagefiles[fcdate][expt].keys()), list(stagefiles[fcdate][expt].values())
                print(f"{mem_labels[expt] = }")
                print(f"{mem_filenames[expt] = }")
                ens_sizes.append(len(mem_filenames[expt]))
            # Before loading data, verify there are the same number of files in each
            if ens_sizes[0] > 0 and len(np.unique(ens_sizes)) == 1:
                ds_fc = []
                for expt in qoidict['expts']:
                    ds_fc.append(xr.open_mfdataset(mem_filenames[expt], preprocess=preprocess, combine='nested', concat_dim='member').assign_coords(member=mem_labels[expt]))
                ds_fc = xr.concat(ds_fc, dim="expt").assign_coords(expt=qoidict['expts'])
                print(f"{ds_fc[0]['time'].to_numpy() = }")
                print(f"Expts concatenated successfully")
                ds.append(ds_fc)
        if len(ds) == 2:
            ds = xr.concat(ds, dim='init').assign_coords(init=qoidict['fcdates'])
            ds.to_netcdf(stagefiles['avgD'])
    if tododict['avgDA'] and (tododict['overwrite'] or (not exists(stagefiles['avgDA']))):
        # Don't anomalize, because absolute minimum temperature might be more relevant than minimum anomaly. Maybe we should subtract the min over the time period from the min over the time period from ERA5
        ds_avgD = xr.open_dataset(stagefiles['avgD'])
        ds_avgDA = datre.area_average(ds_avgD)
        ds_avgDA.to_netcdf(stagefiles['avgDA'])
    return

def plot_avgDA_timeseries(qoidict, stagefiles_model, stagefiles_era5, stagefiles_clim):
    # Only for 1 model
    avgDA_era5 = xr.open_dataarray(stagefiles_era5['avgDA'])
    doy = pd.to_datetime(avgDA_era5['time'].to_numpy()).dayofyear
    avgDA_clim = xr.open_dataarray(stagefiles_clim['avgDA']).sel(dayofyear=doy).rename({'dayofyear': 'time'}).assign_coords({'time': avgDA_era5['time']})
    avgDA_gcm = xr.open_dataarray(stagefiles_model['avgDA'])
    fig,axes = plt.subplots(nrows=2,ncols=len(qoidict['expts']),figsize=(6*len(qoidict['expts']),6),sharex=True,sharey=True)
    for i_expt,expt in enumerate(qoidict['expts']):
        for i_fcdate,fcdate in enumerate(avgDA_gcm.init.values):
            ax = axes[i_fcdate,i_expt]
            for member in avgDA_gcm.member.values:
                hgcm, = xr.plot.plot(avgDA_gcm.sel(expt=expt,init=fcdate,member=member),x="time",color="red", ax=ax, label=model, alpha=0.3)
            hera, = xr.plot.plot(avgDA_era5,x="time",color="black",linewidth=3,linestyle="--",ax=ax,label="ERA5")
            hclim, = xr.plot.plot(avgDA_clim,x="time",color="gray",linewidth=4,linestyle="-",alpha=0.5,ax=ax,label="Climatology")
            hgcm_mean, = xr.plot.plot(avgDA_gcm.sel(expt=expt,init=fcdate).mean("member"),x="time",color="red",linewidth=3,linestyle="--", ax=ax, label="Ens. mean")

            ax.set_title(r"Init %s, %s"%(qoidict['fcdates'][i_fcdate].strftime("%Y-%m-%d"),expt))
            ax.set_ylabel(f"{qoidict['var_name']} [K]")
            ax.set_xlabel("")
    axes[0,0].legend(handles=[hclim,hera,hgcm,hgcm_mean], loc=(-1,0))
    fig.savefig(stagefiles_model['avgDA_plot'],**pltsvargs)
    plt.close(fig)
    return

def risk_calc_pipeline_era5(qoidict, tododict, stagefiles):
    print(f"About to reduce reanalysis")
    reduce_era5(qoidict, tododict, stagefiles)
    return

    

def risk_calc_pipeline_1model(qoidict, tododict, stagefiles_model, stagefiles_era5, stagefiles_clim):
    print(f"{stagefiles_model.keys() = }")
    # qoidict: quantities of interest
    # tododict: to-do items
    # avgD: daily averaged
    # avgDA: daily and area-averaged
    # TODO: add sub-regional averages
    if tododict['avgD'] or tododict['avgDA']:
        print(f"...reducing")
        reduce_gcm(qoidict, stagefiles_model, tododict)
    if tododict['plot_avgDA']:
        print(f"...plotting")
        plot_avgDA_timeseries(qoidict, stagefiles_model, stagefiles_era5, stagefiles_clim)
    avgDA_model = xr.open_dataarray(stagefiles_model['avgDA']) # Do we need to further subset by the variable? Probably not because didn't subtract off ERA5
    avgDA_era5 = xr.open_dataarray(stagefiles_era5['avgDA']) # Do we need to further subset by the variable? Probably not because didn't subtract off ERA5
    # TODO add other severity metrics which are functions of more moderately reduced data
    for svmetric in list(qoidict['severity_metrics_families'].keys()):
        print(f'{svmetric = }')
        if tododict['compute_severity']:
            severity_model = severity_fun_avgDA(avgDA_model.sel(qoidict['timesel_event']), svmetric) # dims: (fcdate,expt,member,svmetric)
            severity_model.to_netcdf(stagefiles_model['severity'][svmetric])
        severity_model = xr.open_dataarray(stagefiles_model['severity'][svmetric])
        severity_era5 = severity_fun_avgDA(avgDA_era5.sel(qoidict['timesel_event']), svmetric)
        thresh_bounds = np.array([min(severity_model.min().item(),severity_era5.item()), max(severity_model.max().item(),severity_era5.item())])
        elongation = 0.0
        thresh_bounds[1] = (1+elongation)*thresh_bounds[1] - elongation*thresh_bounds[0]
        print(f'{thresh_bounds = }')
        thresh_list = np.linspace(thresh_bounds[0], thresh_bounds[1], 30)
        for family in qoidict['severity_metrics_families'][svmetric]:
            print(f'{family = }')
            if tododict['compute_risk']:
                # Absolute risks
                n_boot = qoidict['n_boot']
                rngseed = 987405
                # TODO specialize how uncertainty is specified depending on the statistical family
                ar = xr.DataArray( # absolute risk
                        coords={'init': qoidict['fcdates'], 'expt': qoidict['expts'], 'boot': np.arange(n_boot+1), 'thresh': thresh_list},
                        dims=['init','expt','boot','thresh'],
                        data=np.nan)
                params = xr.DataArray(
                        coords={'init': qoidict['fcdates'], 'expt': qoidict['expts'], 'param_name': stfu.param_names(family), 'boot': np.arange(n_boot+1)},
                        dims=['init','expt','param_name','boot'],
                        data=np.nan)
                for init in qoidict['fcdates']:
                    print(f'{init = }')
                    for expt in qoidict['expts']:
                        #print(f'{expt = }')
                        S = severity_model.sel(init=init, expt=expt).to_numpy()
                        #print(f'{S = }')
                        theta = stfu.fit_statistical_model(S, family, thresh=np.min(thresh_bounds)-1e-10, n_boot=n_boot, rngseed=rngseed)
                        #print(f'{theta = }')
                        ar.loc[dict(init=init, expt=expt)] = stfu.absolute_risk_parametric(family, theta, thresh_list)
                        for pn in params.param_name.to_numpy():
                            params.loc[dict(init=init, expt=expt, param_name=pn)] = theta[pn]

                abs_risk = xr.Dataset(
                        data_vars={'abs_risk': ar, 'params': params})
                abs_risk.attrs = {'severity_era5': severity_era5.item()}
                abs_risk.to_netcdf(stagefiles_model['abs_risk'][svmetric][family])

                # Relative risk
                rr = xr.DataArray(
                        coords={'init': qoidict['fcdates'], 'expt_pair': list(qoidict['expt_pairs'].keys()), 'boot': np.arange(n_boot+1), 'thresh': thresh_list},
                        dims=['init','fcdates','expt_pair','boot','thresh'],
                        data=np.nan)
                for init in qoidict['fcdates']:
                    for expt_pair in qoidict['expt_pairs']:
                        expt0,expt1 = qoidict['expt_pairs'][expt_pair]
                        rr.loc[dict(init=init,expt_pair=expt_pair)] = ar.sel(init=init,expt=expt0) / ar.sel(init=init,expt=expt1)
                rr.to_netcdf(stagefiles_model['rel_risk'][svmetric][family])

                
            abs_risk = xr.open_dataset(stagefiles_model['abs_risk'][svmetric][family])
            rel_risk = xr.open_dataarray(stagefiles_model['rel_risk'][svmetric][family])
            if tododict['plot_risk']:
                # -------------- Absolute risk -------------------
                # Compute return periods and plot 
                fig = plt.figure(constrained_layout=True)
                fig,axes = plt.subplots(ncols=3, nrows=len(qoidict['expts']), figsize=(18,6*len(qoidict['expts'])), sharey=True, sharex='col')
                # Left: histograms in vertical
                # Middle: return period curves 
                # Right: quantile plots
                for i_expt,expt in enumerate(qoidict['expts']):
                    handles = []
                    for i_init,init in enumerate(qoidict['fcdates']):
                        theta = {pn: abs_risk['params'].sel(init=init, expt=expt, param_name=pn) for pn in stfu.param_names(family)}
                        ax = axes[i_expt,0]
                        S = severity_model.sel(init=init, expt=expt).to_numpy()[np.newaxis, :]
                        #print(f'{S.shape = }')
                        hist,bin_edges = np.histogram(severity_model.sel(expt=expt,init=init), bins=5, density=True)
                        bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
                        ax.plot(hist, bin_centers, marker='.', **qoidict['dispprop']['fcdates'][init])
                        ax.xaxis.set_tick_params(which='both',labelbottom=True)
                        ax.set_ylabel(qoidict['dispprop']['severity_metrics'][svmetric]['label'])
                        ax.set_title(f'{expt} histogram')

                        ax = axes[i_expt,1]
                        ar_par = abs_risk['abs_risk'].sel(expt=expt,init=init).to_numpy()
                        thresh_list_empirical = np.sort(S.flat)
                        #print(f'{thresh_list_empirical = }')
                        ar_emp = stfu.absolute_risk_empirical(S, thresh_list_empirical)
                        return_period_parametric = 1.0/np.where(ar_par>1e-10, ar_par, np.nan)
                        return_period_empirical = 1.0/np.where(ar_emp>1e-10, ar_emp, np.nan)
                        hpar, = ax.plot(return_period_parametric[0,:], thresh_list, **qoidict['dispprop']['fcdates'][init])
                        # Confidence interval
                        ax.fill_betweenx(thresh_list, np.quantile(return_period_parametric, 0.025, axis=0), np.quantile(return_period_parametric, 0.975, axis=0), **qoidict['dispprop']['fcdates'][init], alpha=0.3, zorder=-1)
                        
                        hemp = ax.scatter(return_period_empirical[0,:], thresh_list_empirical, marker='x', **qoidict['dispprop']['fcdates'][init])
                        handles.append(hpar)
                        ax.set_xscale('log')
                        ax.xaxis.set_tick_params(which='both',labelbottom=True)
                        ax.yaxis.set_tick_params(which='both',labelbottom=True)
                        ax.set_title(f'{expt} return levels')

                        ax = axes[i_expt,2]
                        qpar = stfu.quantile_parametric(family, theta, ar_emp.flatten())
                        ax.scatter(ar_emp, qpar[0,:], color='black', marker='.')
                        ax.plot([min(np.min(ar_emp),np.min(qpar)), max(np.max(ar_emp),np.max(qpar))], color='black', linestyle='--')
                        ax.set_xlabel("Empirical risk")
                        ax.set_ylabel("Parametric risk")
                        

                    for ax in axes[i_expt,:]:
                        hera5 = ax.axhline(severity_era5.item(), color='black', linestyle='--', label='ERA5')
                handles.append(hera5)
                axes[0,1].legend(handles=handles,loc='lower right')
                axes[-1,0].set_xlabel('Probability density')
                axes[-1,1].set_xlabel('Return period [years]')
                for ax in axes[:,1]:
                    ax.set_xlim([1.0, 500.0])
                fig.suptitle(f'Absolute risk with {qoidict["dispprop"]["families"][family]["label"]} model')
                fig.savefig(stagefiles_model['abs_risk_plot'][svmetric][family], **pltsvargs)
                plt.close(fig)
                # -------------- Relative risk -------------------
                # Compute return periods and plot 
                fig = plt.figure(constrained_layout=True)
                fig,axes = plt.subplots(ncols=3, nrows=len(qoidict['expt_pairs']), figsize=(18,6*len(qoidict['expts'])), sharey=True, sharex='col')
                for i_expt_pair,expt_pair in enumerate(list(qoidict['expt_pairs'].keys())):
                    expt0,expt1 = qoidict['expt_pairs'][expt_pair]
                    handles_ratio = []
                    for i_init,init in enumerate(qoidict['fcdates']):
                        # Left and center: return period curves for both initialization dates
                        ax = axes[i_expt_pair,i_init]
                        handles = []
                        for expt in [expt0,expt1]:
                            S = severity_model.sel(init=init, expt=expt).to_numpy()[np.newaxis, :]

                            ar_par = abs_risk['abs_risk'].sel(expt=expt,init=init).to_numpy()
                            thresh_list_empirical = np.sort(S.flat)
                            ar_emp = stfu.absolute_risk_empirical(S, thresh_list_empirical)
                            return_period_parametric = 1.0/np.where(ar_par>1e-10, ar_par, np.nan)
                            return_period_empirical = 1.0/np.where(ar_emp>1e-10, ar_emp, np.nan)
                            hpar, = ax.plot(return_period_parametric[0,:], thresh_list, **qoidict['dispprop']['expts'][expt], label=expt)
                            # Confidence interval
                            ax.fill_betweenx(thresh_list, np.quantile(return_period_parametric, 0.025, axis=0), np.quantile(return_period_parametric, 0.975, axis=0), **qoidict['dispprop']['expts'][expt], alpha=0.25, zorder=-1)
                        
                            handles.append(hpar)
                            hemp = ax.scatter(return_period_empirical[0,:], thresh_list_empirical, marker='x', **qoidict['dispprop']['expts'][expt])
                        ax.legend(handles=handles)
                        ax.set_xscale('log')
                        #ax.xaxis.set_tick_params(which='both',labelbottom=True)
                        #ax.yaxis.set_tick_params(which='both',labelbottom=True)
                        ax.set_title(r'Init %s'%(init.strftime('%Y-%m-%d')))
                        if i_init==0: ax.set_ylabel(qoidict['dispprop']['severity_metrics'][svmetric]['label'])

                        # Right: ratios of return periods
                        ax = axes[i_expt_pair,2]
                        hrat, = ax.plot(rel_risk.sel(expt_pair=expt_pair,init=init,boot=0).to_numpy().flat, thresh_list, **qoidict['dispprop']['fcdates'][init], label=r'Init %s'%(init.strftime('%Y-%m-%d')))
                        handles_ratio.append(hrat)
                        ax.fill_betweenx(thresh_list, *[rel_risk.sel(expt_pair=expt_pair,init=init,boot=slice(1,None)).quantile(q, dim='boot').to_numpy().flat for q in [0.275,0.975]], **qoidict['dispprop']['fcdates'][init], alpha=0.3, zorder=-1)
                        #ax.xaxis.set_tick_params(which='both',labelbottom=True)
                        #ax.yaxis.set_tick_params(which='both',labelbottom=True)
                        ax.set_title(r'$\frac{\mathrm{%s}}{\mathrm{%s}}$ relative risk'%(expt0,expt1))
                    axes[i_expt_pair,2].legend(handles=handles_ratio)
                    for ax in axes[i_expt_pair,:]:
                        hera5 = ax.axhline(severity_era5.item(), color='black', linestyle='--', label='ERA5')
                axes[-1,0].set_xlabel('Return period [years]')
                axes[-1,1].set_xlabel('Return period [years]')
                axes[-1,2].set_xlabel('Relative risk')
                for ax in axes[:,:2].flat:
                    ax.set_xlim([1.0, 500.0])
                for ax in axes[:,2]:
                    ax.set_xlim([0,10])
                    ax.axvline(1.0, color='gray', linestyle='-', zorder=-1, alpha=0.5, linewidth=5)
                fig.suptitle(f'Relative risk with {qoidict["dispprop"]["families"][family]["label"]} model')
                fig.savefig(stagefiles_model['rel_risk_plot'][svmetric][family], **pltsvargs)
                plt.close(fig)
    return
        
def severity_fun_avgDA(avgDA, svmetric):
    # measures of severity for daily- and areally-averaged timeseries data
    if svmetric == 'mintemp': 
        severity = -avgDA.min(dim='time')
    elif svmetric == 'meantemp':
        severity = -avgDA.mean(dim='time')
    elif svmetric == 'durationcold':
        # Calculate the maximum duration with temps < some threshold
        pass
    return severity

# -----------------------------------------------------------

# ------ main procedure --------------
resultdir = '/gws/nopw/j04/snapsi/processed/wg2/ju26596/feb2018/results_2024-01-13'
figdir = '/home/users/ju26596/snapsi_analysis_figures/feb2018/figures_2024-01-13'
# Bounds of interest in time, variable, etc. 
model2institute,vbl2key,base_dirs = datre.get_dirinfo()
models = list(model2institute.keys())

qoidict = dict({
    'fcdates': np.array([datetime.datetime(2018,1,25),datetime.datetime(2018,2,8)]),
    'expts': ['control','free','nudged'],
    'expt_pairs': dict({"n2f": ["nudged","free"], "f2c": ["free","control"], "n2c": ["nudged","control"]}),
    'timesel_maxposs': dict(time=slice(datetime.datetime(2018,2,12),datetime.datetime(2018,3,11))),
    'timesel_event': dict(time=slice(datetime.datetime(2018,2,21),datetime.datetime(2018,3,8))),
    'spacesel': dict(lat=slice(50,65),lon=slice(-10,130)),
    'var_name': 't2m', # which variable we're interested in the extrema of 
    'severity_metrics_families': dict({
        'mintemp': ['gpd','normal','gev'][1:],
        }),
    'n_boot': 500,
    })
# Display properties
qoidict['dispprop'] = dict({
    'families': dict({
        'gpd': dict(label='GPD'),
        'gev': dict(label='GEV'),
        'normal': dict(label='Gaussian'),
        }),
    'fcdates': dict({
        qoidict['fcdates'][0]: dict({
            'color': 'red',
            }),
        qoidict['fcdates'][1]: dict({
            'color': 'blue',
            }),
        }),
    'severity_metrics': dict({ # display properties
        'mintemp': dict({
            'label': r'$-\mathrm{min}_t\ \mathrm{mean}_{x,y}T$ [K]',
            }),
        'meantemp': dict({
            'label': r'$-\mathrm{min}_t\ \mathrm{mean}_{x,y}T$ [K]',
            }),
        }),
    'expts': dict({
        'control': dict({
            'color': 'purple',
            }),
        'free': dict({
            'color': 'darkorange',
            }),
        'nudged': dict({
            'color': 'cyan',
            }),
        }),
    })

def find_true_in_dict(d):
    # Thanks Bing Chat
    if isinstance(d, dict):
        for v in d.values():
            if find_true_in_dict(v):
                return True
    elif isinstance(d, bool) or isinstance(d,int):
        return bool(d)
    return False

tododict = dict({
    'clim': dict({
        'avgD':   0,
        'avgDA':  0,
        }),
    'era5': dict({
        'avgD':   0,
        'avgDA':  0,
        }),
    'gcms': dict({
        model: dict({ 
            'overwrite':        0,
            'avgD':             0,
            'avgDA':            0,
            'plot_avgDA':       0,
            'compute_severity': 0,
            'compute_risk':     0,
            'plot_risk':        1,
            })
        for model in models
        }),
    'riskcomp': dict({ # comparison or comprehensive
        #'plot_avgDA':      1,
        #'plot_rr':         1, 
        'compare_params':  0,
        'plot_params':     1,
        })
    })

# ----- Begin ad-hoc adjustments to tododict -----------
# ----- End ad-hoc adjustments to tododict ------------
def main():
    print(f'Building stagefiles...', end='')
    stagefiles = dict()
    for dataset in ['clim','era5']:
        stagefiles[dataset] = dict()
        resultdir_dataset = join(resultdir,dataset)
        figdir_dataset = join(figdir,dataset)
        makedirs(resultdir_dataset, exist_ok=True)
        makedirs(figdir_dataset, exist_ok=True)
        stagefiles[dataset]['avgD'] = join(resultdir_dataset, 'avgD.nc')
        stagefiles[dataset]['avgDA'] = join(resultdir_dataset, 'avgDA.nc')
    models2include = []
    for model in models:
        print(f'checking whether to include {model}')
        include_model = True
        resultdir_model = join(resultdir,model)
        figdir_model = join(figdir,model)
        makedirs(resultdir_model, exist_ok=True)
        makedirs(figdir_model, exist_ok=True)
        stagefiles[model] = dict()
        for i_fcdate,fcdate in enumerate(qoidict['fcdates']):
            stagefiles[model][fcdate] = dict()
            ens_sizes = []
            for expt in qoidict['expts']:
                mem_filenames,mem_labels = datre.get_gcm_6hrPt_filenames(model,qoidict['var_name'],expt,fcdate)
                stagefiles[model][fcdate][expt] = dict({label: filename for (label,filename) in zip(mem_labels,mem_filenames)})
                ens_sizes.append(len(mem_filenames))
            include_model *= (len(np.unique(ens_sizes)) == 1) * (ens_sizes[0] > 0)
        if include_model:
            print(f'yes')
            models2include.append(model)
            stagefiles[model]['avgD'] = join(resultdir_model, 'avgD.nc')
            stagefiles[model]['avgDA'] = join(resultdir_model, 'avgDA.nc')
            stagefiles[model]['avgDA_plot'] = join(figdir_model, 'avgDA_plot.png')
            stagefiles[model]['severity'] = dict()
            stagefiles[model]['abs_risk'] = dict()
            stagefiles[model]['abs_risk_plot'] = dict()
            stagefiles[model]['rel_risk'] = dict()
            stagefiles[model]['rel_risk_plot'] = dict()
            for svmetric in list(qoidict['severity_metrics_families'].keys()):
                stagefiles[model]['severity'][svmetric] = join(resultdir_model, f'severity_{svmetric}.nc')
                stagefiles[model]['abs_risk'][svmetric] = dict()
                stagefiles[model]['abs_risk_plot'][svmetric] = dict()
                stagefiles[model]['rel_risk'][svmetric] = dict()
                stagefiles[model]['rel_risk_plot'][svmetric] = dict()
                for family in qoidict['severity_metrics_families'][svmetric]: # TODO metrics and stat models should not be orthogonal 
                    stagefiles[model]['abs_risk'][svmetric][family] = join(resultdir_model, f'risk_{svmetric}_{family}.nc')
                    stagefiles[model]['abs_risk_plot'][svmetric][family] = join(figdir_model, f'risk_{svmetric}_{family}_plot.png')
                    stagefiles[model]['rel_risk'][svmetric][family] = join(resultdir_model, f'rel_risk_{svmetric}_{family}.nc')
                    stagefiles[model]['rel_risk_plot'][svmetric][family] = join(figdir_model, f'rel_risk_{svmetric}_{family}_plot.png')
        else:
            print(f'no')


    stagefiles['riskcomp'] = dict()
    stagefiles['riskcomp_plot'] = dict()
    for svmetric in list(qoidict['severity_metrics_families'].keys()):
        compdir_svmetric = join(resultdir,f'riskcomp_{svmetric}')
        compdir_svmetric_plot = join(figdir,f'riskcomp_{svmetric}')
        makedirs(compdir_svmetric, exist_ok=True)
        makedirs(compdir_svmetric_plot, exist_ok=True)
        stagefiles['riskcomp'][svmetric] = dict()
        stagefiles['riskcomp_plot'][svmetric] = dict()
        for family in qoidict['severity_metrics_families'][svmetric]:
            stagefiles['riskcomp'][svmetric][family] = join(compdir_svmetric, f'{family}.nc')
            stagefiles['riskcomp_plot'][svmetric][family] = join(compdir_svmetric_plot, f'{family}_plot.png')
    print(f'done')
    
    
        
    # ------------ Reduce climatology and ERA5 --------------
    print(f"About to reduce climatology")
    if find_true_in_dict(tododict['clim']):
        reduce_clim(qoidict, tododict['clim'], stagefiles['clim'])
    if find_true_in_dict(tododict['era5']):
        risk_calc_pipeline_era5(qoidict, tododict['era5'], stagefiles['era5'])
    # --------------------------------------------------
    
    # ------------ Reduce models  one at a time ---------------
    if find_true_in_dict(tododict['gcms']):
        for model in models2include:
            print(f"About to start pipeline for {model}")
            if find_true_in_dict(tododict['gcms'][model]):
                risk_calc_pipeline_1model(qoidict, tododict['gcms'][model], stagefiles[model], stagefiles['era5'], stagefiles['clim'])
    # ----------------------------------------------------------

    # ------------ Compare aspects of risk across models ----------
    for svmetric in list(qoidict['severity_metrics_families'].keys()):
        family = 'gev'
        # Collect shape parameters (point estimate and CI) across all models
        if tododict['riskcomp']['compare_params']:
            paramset = xr.DataArray(
                    coords={'model': models2include, 'init': qoidict['fcdates'], 'expt': qoidict['expts'], 'param_name': stfu.param_names(family), 'boot': np.arange(qoidict['n_boot']+1)},
                    dims=['model','init','expt','param_name','boot'],
                    data=np.nan)
            for model in models2include:
                abs_risk = xr.open_dataset(stagefiles[model]['abs_risk'][svmetric][family])
                paramset.loc[dict(model=model)] = abs_risk['params']
            paramset.to_netcdf(stagefiles['riskcomp'][svmetric][family])
            print(f'{paramset.sel(boot=0) = }')
        if tododict['riskcomp']['plot_params']:
            paramset = xr.open_dataarray(stagefiles['riskcomp'][svmetric][family])
            nrows = paramset['param_name'].size
            ncols = paramset['init'].size
            fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 6*nrows), sharex='col', sharey='row')
            offsets_expt = {'control': -0.25, 'free': 0.0, 'nudged': 0.25}
            handles = dict()
            for i_param_name,param_name in enumerate(paramset['param_name'].values):
                for i_init,init in enumerate(qoidict['fcdates']):
                    ax = axes[i_param_name,i_init]
                    for i_model,model in enumerate(models2include):
                        if i_model < len(models2include)-1:
                            ax.axvline(i_model+0.5,color='gray')
                        for i_expt,expt in enumerate(paramset['expt'].values):
                            exptsel = dict(model=model,param_name=param_name,init=init,expt=expt)
                            xdata = i_model+offsets_expt[expt]
                            ydata = paramset.sel(exptsel).sel(boot=0).item()
                            handles[expt] = ax.scatter(xdata, ydata, c=qoidict['dispprop']['expts'][expt]['color'],marker='o',label=expt)
                            ciwidths2plot = [0.5,0.95]
                            for i_ciwidth,ciwidth in enumerate(ciwidths2plot):
                                qs = [0.5-0.5*ciwidth,0.5+0.5*ciwidth]
                                linewidth = 2*(len(ciwidths2plot) - i_ciwidth)
                                ax.plot((i_model+offsets_expt[expt])*np.ones(2), [paramset.sel(exptsel).quantile(q,dim='boot') for q in qs], color=qoidict['dispprop']['expts'][expt]['color'], linewidth=linewidth)
                    ax.set_title(r'Init %s'%(init.strftime('%Y-%m-%d')))
                    ax.set_xlabel('')
                    ax.set_ylabel(param_name)
                    ax.set_xticks(np.arange(len(models2include)))
                    ax.set_xticklabels(models2include)
                    ax.tick_params(axis='x', rotation=90)
                    if family == 'gev' and param_name == 'shape':
                        ax.axhline(0, color='black', linestyle='--')
            axes[0,0].legend(handles=list(handles.values()), ncol=3, loc=(0,1.1))
            fig.savefig(stagefiles['riskcomp_plot'][svmetric][family], **pltsvargs)
            plt.close(fig)




    # -------------------------------------------------------------

    return

if __name__ == "__main__":
    main()
    



