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
    if tododict['avgD'] or tododict['avgDA']:
        print(f"...reducing")
        reduce_gcm(qoidict, stagefiles_model, tododict)
    if tododict['plot_avgDA']:
        print(f"...plotting")
        plot_avgDA_timeseries(qoidict, stagefiles_model, stagefiles_era5, stagefiles_clim)
    avgDA_model = xr.open_dataarray(stagefiles_model['avgDA']) # Do we need to further subset by the variable? Probably not because didn't subtract off ERA5
    avgDA_era5 = xr.open_dataarray(stagefiles_era5['avgDA']) # Do we need to further subset by the variable? Probably not because didn't subtract off ERA5
    for svmetric in list(qoidict['severity_metrics_families'].keys()):
        print(f'{svmetric = }')
        if tododict['compute_severity']:
            severity_model = severity_fun_avgDA(avgDA_model.sel(qoidict['timesel_event']), svmetric) # dims: (fcdate,expt,member,svmetric)
            severity_model.to_netcdf(stagefiles_model['severity'][svmetric])
        severity_model = xr.open_dataarray(stagefiles_model['severity'][svmetric])
        severity_era5 = severity_fun_avgDA(avgDA_era5.sel(qoidict['timesel_event']), svmetric)
        thresh_bounds = np.array([min(severity_model.min().item(),severity_era5.item()), max(severity_model.max().item(),severity_era5.item())])
        elongation = 0.5
        thresh_bounds[1] = (1+elongation)*thresh_bounds[1] - elongation*thresh_bounds[0]
        thresh_list = np.linspace(thresh_bounds[0], thresh_bounds[1], 30)
        for family in qoidict['severity_metrics_families'][svmetric]:
            if tododict['compute_risk']:
                # Absolute risks
                n_boot = 10
                rngseed = 987405
                # TODO specialize how uncertainty is specified depending on the statistical family
                ar = xr.DataArray(
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
                        print(f'{expt = }')
                        S = severity_model.sel(init=init, expt=expt).to_numpy()
                        print(f'{S = }')
                        theta = stfu.fit_statistical_model(S, family, thresh=severity_era5.item(), n_boot=n_boot, rngseed=rngseed)
                        print(f'{theta = }')
                        ar.loc[dict(init=init, expt=expt)] = stfu.absolute_risk_parametric(family, theta, thresh_list)
                        for pn in params.param_name.to_numpy():
                            params.loc[dict(init=init, expt=expt, param_name=pn)] = theta[pn]

                abs_risk = xr.Dataset(
                        data_vars={'abs_risk': ar, 'params': params})
                abs_risk.attrs = {'severity_era5': severity_era5.item()}
                abs_risk.to_netcdf(stagefiles_model['abs_risk'][svmetric][family])
            abs_risk = xr.open_dataset(stagefiles_model['abs_risk'][svmetric][family])
            if tododict['plot_risk']:
                # Compute return periods and plot 
                fig = plt.figure(constrained_layout=True)
                fig,axes = plt.subplots(ncols=2, nrows=len(qoidict['expts']), figsize=(12,6*len(qoidict['expts'])), sharey=True, sharex='col')
                # Left: histograms in vertical
                # Right: return period curves
                for i_expt,expt in enumerate(qoidict['expts']):
                    handles = []
                    for i_init,init in enumerate(qoidict['fcdates']):
                        S = severity_model.sel(init=init, expt=expt).to_numpy()[np.newaxis, :]
                        hist,bin_edges = np.histogram(severity_model.sel(expt=expt,init=init), bins=5, density=True)
                        bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
                        ax.plot(hist, bin_centers, marker='.', **qoidict['dispprop']['fcdates'][init])
                        ax.xaxis.set_tick_params(which='both',labelbottom=True)
                        ax.set_ylabel(qoidict['dispprop']['severity_metrics'][svmetric]['label'])
                        ax.set_title(f'{expt} histogram')

                        ax = axes[i_expt,1]
                        return_period_parametric = 1.0/abs_risk['abs_risk'].sel(expt=expt,init=init).where(abs_risk['abs_risk'].sel(expt=expt,init=init)>1e-10)
                        return_period_empirical = stfu.absolute_risk_empirical(S)
                        # TODO plot empirical
                        hreturn, = ax.plot(return_period_parametric, thresh_list, marker='.', **qoidict['dispprop']['fcdates'][init])
                        handles.append(hreturn)
                        ax.set_xscale('log')
                        ax.xaxis.set_tick_params(which='both',labelbottom=True)
                        ax.yaxis.set_tick_params(which='both',labelbottom=True)
                        ax.set_title(f'{expt} return levels')
                    for ax in axes[i_expt,:]:
                        hera5 = ax.axhline(severity_era5.item(), color='black', linestyle='--', label='ERA5')
                handles.append(hera5)
                axes[0,1].legend(handles=handles,loc='lower right')
                axes[-1,0].set_xlabel('Probability density')
                axes[-1,1].set_xlabel('Return period [years]')
                fig.suptitle(f'Absolute risk via {family}')
                fig.savefig(stagefiles_model['abs_risk_plot'][svmetric][family], **pltsvargs)
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
models = ['IFS'] #list(model2institute.keys())

qoidict = dict({
    'fcdates': np.array([datetime.datetime(2018,1,25),datetime.datetime(2018,2,8)]),
    'expts': ['control','free','nudged'],
    'expt_pairs': dict({"n2f": ["nudged","free"], "f2c": ["free","control"], "n2c": ["nudged","control"]}),
    'timesel_maxposs': dict(time=slice(datetime.datetime(2018,2,12),datetime.datetime(2018,3,11))),
    'timesel_event': dict(time=slice(datetime.datetime(2018,2,21),datetime.datetime(2018,3,8))),
    'spacesel': dict(lat=slice(50,65),lon=slice(-10,130)),
    'var_name': 't2m', # which variable we're interested in the extrema of 
    'severity_metrics_families': dict({
        'mintemp': ['normal','gev'],
        'meantemp': ['normal','gev'],
        }),
    })
# Display properties
qoidict['dispprop'] = dict({
    'fcdates': dict({
        qoidict['fcdates'][0]: dict({
            'color': 'red',
            'label': r'Init %s'%(qoidict['fcdates'][0].strftime('%Y-%m-%d')),
            }),
        qoidict['fcdates'][1]: dict({
            'color': 'blue',
            'label': r'Init %s'%(qoidict['fcdates'][1].strftime('%Y-%m-%d')),
            }),
        }),
    'severity_metrics': dict({ # display properties
        'mintemp': dict({
            'label': r'$-\min_t\mathrm{mean}_{x,y}T$ [K]',
            }),
        'meantemp': dict({
            'label': r'$-\mathrm{min}_t\mathrm{mean}_{x,y}T$ [K]',
            }),
        }),
    })

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
            'compute_risk':     1,
            'plot_risk':        1,
            })
        for model in models
        }),
    'comp': dict({ # comparison or comprehensive
        'plot_avgDA': 1,
        'plot_rr':    1, 
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
            models2include.append(model)
            stagefiles[model]['avgD'] = join(resultdir_model, 'avgD.nc')
            stagefiles[model]['avgDA'] = join(resultdir_model, 'avgDA.nc')
            stagefiles[model]['avgDA_plot'] = join(figdir_model, 'avgDA_plot.png')
            stagefiles[model]['severity'] = dict()
            stagefiles[model]['abs_risk'] = dict()
            stagefiles[model]['abs_risk_plot'] = dict()
            for svmetric in list(qoidict['severity_metrics_families'].keys()):
                stagefiles[model]['severity'][svmetric] = join(resultdir_model, f'severity_{svmetric}.nc')
                stagefiles[model]['abs_risk'][svmetric] = dict()
                stagefiles[model]['abs_risk_plot'][svmetric] = dict()
                for family in qoidict['severity_metrics_families'][svmetric]: # TODO metrics and stat models should not be orthogonal 
                    stagefiles[model]['abs_risk'][svmetric][family] = join(resultdir_model, f'risk_{svmetric}_{family}.nc')
                    stagefiles[model]['abs_risk_plot'][svmetric][family] = join(figdir_model, f'risk_{svmetric}_{family}_plot.png')
    print(f'done')
    
    
        
    # ------------ Reduce climatology and ERA5 --------------
    print(f"About to reduce climatology")
    reduce_clim(qoidict, tododict['clim'], stagefiles['clim'])
    risk_calc_pipeline_era5(qoidict, tododict['era5'], stagefiles['era5'])
    # --------------------------------------------------
    
    for model in models2include:
        print(f"About to start pipeline for {model}")
        risk_calc_pipeline_1model(qoidict, tododict['gcms'][model], stagefiles[model], stagefiles['era5'], stagefiles['clim'])

    return

if __name__ == "__main__":
    main()
    



