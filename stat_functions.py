# Subroutines for statistical calculations
import numpy as np
from scipy.stats import norm as spnorm, genextreme as spgex, genpareto as spgpd, uniform as spunif
from numpy.random import default_rng

# Binomial (essentially non-parametric) models

def fit_statistical_model(S, family, thresh=None, n_boot=1, rng=None, rngseed=None):
    # Given a dataset of severities S = [S_0, S_1, ..., S_{n-1}], return a dictionary of parameters (like mean and variance for a normal, or location, shape and scale for a GEV) that fits the data. Maybe also likelihoods.
    # Additionally, return a bootstrap distribution of fits. 
    if rng is None:
        if rngseed is None:
            rngseed = 2718
        rng = default_rng(rngseed)
    n = len(S)
    Sboot = np.zeros((n_boot+1, n))
    Sboot[0,:] = S[np.arange(n)]
    Sboot[1:,:] = S[rng.choice(np.arange(n), replace=True, size=(n_boot,n))]

    if family == 'normal':
        params = dict({'mean': np.mean(Sboot, axis=1), 'stddev': np.std(Sboot, axis=1)})
    elif family == 'gev':
        # TODO probability-weighted moments, too
        gevpar = np.apply_along_axis(spgex.fit, 1, Sboot, method="MLE")
        params = dict({'shape': gevpar[:,0], 'location': gevpar[:,1], 'scale': gevpar[:,2]})
    elif family == 'gpd':
        # TODO encode the choice of thresh somehow 
        assert(thresh is not None)
        func = lambda s: spgpd.fit(s[s > thresh]-thresh, method="MLE", floc=0)
        gpdpar = np.apply_along_axis(func, 1, Sboot)
        params = dict({'shape': gpdpar[:,0], 'location': gpdpar[:,1], 'scale': gpdpar[:,2], 'base_level': thresh*np.ones(n_boot+1), 'base_prob': np.mean(Sboot>thresh, axis=1)})
    #params.update(family=family)
    # TODO add confidence intervals via delta method or bootstrap
    return params

def param_names(family):
    if family == 'normal':
        pn = ['mean','stddev'] 
    elif family == 'gev':
        pn = ['shape','location','scale']
    elif family == 'gpd':
        pn = ['shape','location','scale','base_level','base_prob']
    return pn

def absolute_risk_parametric(family, params, thresh):
    # Assume both params and thresh are 1D arrays
    parnames = param_names(family)
    print(f'{parnames = }')
    nboot = len(params[parnames[0]])
    nth = len(thresh)
    params_flat = dict({param_name: np.outer(params[param_name], np.ones(nth)).flatten() for param_name in parnames})
    thresh_flat = np.outer(np.ones(nboot), thresh).flatten()
    if family == 'normal':
        p = spnorm.sf(thresh_flat, loc=params_flat['mean'], scale=params_flat['stddev'])
    elif family == 'gev':
        p = spgex.sf(thresh_flat, params_flat['shape'], loc=params_flat['location'], scale=params_flat['scale'])
    elif family == 'gpd':
        p = np.nan*np.ones(nboot*nth)
        idx = np.where(thresh_flat > params_flat['base_level'])[0]
        print(f'{len(idx)/len(p) = }')
        p[idx] = spgpd.sf(thresh_flat[idx]-params_flat['base_level'][idx], params_flat['shape'][idx], loc=params_flat['location'][idx], scale=params_flat['scale'][idx]) * params_flat['base_prob'][idx]
        print(f'{p = }')
        print(f'{np.mean(np.isfinite(p)) = }')
    p = p.reshape((nboot,nth))
    return p

def absolute_risk_empirical(S, thresh):
    # S is nboot x ndata
    nboot,ndata = S.shape
    nth = len(thresh)
    S_flat = np.outer(S.flatten(), np.ones(nth)).reshape((nboot,ndata,nth))
    thresh_flat = np.outer(np.ones(nboot*ndata), thresh).reshape((nboot,ndata,nth))
    p = np.mean(S_flat > thresh_flat, axis=1)
    return p

def relative_risk(ar0, ar1):
    # ar0 and ar1 are absolute risks
    zidx0 = np.where(ar0==0)
    zidx01 = np.where((ar0==0) * (ar1==0))
    rr = ar0 / np.where(ar1>0, ar1, np.nan)
    rr[zidx0] = np.inf # positive / zero
    rr[zidx01] = np.nan # zero / zero 
    return rr

def confidence_interval_bootstrap(param, ciwidth):
    # Given many bootstrap resamplings, estimate the confidence interval
    qlo = np.quantile(param[1:], 0.5*(1-ciwidth))
    qhi = np.quantile(param[1:], 0.5*(1+ciwidth))
    ci = {'percentile': np.array([qlo,qhi]), 'basic': 2*param[0] - np.array([qhi,qlo])}
    return ci


def confidence_interval_wilson(nsucc, nfail, ciwidth):
    # compute the binomial confidence interval on p given nsucc successes and nfail failures
    z = spnorm.ppf(0.5 + 0.5*ciwidth)
    z2 = z*z
    n = nsucc + nfail
    phat = nsucc/n
    center = (phat + z2/(2*n))/(1 + z2/n)
    radius = z/(1+z2/n) * np.sqrt(phat*(1-phat)/n + z2/(4*n*n))
    lower = center - radius
    idx0 = np.where(nsucc==0)[0]
    #lower[idx0] = 3/n[idx0]
    upper = center + radius
    idx0 = np.where(nfail==0)[0]
    #upper[idx0] = 1-3/n[idx0]
    return lower,upper

