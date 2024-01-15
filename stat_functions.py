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
    elif family == 'bernoulli':
        assert(thresh is not None)
        params = dict({'mean': np.mean(Sboot > thresh, axis=1)})
    params.update(family=family)
    # TODO add confidence intervals via delta method or bootstrap
    return params

def param_names(family):
    if family == 'normal':
        pn = ['mean','stddev'] 
    elif family == 'gev':
        pn = ['shape','location','scale']
    elif family == 'bernoulli':
        pn = ['mean']
    return pn

def absolute_risk(family, params, thresh):
    # also returns an array with bootstraps 
    if family == 'normal':
        p = spnorm.sf(thresh, loc=params['mean'], scale=params['stddev'])
    elif family == 'gev':
        p = spgex.sf(thresh, params['shape'], loc=params['location'], scale=params['scale'])
    elif family == 'bernoulli': 
        p = params['mean']*np.ones_like(thresh)
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

