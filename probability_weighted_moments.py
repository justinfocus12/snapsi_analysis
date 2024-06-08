
# Subroutines for statistical calculations
import numpy as np
from scipy.stats import norm as spnorm, genextreme as spgex, genpareto as spgpd, uniform as spunif
from scipy.special import gamma as GammaFunction, logsumexp
from scipy.optimize import bisect
from numpy.random import default_rng
import matplotlib.pyplot as plt

# Binomial (essentially non-parametric) models

def fit_statistical_model(S, family, thresh=None, n_boot=0, rng=None, rngseed=None, method=None):
    # Given a dataset of severities S = [S_0, S_1, ..., S_{n-1}], return a dictionary of parameters (like mean and variance for a normal, or location, shape and scale for a GEV) that fits the data. 
    # Additionally, return a bootstrap distribution of fits. 
    if rng is None:
        if rngseed is None:
            rngseed = 2718
        rng = default_rng(rngseed)
    n = len(S)
    Sboot = np.zeros((n_boot+1, n)) # first row is entire sample; subsequent rows are bootstrap resamplings
    Sboot[0,:] = S[np.arange(n)]
    Sboot[1:,:] = S[rng.choice(np.arange(n), replace=True, size=(n_boot,n))]

    if family == 'normal':
        params = dict({'mean': np.mean(Sboot, axis=1), 'stddev': np.std(Sboot, axis=1)})
    elif family == 'gev':
        func = lambda X: fit_gev_single(X, method)
        # Note, we use the convention that positive shape parameter means unbounded tail --- the opposite of genextreme
        gevpar = np.apply_along_axis(func, 1, Sboot)
        params = dict({'shape': gevpar[:,0], 'loc': gevpar[:,1], 'scale': gevpar[:,2]})
    elif family == 'gpd':
        # TODO encode the choice of thresh somehow 
        assert(thresh is not None)
        func = lambda s: spgpd.fit(s[s > thresh]-thresh, method="MLE", floc=0)
        gpdpar = np.apply_along_axis(func, 1, Sboot)
        params = dict({'shape': gpdpar[:,0], 'loc': gpdpar[:,1], 'scale': gpdpar[:,2], 'base_level': thresh*np.ones(n_boot+1), 'base_prob': np.mean(Sboot>thresh, axis=1)})
    #params.update(family=family)
    # TODO add confidence intervals via delta method or bootstrap
    return params

def fit_gev_single(X, method):
    if method == 'MLE':
        gevpar = np.array(spgex.fit(X, method='MLE'))
        gevpar[0] *= -1 # We use the standard convention that negative shape parameter means bounded tail 
    elif method == 'PWM':
        # Based on Hosking 1985, "Estimation of the Generalized Extreme-Value Distribution by the Method of Probability-Weighted Moments", Technometrics, Vol. 27, No. 3
        N = len(X)
        order = np.argsort(X)
        # Shift lower boundary of data to be positive ---- otherwise it doesn't work!
        offset = 1e-6 - X[order[0]] 
        Xord = X[order] + offset
        logwnorm = -np.log(N)*np.ones(N) # low weights on data (all the same when data is uniformly weighted)
        logWord = logwnorm[order]
        logFord = np.logaddexp.accumulate(logWord) # log-empirical CDF
        b0 = np.exp(logsumexp(logWord, b=Xord)) # equivalent to np.exp(np.sum(Word * Xord))
        b1 = np.exp(logsumexp(logWord + logFord, b=Xord)) # equivalent to np.exp(np.sum(Word * Xord * Ford))
        b2 = np.exp(logsumexp(logWord + 2*logFord, b=Xord)) # equivalent to np.exp(np.sum(Word * Xord * Ford**2))
        # Solve for the shape, location, and scale parameters. Don't use the linear approximation
        b_ratio = (3*b2 - b0)/(2*b1 - b0)
        if b_ratio <= 0.0:
            # TODO come up with the best possible alternative...xi is a very large number, probably 
            raise Exception(f"The PWM method has no solution; {b_ratio = }")
        # Choose initialization for bisection method 
        tol = 1e-2
        psf0 = pwm_shape_func(0,b_ratio) 
        if psf0 == 0:
            shape = 0.0
        elif psf0 < 0: # shape > 0
            lower = 0.0
            upper = 1.0
            while pwm_shape_func(upper,b_ratio) < 0.0:
                upper *= 2.0
        else: # shape < 0
            lower = -1.0
            upper = 0.0
            while pwm_shape_func(lower,b_ratio) > 0.0:
                lower *= 2.0
        shape,root_result = bisect(pwm_shape_func, lower, upper, args=(b_ratio,), full_output=True, disp=True)

        g = GammaFunction(1 - shape)
        if shape == 0:
            scale = (2*b1 - b0)/np.log(2)
            loc = b0 - 0.5772*scale
        else:
            scale = shape*(2*b1 - b0)/((2**shape-1) * g)
            loc = b0 + scale*(1 - g)/shape
        gevpar = np.array([shape,loc-offset,scale])
    print(f'{gevpar = }')
    return gevpar

def pwm_shape_func(shape,b_ratio): # The function to solve: (3**shape-1)/(2**shape-1) - (3*b2-b0)/(2*b1-b0)
    if np.abs(shape) < 1e-6:
        # Use local quadratic approximation near zero
        return np.log(3)/np.log(2)*(1 + np.log(3/2)/2*shape - np.log(6)/4*shape**2) - b_ratio
    return (3**shape - 1)/(2**shape - 1) - b_ratio

def gev_cdf(xin, shape, loc, scale):
    issc = np.isscalar(xin)
    x = np.array([xin]) if issc else xin
    if shape == 0:
        F = np.exp(-np.exp(-(x-loc)/scale))
    else:
        F = np.zeros_like(x)
        arg = 1 + shape*(x - loc)/scale
        idx_pos = np.where(arg > 0)[0]
        if len(idx_pos) > 0:
            F[idx_pos] = np.exp(-arg[idx_pos]**(-1/shape))
        if len(idx_pos) < len(x):
            F[np.setdiff1d(range(len(x)),idx_pos)] = 1.0 if shape < 0 else 0.0
    if issc: F = F[0]
    return F

def gev_lsf(xin, shape, loc, scale):
    # log-survival function
    issc = np.isscalar(xin)
    x = np.array([xin]) if issc else xin

    if shape == 0:
        F = np.log(-np.expm1(-np.exp(-(x-loc)/scale)))
    else:
        F = np.zeros_like(x)
        arg = 1 + shape*(x - loc)/scale
        idx_pos = np.where(arg > 0)[0]
        if len(idx_pos) > 0:
            F[idx_pos] = np.log(-np.expm1(-arg[idx_pos]**(-1/shape)))
        if len(idx_pos) < len(x):
            F[np.setdiff1d(range(len(x)),idx_pos)] = 1.0 if shape < 0 else 0.0
    if issc: F = F[0]
    return F
    

def gev_invcdf(u, shape, loc, scale):
    # inverse CDF
    if shape == 0:
        x = loc - scale * np.log(-np.log(u))
    else:
        x = loc + scale/shape * (np.power(-np.log(u), -shape) - 1)
    return x

def gev_invlsf(u, shape, loc, scale):
    # = gev_invcdf(1 - exp(u))
    return gev_invcdf(-np.expm1(u), shape, loc, scale)

def pwm_unit_test(shape, loc, scale, seed=48732, nsamp_max=1000, ntrials_per_samp=10):

    rng = default_rng(seed)

    # True parameters
    param_symbols = [r"$\xi$",r"$\mu$",r"$\sigma$"]
    param_names = ["shape", "loc", "scale"]
    gevpar = dict(shape=shape, loc=loc, scale=scale)
    
    nsamp_list = np.exp(np.linspace(np.log(20), np.log(nsamp_max), 5)).astype(int)
    gevpar_hat = np.zeros((len(nsamp_list),ntrials_per_samp,len(param_names)))
    for i_nsamp,nsamp in enumerate(nsamp_list):
        print(f"Starting sample size {nsamp}")
        for trial in range(ntrials_per_samp):
            U = rng.uniform(size=nsamp)
            X = gev_invcdf(U, shape, loc, scale)
            W = np.ones(nsamp)
            params_fitted = fit_statistical_model(X, "gev", n_boot=1, method="PWM")
            gevpar_hat[i_nsamp,trial,:] = [params_fitted[param][0] for (i_param,param) in enumerate(param_names)]

    fig,axes = plt.subplots(nrows=3, figsize=(10,15), sharex=True, gridspec_kw={"hspace": 0.25})
    for i_param,param in enumerate(["shape","loc","scale"]):
        ax = axes[i_param]
        htrue = ax.axhline(gevpar[param], color="black", linestyle="--", linewidth=3, label="Truth")
        for i_nsamp,nsamp in enumerate(nsamp_list): 
            ax.scatter(nsamp*np.ones(ntrials_per_samp), gevpar_hat[i_nsamp,:,i_param], color="red", marker=".")
        hpwm, = ax.plot(nsamp_list,np.mean(gevpar_hat[:,:,i_param],axis=1), color="red", label="PWM")
        ax.set_xscale("log")
        ax.set_xlabel("Sample size")
        ax.xaxis.set_tick_params(which="both",labelbottom=True)
        ax.set_ylabel(param_symbols[i_param])
        ax.set_title(param_names[i_param])
    fig.savefig((f"gev_fit_shp{gevpar['shape']}_loc{gevpar['shape']}_scale{gevpar['scale']}").replace(".","p"),bbox_inches="tight",pad_inches=0.2)
    plt.close(fig)
    return

if __name__ == "__main__":
    loc = 1.6
    scale = 3.14
    for shape in [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]:
        pwm_unit_test(shape, loc, scale, nsamp_max=10000, ntrials_per_samp=20)
