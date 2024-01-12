import numpy as np
from scipy.stats import norm as spnorm


def wilson_score_interval(nsucc, nfail, confint):
    # compute the binomial confidence interval on p given nsucc successes and nfail failures
    z = spnorm.ppf(0.5 + 0.5*confint)
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

