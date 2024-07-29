import numpy as np
from scipy.optimize import minimize
# from modules_fitting_gen import *
import mpmath as mpm
from scipy.special import erfc
# import pylab as pl

## lognormal pk = C*1/k*exp(-0.5*((lnk - mu)/sigma)^2)
#
#Normalization
#
def lognormal_minmax(kmin,kmax,mu,sigma):
    '''returns the sum( 1/sqrt(2*pi*sigma^2)/k*exp(-0.5*((lnk - mu)/sigma)^2),{k,kmin,kmax}  ) with kmax=None for infty
    '''
    mpm.mp.dps = 15
    if kmax == None:
        S = (float(mpm.sumem(lambda k: 1.0/(k-kmin+1.0)*mpm.exp(-0.5*(mpm.log((k-kmin+1.0))-mu)*(mpm.log((k-kmin+1.0))-mu)/sigma/sigma),[kmin,mpm.inf])))
    else:
        S = (float(mpm.sumem(lambda k: 1.0/(k-kmin+1.0)*mpm.exp(-0.5*(mpm.log((k-kmin+1.0))-mu)*(mpm.log((k-kmin+1.0))-mu)/sigma/sigma),[kmin,kmax])))
    return S

def lognormal_minmax_log(kmin,kmax,mu,sigma):
    '''returns the sum( 1/k*exp(-0.5*((lnk - mu)/sigma)^2),{k,kmin,kmax}  ) with kmax=None for infty
    '''
#     print kmin,kmax,mu,sigma
    mpm.mp.dps = 15
    if kmax == None:
        x = mpm.log(mpm.sumem(lambda k: 1.0/(k-kmin+1.0)*mpm.exp(-0.5*(mpm.log((k-kmin+1.0))-mu)*(mpm.log((k-kmin+1.0))-mu)/sigma/sigma),[kmin,mpm.inf]))
        if mpm.im(x) != 0:
            logS = float(mpm.fabs(x))
        else:
            logS = float(x)
    else:
        logS = (float(mpm.log(mpm.sumem(lambda k: 1.0/(k-kmin+1.0)*mpm.exp(-0.5*(mpm.log((k-kmin+1.0))-mu)*(mpm.log((k-kmin+1.0))-mu)/sigma/sigma),[kmin,kmax]))))

    return logS

#
# pdf + cdf
#
def pdf_lognormal_disc(x,kmin,kmax,mu,sigma):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
    '''
    pk = 1.0/lognormal_minmax(kmin,kmax,mu,sigma)/(x-kmin+1.0)*np.exp(-0.5*(np.log((x-kmin+1.0))-mu)*(np.log((x-kmin+1.0))-mu)/sigma/sigma)
    return pk
    
def cdf_lognormal_disc(x,kmin,kmax,mu,sigma):
    '''
    '''
    Fk = lognormal_minmax(kmin,x,mu,sigma)/lognormal_minmax(kmin,kmax,mu,sigma)
    return Fk
    
#
# optimization
#   
def mle_disc_minmax_optimize_lognormalzipf(x,kmin,kmax,mu_0,sigma_0,method = 'Nelder-Mead'):
    N = float(len(x)) #types
    M = float(sum(x)) #tokens
    x = np.sort(x)[::-1]
    rlog = np.log(np.arange(N)+1)

    D1N = sum((x/M)*rlog)
    D2N = sum((x/M)*rlog*rlog)
    # gamma_opt = fmin(func_mle_disc_lognormalzipf, [mu_0,sigma_0], args=(D1N,D2N,kmin,kmax),xtol=0.0001,ftol=0.0001,disp=0,full_output=1)
    result = minimize(
        func_mle_disc_lognormalzipf, 
        [mu_0,sigma_0], 
        args=(D1N,D2N,kmin,kmax),
        # bounds = ((1.0,None),(1.,None)),
        options = {'disp':False},
        method=method)
    # convergence message 0...good,1...bad
    # warnflag = result[4]
    warnflag = result['success']
    if warnflag != True:
        print('No convergence in maximizing likelihood!')
    return result

def func_mle_disc_lognormalzipf(params,D1N,D2N,kmin,kmax):
    mu = params[0]
    sigma = np.abs(params[1])
    logS = lognormal_minmax_log(kmin,kmax,mu,sigma)
    L = logS + 0.5*mu*mu/sigma/sigma + (1.0 - mu/sigma/sigma)*D1N + 0.5/sigma/sigma*D2N
    return L