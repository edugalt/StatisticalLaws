import numpy as np
from scipy.optimize import minimize
from modules_fitting_gen import *
import mpmath as mpm
# from pylab import *

def weibull_minmax(kmin,kmax,gamma,b):
    '''returns the sum( k^(gamma-1)*exp(-b*(k)^gamma),{k,kmin,kmax}  ) with kmax=None for infty
    '''
    gamma = float(gamma)
    b = float(b)
    mpm.mp.dps = 15
    if kmax == None:
        S = (float(mpm.sumem(lambda k: k**(gamma-1.0)*mpm.exp(-b*k**gamma),[kmin,mpm.inf])))
    else:
        S = (float(mpm.sumem(lambda k: k**(gamma-1.0)*mpm.exp(-b*k**gamma),[kmin,kmax])))
    return S

def weibull_minmax_log(kmin,kmax,gamma,b):
    '''returns the sum( k^(gamma-1)*exp(-b*(k)^gamma),{k,kmin,kmax}  ) with kmax=None for infty
    '''
    gamma = float(gamma)
    b = float(b)
    mpm.mp.dps = 15
    if kmax == None:
        x = mpm.sumem(lambda k: k**(gamma-1.0)*mpm.exp(-b*k**gamma),[kmin,mpm.inf])
        if mpm.im(x) != 0:
            logS = float(mpm.fabs(x))
        else:
            logS = float(x)
    else:
        logS = float(mpm.log(mpm.sumem(lambda k: k**(gamma-1.0)*mpm.exp(-b*k**gamma),[kmin,kmax])))
    return logS

def pdf_weibull_disc(x,kmin,kmax,gamma,b):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
    '''
    gamma = float(gamma)
    b = float(b)
    pk = 1.0/weibull_minmax(kmin,kmax,gamma,b)*x**(gamma - 1.0)*np.exp(-b*x**gamma)
    return pk
    
def cdf_weibull_disc(x,kmin,kmax,gamma,b):
    '''not for arrays!!!
    '''
    Fk = weibull_minmax(kmin,x,gamma,b)/weibull_minmax(kmin,kmax,gamma,b)
    return Fk
    
def mle_disc_minmax_optimize_weibull_zipf(x,kmin,kmax,gamma_0,b_0,method = 'Nelder-Mead'):
    N = float(len(x)) #types
    M = float(sum(x)) #tokens
    x = np.sort(x)[::-1]
    D1N = sum(x*np.log(np.arange(N)+1))/M
    # gamma_opt = fmin(func_mle_disc_weibull_zipf, [gamma_0,b_0], args=(x,N,M,D1N,kmin,kmax),xtol=0.0001,ftol=0.0001,disp=0,full_output=1)
    result = minimize(
        func_mle_disc_weibull_zipf,
         [gamma_0,b_0], 
         args=(x,N,M,D1N,kmin,kmax),
        # bounds = ((1.0,None),(1.,None)),
        options = {'disp':False},
        method=method)
    warnflag = result['success']
    if warnflag != True:
        print('No convergence in maximizing likelihood!')
    return result
    
def func_mle_disc_weibull_zipf(params,x,N,M,D1N,kmin,kmax):
    
    gamma = params[0]
#     b = np.abs(params[1])
    if gamma>1.0:
        gamma = 1.0
    b = np.abs(params[1])
    logS = weibull_minmax_log(kmin,kmax,gamma,b)
    L = logS  + (1.0-gamma)*D1N + b/M*sum(x*(np.arange(N)+1)**gamma)
#     print gamma,b
    return L
