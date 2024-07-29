import numpy as np
from scipy.optimize import minimize
from modules_fitting_gen import *
from mpmath import lerchphi
import string
import sys,os
import mpmath as mpm

## normalization
def lerch_minmax(gamma,b,kmin,kmax):
    '''sum[exp(-bk)*(k)**(-gamma),{k,kmin,kmax}]
    '''
    if kmax == None:
        C = np.exp(-b*kmin)*lerchphi(np.exp(-b),gamma,kmin)
    else:
        C = np.exp(-b*kmin)*lerchphi(np.exp(-b),gamma,kmin) \
          - np.exp(-b*(kmax*1.0))*lerchphi(np.exp(-b),gamma,kmax+1.0)
    return float(C.real)
    
def rgf_minmax(kmin,kmax,gamma,b):
    '''returns the sum( k**(-gamma)*mpm.exp(-b/k),{k,kmin,kmax}  ) with kmax=None for infty
    '''
    mpm.mp.dps = 15
    if kmax == None:
        S = (float(mpm.sumem(lambda k: k**(-gamma)*mpm.exp(-b*k),[kmin,mpm.inf])))
    else:
        S = (float(mpm.sumem(lambda k: k**(-gamma)*mpm.exp(-b*k),[kmin,kmax])))
    return S

def rgf_minmax_log(kmin,kmax,gamma,b):
    '''
    '''
    mpm.mp.dps = 15
    if kmax == None:
        x = mpm.log(mpm.sumem(lambda k: k**(-gamma)*mpm.exp(-b*k),[kmin,mpm.inf]))
        if mpm.im(x) != 0:
            logS = float(mpm.fabs(x))
        else:
            logS = float(x)
    else:
        logS = (float(mpm.log(mpm.sumem(lambda k: k**(-gamma)*mpm.exp(-b*k),[kmin,kmax]))))
    return logS    
    
def pdf_rgf_disc(x,kmin,kmax,gamma,b):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
    '''
    
#     c = 1.0/lerch_minmax(gamma,b,kmin,kmax)
#     p = c*np.exp(-b*x)*x**(-gamma)
    p = 1.0/rgf_minmax(kmin,kmax,gamma,b)*x**(-gamma)*np.exp(-b*x)

    return p
    
def cdf_rgf_disc(x,kmin,kmax,gamma,b):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
       use ONLY for small arrays, since lerchphi from mpmath does not cooperate
       with numpy arrays!!! --> call pdf --> call cumsum  
    '''
#     c = 1.0/lerch_minmax(gamma,b,kmin,kmax)
#     c1=float(lerchphi(np.exp(-b),gamma,kmin))
#     y = float(lerchphi(np.exp(-b),gamma,x+1.0))
#     F = c*(np.exp(-b*kmin)*c1 - np.exp(-b*(x+1))*y)
    Fk = rgf_minmax(kmin,x,gamma,b)/rgf_minmax(kmin,kmax,gamma,b)

    return F
    
def cdf_rgf_disc_array(x,kmin,kmax,gamma,b):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
       use ONLY for small arrays, since lerchphi from mpmath does not cooperate
       with numpy arrays!!! --> call pdf --> call cumsum  
    '''
    c = 1.0/lerch_minmax(gamma,b,kmin,kmax)
    c1=float(lerchphi(np.exp(-b),gamma,kmin))
    N = len(x)
    y = HurwitzLerchMath(x+1,gamma,b)
    F = c*(np.exp(-b*kmin)*c1 - np.exp(-b*(x+1))*y)
    return F
    
def mle_disc_minmax_optimize_rgfzipf(x,kmin,kmax,gamma_0,b_0,method = 'Nelder-Mead'):
    N = float(len(x)) #types
    M = float(sum(x)) #tokens
    x = np.sort(x)[::-1]
    r = np.arange(N)+1
    
    D_rN = sum((x/M)*r)
    D_lnrN = sum((x/M)*np.log(r))
    
    # gamma_opt = fmin(func_mle_disc_rgfzipf, [gamma_0,b_0], args=(D_rN,D_lnrN,kmin,kmax),xtol=0.0001,ftol=0.0001,disp=0,full_output=1)
    result = minimize(
        func_mle_disc_rgfzipf, 
        [gamma_0,b_0], 
        args=(D_rN,D_lnrN,kmin,kmax),
        # bounds = ((1.0,None),(1.,None)),
        options = {'disp':False},
        method=method)
    warnflag = result['success']
    if warnflag != True:
        print('No convergence in maximizing likelihood!')
    return result

def func_mle_disc_rgfzipf(params,D_rN,D_lnrN,kmin,kmax):
    gamma = params[0]
    b = params[1]
#     S = lerch_minmax(gamma,b,kmin,kmax)
    logS = rgf_minmax_log(kmin,kmax,gamma,b)
    L = logS + b*D_rN + gamma*D_lnrN
    return L

