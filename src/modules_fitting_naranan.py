import numpy as np
from scipy.optimize import minimize
from modules_fitting_gen import *
import mpmath as mpm


def naranan_minmax(kmin,kmax,gamma,b):
    '''returns the sum( k**(-gamma)*mpm.exp(-b/k),{k,kmin,kmax}  ) with kmax=None for infty
    '''
    mpm.mp.dps = 15
    if kmax == None:
        S = (float(mpm.sumem(lambda k: k**(-gamma)*mpm.exp(-b/k),[kmin,mpm.inf])))
    else:
        S = (float(mpm.sumem(lambda k: k**(-gamma)*mpm.exp(-b/k),[kmin,kmax])))
    return S

def naranan_minmax_log(kmin,kmax,gamma,b):
    '''
    '''
    mpm.mp.dps = 15
    if kmax == None:
        x = mpm.log(mpm.sumem(lambda k: k**(-gamma)*mpm.exp(-b/k),[kmin,mpm.inf]))
        if mpm.im(x) != 0:
            logS = float(mpm.fabs(x))
        else:
            logS = float(x)
    else:
        logS = (float(mpm.log(mpm.sumem(lambda k: k**(-gamma)*mpm.exp(-b/k),[kmin,kmax]))))
    return logS

def pdf_naranan_disc(x,kmin,kmax,gamma,b):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
    '''
    pk = 1.0/naranan_minmax(kmin,kmax,gamma,b)*x**(-gamma)*np.exp(-b/x)
    return pk
    
def cdf_naranan_disc(x,kmin,kmax,gamma,b):
    '''not for arrays!!!
    '''
    Fk = naranan_minmax(kmin,x,gamma,b)/naranan_minmax(kmin,kmax,gamma,b)
    return Fk
    
   
def mle_disc_minmax_optimize_narananzipf(x,kmin,kmax,gamma_0,b_0,method = 'Nelder-Mead'):
    N = float(len(x)) #types
    M = float(sum(x)) #tokens
    x = np.sort(x)[::-1]
    r = np.arange(N)+1
    
    D1N = sum(x*np.log(r))/M
    Dm1N = sum(x/r)/M
    # gamma_opt = fmin(func_mle_disc_narananzipf, [gamma_0,b_0], args=(D1N,Dm1N,kmin,kmax),xtol=0.0001,ftol=0.0001,disp=0,full_output=1)
    result = minimize(
        func_mle_disc_narananzipf, 
        [gamma_0,b_0], 
        args=(D1N,Dm1N,kmin,kmax),
        # bounds = ((1.0,None),(1.,None)),
        options = {'disp':False},
        method=method)
    warnflag = result['success']
    if warnflag != True:
        print('No convergence in maximizing likelihood!')
    return result

def func_mle_disc_narananzipf(params,D1N,Dm1N,kmin,kmax):
    gamma=params[0]
    if kmax==None:
        eps=10.0**(-6)
        if gamma <= 1.:
            gamma=1. + eps    
    b = np.abs(params[1])
    logS = naranan_minmax_log(kmin,kmax,gamma,b)
    L = logS + gamma*D1N +b*Dm1N
    return L
