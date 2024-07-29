import numpy as np
from scipy.optimize import minimize
from modules_fitting_gen import *

## shifted power law: pk(gamma,b) = (k+b)**(-gamma)

def pdf_spl_disc(x,kmin,kmax,gamma,b):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
    '''
    if kmax == None:
        C = ((x+b)**(-gamma))/zeta_minmax(gamma,kmin+b,kmax)
    else:
        C = ((x+b)**(-gamma))/zeta_minmax(gamma,kmin+b,kmax+b+1)
    return C
    
def cdf_spl_disc(x,kmin,kmax,gamma,b):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
    '''
    if kmax == None:
        C = zeta_minmax(gamma,kmin+b,x+b+1)/zeta_minmax(gamma,kmin+b,kmax)
    else:
        C = zeta_minmax(gamma,kmin+b,x+b+1)/zeta_minmax(gamma,kmin+b,kmax+b+1)
    return C
    
def mle_disc_minmax_optimize_splzipf(x,kmin,kmax,gamma_0,b_0,method = 'Nelder-Mead'):
    N = float(len(x)) #types
    M = float(sum(x)) #tokens
    x = np.sort(x)[::-1]
    # gamma_opt = fmin(func_mle_disc_splzipf, [gamma_0,b_0], args=(x,N,M,kmin,kmax),xtol=0.0001,ftol=0.0001,disp=0,full_output=1)
    result = minimize(
        func_mle_disc_splzipf, 
        [gamma_0,b_0], 
        args=(x,N,M,kmin,kmax),
        # bounds = ((1.0,None),(1.,None)),
        options = {'disp':False},
        method=method)
    warnflag = result['success']
    if warnflag != True:
        print('No convergence in maximizing likelihood!')
    return result

def func_mle_disc_splzipf(params,x,N,M,kmin,kmax):
    gamma = params[0]
    if kmax==None:
        eps=10.0**(-6)
        if gamma <= 1.:
            gamma=1. + eps
    b = params[1]
    if kmax == None:
        C = zeta_minmax(gamma,kmin+b,kmax)
    else:
        C = zeta_minmax(gamma,kmin+b,kmax+b+1)
    D = sum(x/M*np.log((np.arange(N)+1)+b))
    L = np.log(C) + gamma*D
    return L
