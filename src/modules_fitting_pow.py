import numpy as np
from scipy.optimize import minimize
from modules_fitting_gen import *

## power law
#disc
def pdf_power_disc(x,kmin,kmax,gamma):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
    '''
    if kmax == None:
        C = (x**(-gamma))/zeta_minmax(gamma,kmin,kmax)
    else:
        C = (x**(-gamma))/zeta_minmax(gamma,kmin,kmax+1)
    return C

def cdf_power_disc(x,kmin,kmax,gamma):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
    '''
    if kmax == None:
        C = zeta_minmax(gamma,kmin,x+1)/zeta_minmax(gamma,kmin,kmax)
    else:
        C = zeta_minmax(gamma,kmin,x+1)/zeta_minmax(gamma,kmin,kmax+1)
    return C

def mle_disc_minmax_optimize_powzipf(x,kmin,kmax,gamma_0,method = 'Nelder-Mead'):
    N = float(len(x)) #types
    M = float(sum(x)) #tokens
    x = np.sort(x)[::-1]
    r = np.arange(N)+1
    D = sum(x/M*np.log(r))
    # gamma_opt = fmin(func_mle_disc, gamma_0, args=(D,kmin,kmax),xtol=0.0000001,ftol=0.0000001,disp=0,full_output=1)
    result = minimize(
        func_mle_disc, 
        gamma_0, 
        args=(D,kmin,kmax),
        # bounds = ((1.0,None),(1.,None)),
        options = {'disp':False},
        method=method)
    warnflag = result['success']
    if warnflag != True:
        print('No convergence in maximizing likelihood!')
    return result
    
def func_mle_disc(gamma,D,kmin,kmax):
    if kmax ==None:
        eps=10.0**(-6)
        if gamma <= 1.:
            gamma=1. + eps
    S = zeta_minmax(gamma,kmin,kmax)
    L = np.log(S) + gamma*D
    return L

## Random number generator for discrete powerlaw
def random_pow_disc_binary_array(N,kmin,kmax,gamma):
    np.random.seed()
    x_uni = np.sort(np.random.rand(N))
    x_uni_tmp = 1.0*x_uni
    x1 = kmin
    x2 = kmin
    ind_done = ([])
    x1_array = ([])
    x2_array = ([])
    while len(x_uni_tmp)>0:
        cdf_x2 = cdf_power_disc(x2,kmin,kmax,gamma)
        x_uni_tmp = np.delete(x_uni_tmp,ind_done)
        ind_done = np.where(cdf_x2>x_uni_tmp)[0]
        x1_array = np.append(x1_array,x1*np.ones(len(ind_done)))
        x2_array = np.append(x2_array,x2*np.ones(len(ind_done)))
        x1 = x2
        x2 = 2.0*x1
        
    x_uni_tmp = 1.0*x_uni
    ind_done = ([])
    x_random = ([])
    ind_done = np.where(x1_array==x2_array)[0]
    x_random = np.append(x_random,x2_array[ind_done])
    x1_array = np.delete(x1_array,ind_done)
    x2_array = np.delete(x2_array,ind_done)
    x_uni_tmp = np.delete(x_uni_tmp,ind_done)   
    while len(x_uni_tmp)>0:
        ind_done = np.where(x1_array==(x2_array-1))[0]
        x_random = np.append(x_random,x2_array[ind_done])
        x1_array = np.delete(x1_array,ind_done)
        x2_array = np.delete(x2_array,ind_done)
        x_uni_tmp = np.delete(x_uni_tmp,ind_done)
        x_mid = (0.5*(x2_array+x1_array)).astype(int)
        cdf_xmid = cdf_power_disc(x_mid,kmin,kmax,gamma)
        
        x1_array = (cdf_xmid > x_uni_tmp)*x1_array + (cdf_xmid <= x_uni_tmp)*(x_mid)
        x2_array = (cdf_xmid > x_uni_tmp)*x_mid + (cdf_xmid <= x_uni_tmp)*x2_array
    return np.array(x_random).astype('int')
