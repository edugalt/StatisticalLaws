import numpy as np
from scipy.optimize import minimize
from modules_fitting_gen import *


## 2 powerlaws: pk = c*k**(-gamma1) if k<= km, c*km**(gamma2-gamm1)*k**(-gamma2) if k >= km
# faktor fuer k>=km ergibt sich durch anschlussbedingung

def pdf_2exp_disc(x,kmin,kmax,gamma1,gamma2,km):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
    '''
    if kmax == None:
        C = 1.0/(zeta_minmax(gamma1,kmin,km+1) + km**(gamma2-gamma1)*zeta_minmax(gamma2,km+1,kmax) )
    else:
        C = 1.0/(zeta_minmax(gamma1,kmin,km+1) + km**(gamma2-gamma1)*zeta_minmax(gamma2,km+1,kmax+1) )
    pk = C*((x<=km)*x**(-gamma1) + (x>km)*km**(gamma2-gamma1)*x**(-gamma2))
    return pk
    
def cdf_2exp_disc(x,kmin,kmax,gamma1,gamma2,km):
    '''returns discrete power law with cutoff kmin,kmax...including kmax as last element
       therefore kmax+1 in argument. 
    '''
    zeta_km = zeta_minmax(gamma1,kmin,km+1)
    if kmax == None:
        C = 1.0/(zeta_minmax(gamma1,kmin,km+1) + km**(gamma2-gamma1)*zeta_minmax(gamma2,km+1,kmax) )
    else:
        C = 1.0/(zeta_minmax(gamma1,kmin,km+1) + km**(gamma2-gamma1)*zeta_minmax(gamma2,km+1,kmax+1) )
    Fk = C*((x<=km)*zeta_minmax(gamma1,kmin,x+1) + (x>km)*(zeta_km + km**(gamma2-gamma1)*zeta_minmax(gamma2,km+1,x+1)))
    return Fk


## fit 3 parameters, gamma1 as well
def mle_disc_minmax_optimize_2expzipf_3par(x,kmin,kmax,gamma1_0,gamma2_0,km_0,method = 'Nelder-Mead'):
    N = float(len(x)) #types
    M = float(sum(x)) #tokens
    x = np.sort(x)[::-1]
    rlog = np.log(np.arange(N)+1)
    
    D0 = sum((x/M)*rlog)
    # result = fmin(func_mle_disc_2expzipf_2par, [gamma2_0,km_0], args=(x,N,M,D0,kmin,kmax,gamma1),xtol=10**(-6),ftol=10**(-6),disp=0,full_output=1)
    result = minimize(
        func_mle_disc_2expzipf_3par, 
        [gamma1_0,gamma2_0,km_0], 
        args=(x,N,M,D0,kmin,kmax),
        # bounds = ((1.0,None),(1.,None)),
        options = {'disp':False},
        method=method)
    # convergence message 0...good,1...bad
    # warnflag = result[4]
    warnflag = result['success']
    if warnflag != True:
        print('No convergence in maximizing likelihood!')
    return result



## fit only 2 parameters
def mle_disc_minmax_optimize_2expzipf_2par(x,kmin,kmax,gamma1,gamma2_0,km_0,method = 'Nelder-Mead'):
    N = float(len(x)) #types
    M = float(sum(x)) #tokens
    x = np.sort(x)[::-1]
    rlog = np.log(np.arange(N)+1)
    
    D0 = sum((x/M)*rlog)
    # result = fmin(func_mle_disc_2expzipf_2par, [gamma2_0,km_0], args=(x,N,M,D0,kmin,kmax,gamma1),xtol=10**(-6),ftol=10**(-6),disp=0,full_output=1)
    result = minimize(
        func_mle_disc_2expzipf_2par, 
        [gamma2_0,km_0], 
        args=(x,N,M,D0,kmin,kmax,gamma1),
        # bounds = ((1.0,None),(1.,None)),
        options = {'disp':False},
        method=method)
    # convergence message 0...good,1...bad
    # warnflag = result[4]
    warnflag = result['success']
    if warnflag != True:
        print('No convergence in maximizing likelihood!')
    return result

def func_mle_disc_2expzipf_2par(params,x,N,M,D0,kmin,kmax,gamma1):
    gamma2 = params[0]
    if kmax==None:
        eps=10.0**(-6)
        if gamma2 <= 1.:
            gamma2=1. + eps
    km = int(np.abs(params[1]))
    # print(gamma2,km,gamma1,N)
    if kmax == None:
        C = zeta_minmax(gamma1,kmin,km+1) + km**(gamma2-gamma1)*zeta_minmax(gamma2,km+1,kmax) 
    else:
#         print(zeta_minmax(gamma1,kmin,km+1))
#         print(zeta_minmax(gamma2,km+1,kmax+1))
        C = zeta_minmax(gamma1,kmin,km+1) + km**(gamma2-gamma1)*zeta_minmax(gamma2,km+1,kmax+1)
    if km > N:
        summe1 = sum((x/M)*np.log(np.arange(N)+1))
    else:
#         print(km)
        summe1 = sum((x[:int(km)]/M)*np.log(np.arange(int(km))+1))
    summe2 = D0 - summe1
    summe3 = sum((x[int(km)+1:]/M))
    L = np.log(C) + gamma1*summe1 + gamma2*summe2 - summe3*(gamma2-gamma1)*np.log(km)
    return L

def func_mle_disc_2expzipf_3par(params,x,N,M,D0,kmin,kmax):
    gamma1 = params[0]
    gamma2 = params[1]
    if kmax==None:
        eps=10.0**(-6)
        if gamma2 <= 1.:
            gamma2=1. + eps
    km = int(np.abs(params[2]))
    # print(gamma2,km,gamma1,N)
    if kmax == None:
        C = zeta_minmax(gamma1,kmin,km+1) + km**(gamma2-gamma1)*zeta_minmax(gamma2,km+1,kmax) 
    else:
#         print(zeta_minmax(gamma1,kmin,km+1))
#         print(zeta_minmax(gamma2,km+1,kmax+1))
        C = zeta_minmax(gamma1,kmin,km+1) + km**(gamma2-gamma1)*zeta_minmax(gamma2,km+1,kmax+1)
    if km > N:
        summe1 = sum((x/M)*np.log(np.arange(N)+1))
    else:
#         print(km)
        summe1 = sum((x[:int(km)]/M)*np.log(np.arange(int(km))+1))
    summe2 = D0 - summe1
    summe3 = sum((x[int(km)+1:]/M))
    L = np.log(C) + gamma1*summe1 + gamma2*summe2 - summe3*(gamma2-gamma1)*np.log(km)
    return L
