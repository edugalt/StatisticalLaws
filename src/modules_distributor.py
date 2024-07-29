import numpy as np
import os, sys,codecs,time
from modules_fitting_2exp import *
from modules_fitting_lognormal import *
from modules_fitting_naranan import *
from modules_fitting_rgf import *
from modules_fitting_weibull import *
from modules_fitting_spl import *
from modules_fitting_pow import *



src_dir = os.path.abspath(os.path.join(os.pardir,'src'))
sys.path[0] = src_dir

def fit(model,counts,nrep=2,rmin_Fit=1,rmax_Fit=None):
    ''' Fit model to dataset, using nrep=2 repetitions and range from r_min=1 to r_max=infty'''

    #########################################################
    
    if model=="double_powerlaw":
        out=fit_double_powerlaw(counts,nrep,rmin_Fit,rmax_Fit)

    if model=="lognormal":
        out=fit_lognormal(counts,nrep,rmin_Fit,rmax_Fit)

    if model=="naranan":
        out=fit_naranan(counts,nrep,rmin_Fit,rmax_Fit)
        
    if model=="expcutoff":
        out=fit_expcutoff(counts,nrep,rmin_Fit,rmax_Fit)

    if model=="weibull":
        out=fit_weibull(counts,nrep,rmin_Fit,rmax_Fit)

    if model=="shifted":
        out=fit_shifted(counts,nrep,rmin_Fit,rmax_Fit)
        
    if model=="simple":
        out=fit_simple(counts,nrep,rmin_Fit,rmax_Fit)

    if model=="double_2gammas":
        out=fit_double_2gammas(counts,nrep,rmin_Fit,rmax_Fit)
    
    return(out)
    

def fit_double_powerlaw(counts,nrep,rmin_fit,rmax_fit):
    '''Fits the double power-law model'''
    gamma1 = 1.0 # gamma1 is fixed
    L = np.inf # normalized negative log-likelihood
    gamma2 = -1
    rm = -1
    # range to sample initial conditions for the free parameters
    x1_0 = (1.,3.)
    x2_0 = (1.,10**4)
    n_success = 0
    np.random.seed(41)
    for i_nrep in range(nrep):

        gamma2_0 =  np.random.random()*( x1_0[1]-x1_0[0]  ) + x1_0[0] 
        rm_0 = int(  np.random.random()*( x2_0[1]-x2_0[0]  ) + x2_0[0] )
    #     print(gamma2_0,rm_0)
        result = mle_disc_minmax_optimize_2expzipf_2par(counts,rmin_fit,rmax_fit,gamma1,gamma2_0,rm_0)

        gamma2_tmp,rm_tmp = result['x']
        L_tmp = result['fun']
        warnflag = result['success']
        if warnflag==True:
            n_success+=1
        if L_tmp<L and warnflag==True:
            L=L_tmp
            gamma2 = gamma2_tmp
            rm = rm_tmp
    return([gamma2,rm],L,n_success)

def fit_double_2gammas(counts,nrep,rmin_fit,rmax_fit):
    '''Fits the double power-law model'''
    gamma1 = -1 
    L = np.inf # normalized negative log-likelihood
    gamma2 = -1
    rm = -1
    # range to sample initial conditions for the free parameters
    x1_0 = (1.,3.)
    x2_0 = (1.,10**4)
    n_success = 0
    np.random.seed(41)
    for i_nrep in range(nrep):
        gamma1_0 =  np.random.random()*( x1_0[1]-x1_0[0]  ) + x1_0[0] 
        gamma2_0 =  np.random.random()*( x1_0[1]-x1_0[0]  ) + x1_0[0] 
        rm_0 = int(  np.random.random()*( x2_0[1]-x2_0[0]  ) + x2_0[0] )
    #     print(gamma2_0,rm_0)
        result = mle_disc_minmax_optimize_2expzipf_3par(counts,rmin_fit,rmax_fit,gamma1_0,gamma2_0,rm_0)
        gamma1_tmp,gamma2_tmp,rm_tmp = result['x']
        L_tmp = result['fun']
        warnflag = result['success']
        if warnflag==True:
            n_success+=1
        if L_tmp<L and warnflag==True:
            L=L_tmp
            gamma1 = gamma1_tmp
            gamma2 = gamma2_tmp
            rm = rm_tmp
    return([gamma1,gamma2,rm],L,n_success)


def fit_lognormal(counts,nrep,rmin_fit,rmax_fit):
    '''Fits lognormal model'''
    L = np.inf # normalized negative log-likelihood
    mu = -1
    sigma = -1

    # range to sample initial conditions for the free parameters
    x1_0 = (-10,10)
    x2_0 = (0.,10.)

    n_success = 0
    np.random.seed(41)
    for i_nrep in range(nrep):

        mu_0 =  np.random.random()*( x1_0[1]-x1_0[0]  ) + x1_0[0] 
        sigma_0 = np.random.random()*( x2_0[1]-x2_0[0]  ) + x2_0[0] 
#        print(mu_0,sigma_0)
        result = mle_disc_minmax_optimize_lognormalzipf(counts,rmin_fit,rmax_fit,mu_0,sigma_0)

        mu_tmp,sigma_tmp = result['x']
        L_tmp = result['fun']
        warnflag = result['success']
        if warnflag==True:
            n_success+=1
        if L_tmp<L and warnflag==True:
            L=L_tmp
            mu = mu_tmp
            sigma = sigma_tmp
#        print(mu,sigma,L,warnflag)
#        print('')
    return([mu,sigma],L,n_success)


def fit_naranan(counts,nrep,rmin_fit,rmax_fit):
    '''Fits Naranan (powerlaw with exponential cutoff -beginning)'''
    L = np.inf # normalized negative log-likelihood
    gamma = -1
    b = -1

    # range to sample initial conditions for the free parameters
    x1_0 = (0,10)
    x2_0 = (0.,100.)

    n_success = 0
    np.random.seed(41)
    for i_nrep in range(nrep):

        gamma_0 =  np.random.random()*( x1_0[1]-x1_0[0]  ) + x1_0[0] 
        b_0 = np.random.random()*( x2_0[1]-x2_0[0]  ) + x2_0[0] 
#        print(gamma_0,b_0)
        result = mle_disc_minmax_optimize_narananzipf(counts,rmin_fit,rmax_fit,gamma_0,b_0)

        gamma_tmp,b_tmp = result['x']
        L_tmp = result['fun']
        warnflag = result['success']
        if warnflag==True:
            n_success+=1
        if L_tmp<L and warnflag==True:
            L=L_tmp
            gamma = gamma_tmp
            b = b_tmp
      #  print(gamma,b,L,warnflag)
      #  print('')
    return([gamma,b],L,n_success)


def fit_expcutoff(counts,nrep,rmin_fit,rmax_fit):
    '''Fits powerlaw with exponential cutoff '''

    L = np.inf # normalized negative log-likelihood
    gamma = -1
    b = -1

    # range to sample initial conditions for the free parameters
    x1_0 = (0,10)
    x2_0 = (0.,100.)

    n_success = 0
    np.random.seed(41)
    for i_nrep in range(nrep):

        gamma_0 =  np.random.random()*( x1_0[1]-x1_0[0]  ) + x1_0[0] 
        b_0 = np.random.random()*( x2_0[1]-x2_0[0]  ) + x2_0[0] 
#        print(gamma_0,b_0)
        result = mle_disc_minmax_optimize_rgfzipf(counts,rmin_fit,rmax_fit,gamma_0,b_0)

        gamma_tmp,b_tmp = result['x']
        L_tmp = result['fun']
        warnflag = result['success']
        if warnflag==True:
            n_success+=1
        if L_tmp<L and warnflag==True:
            L=L_tmp
            gamma = gamma_tmp
            b = b_tmp
#        print(gamma,b,L,warnflag)
#        print('')
    return([gamma,b],L,n_success)


def fit_weibull(counts,nrep,rmin_fit,rmax_fit):
    '''Fits Weibull '''
    L = np.inf # normalized negative log-likelihood
    gamma = -1
    b = -1

    # range to sample initial conditions for the free parameters
    x1_0 = (0,1)
    x2_0 = (0.,100.)

    n_success = 0
    np.random.seed(41)
    for i_nrep in range(nrep):

        gamma_0 =  np.random.random()*( x1_0[1]-x1_0[0]  ) + x1_0[0] 
        b_0 = np.random.random()*( x2_0[1]-x2_0[0]  ) + x2_0[0] 
#        print(gamma_0,b_0)
        result = mle_disc_minmax_optimize_weibull_zipf(counts,rmin_fit,rmax_fit,gamma_0,b_0)

        gamma_tmp,b_tmp = result['x']
        L_tmp = result['fun']
        warnflag = result['success']
        if warnflag==True:
            n_success+=1
        if L_tmp<L and warnflag==True:
            L=L_tmp
            gamma = gamma_tmp
            b = b_tmp
#        print(gamma,b,L,warnflag)
#        print('')
    return([gamma,b],L,n_success)



def fit_shifted(counts,nrep,rmin_fit,rmax_fit):
    '''Fits shifted power law '''
    L = np.inf # normalized negative log-likelihood
    gamma = -1
    b = -1

    # range to sample initial conditions for the free parameters
    x1_0 = (1.,3.)
    x2_0 = (0.,100.)

    n_success = 0
    np.random.seed(41)
    for i_nrep in range(nrep):

        gamma_0 =  np.random.random()*( x1_0[1]-x1_0[0]  ) + x1_0[0] 
        b_0 = np.random.random()*( x2_0[1]-x2_0[0]  ) + x2_0[0] 
#        print(gamma_0,b_0)
        result = mle_disc_minmax_optimize_splzipf(counts,rmin_fit,rmax_fit,gamma_0,b_0)

        gamma_tmp,b_tmp = result['x']
        L_tmp = result['fun']
        warnflag = result['success']
        if warnflag==True:
            n_success+=1
        if L_tmp<L and warnflag==True:
            L=L_tmp
            gamma = gamma_tmp
            b = b_tmp
 #       print(gamma,b,L,warnflag)
 #       print('')
    return([gamma,b],L,n_success)


def fit_simple(counts,nrep,rmin_fit,rmax_fit):
    '''Fits simple power law '''

    L = np.inf # normalized negative log-likelihood
    gamma = -1

    # range to sample initial conditions for the free parameters
    x1_0 = (1.,3.)

    n_success = 0
    np.random.seed(41)
    for i_nrep in range(nrep):

        gamma_0 =  np.random.random()*( x1_0[1]-x1_0[0]  ) + x1_0[0] 
#        print(gamma_0,b_0)
        result = mle_disc_minmax_optimize_powzipf(counts,rmin_fit,rmax_fit,gamma_0)

        gamma_tmp, = result['x']
        L_tmp = result['fun']
        warnflag = result['success']
        if warnflag==True:
            n_success+=1
        if L_tmp<L and warnflag==True:
            L=L_tmp
            gamma = gamma_tmp
#        print(gamma,L,warnflag)
#        print('')
    return([gamma],L,n_success)


