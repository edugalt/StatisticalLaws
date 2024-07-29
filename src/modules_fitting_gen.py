import numpy as np
from scipy.special import zeta
import mpmath as mpm

def wfd_lin(x):
    '''In: x...list with floats representing data...size of all the words
       Out: k, P(k)...how many words with length k
            array is only constructed for those k-values for which Pk > 0!!!
    '''
    x.sort()
    k = []
    P_k = []
    count_temp = 0
    total = float(len(x))

    for i in np.arange(len(x)):
        if x[i] != count_temp:
            k += [float(x[i])]
            P_k += [float(1)]
            count_temp = x[i]
        else:
            P_k[-1] += 1.0
    P_k = np.array(P_k)/total
    return np.array(k),P_k
    
def wfd_histo(x):
    '''In: x...list with floats representing data...size of all the words
       Out: k, N(k)...how many words with length k
            array is only constructed for those k-values for which Pk > 0!!!
    '''
    x.sort()
    k = []
    N_k = []
    count_temp = 0
    N = float(len(x))
    i = 0
    while i < N:
        if x[i] != count_temp:
            k += [float(x[i])]
            N_k += [float(1)]
            count_temp = x[i]
        else:
            N_k[-1] += 1.0
        i += 1
    return np.array(k),np.array(N_k)

    
## zeta function
def zeta_minmax(gamma,kmin,kmax):
    '''kmax == None means kmax --> infty
    '''
#     gamma = gamma[0]
    if gamma <= 1.0:
        if kmax == None:
            print('ERROR: Series does not converge!!!')
            C = 0
        else:
            mpm.dps=25
#             print(gamma,kmin,kmax,'huhu')
            C = (float(mpm.sumem(lambda k: k**(-gamma),[kmin,kmax])))
    else:
        # print(kmax)
        # print(type(kmax))
        # print(kmax==None)
        # print('')
        if isinstance(kmax,(list,np.ndarray)):
            C = zeta(gamma,kmin)-zeta(gamma,kmax)
        elif kmax == None:
            C = zeta(gamma,kmin)
        else:
            C = zeta(gamma,kmin)-zeta(gamma,kmax)
        # print(C)
    return C
    
def zeta_minmax_cont(gamma,kmin,kmax):
    '''kmax == None means kmax --> infty
    '''
    if kmax==None:
        if gamma > 1.0:
            C = 1.0/(gamma-1)*(kmin**(1-gamma))
        else:
            C = -1#1.0/(gamma-1)*(kmin**(1-gamma) - kmax**(1-gamma))
            print('Error')
    else:
        C = 1.0/(gamma-1)*(kmin**(1-gamma) - kmax**(1-gamma))
    return C

## Leastsquares
def leastsquare(x,y):
    '''least square for y = a0 +a1*x
    '''
    a_0 = (sum(y)*sum(x*x) - sum(x)*sum(x*y))/(len(x)*sum(x*x) - (sum(x))**2.0)
    a_1 = (len(x)*sum(x*y) - sum(x)*sum(y))/(len(x)*sum(x*x) - (sum(x))**2.0)
    return a_0,a_1
    
## Filter functions
def filter_xy(x,y,xmin,xmax,ymin,ymax):
    '''filter where P(k) = 0 when bin in histogram still empty and y = P_k < kmin
    '''
    ind_del_ymin = np.where(y<ymin)[0]
    ind_del_ymax = np.where(y>ymax)[0]
    ind_del_y = np.append(ind_del_ymin,ind_del_ymax)
    
    if xmin != None:
        ind_del_xmin = np.where(x<xmin)[0]
    else:
        ind_del_xmin = []
    if xmax != None:
        ind_del_xmax = np.where(x>xmax)[0]
    else:
        ind_del_xmax = []
    ind_del_x = np.append(ind_del_xmin,ind_del_xmax)
    
    ind_del = np.append(ind_del_y,ind_del_x)
    x_f = np.delete(x,ind_del) 
    y_f = np.delete(y,ind_del)
    return x_f,y_f

def filter_x(x,xmin,xmax):
    '''filter where P(k) = 0 when bin in histogram still empty and y = P_k < kmin
       if xmin or xmax = None it means that it does not exist
    '''
    if xmin != None:
        ind_del_xmin = np.where(x<xmin)[0]
    else:
        ind_del_xmin = []
    if xmax != None:
        ind_del_xmax = np.where(x>xmax)[0]
    else:
        ind_del_xmax = []
    ind_del = np.append(ind_del_xmin,ind_del_xmax)
    
    x_f = np.delete(x,ind_del) 
    return x_f
    
