from math import *
import numpy as np


def a(params,x,l): #FIRST TRY: FUNCTIONS TOO INNEFICIENT, MOVING TO MATRIX APPROACH
    "computes the location-population parameter a for each city in x"
    alist=[]
    alpha=params[0]
    for i in l:
        aii=1/(1+alpha*d(i,x))
        alist.append(np.sum(aii*x))
    return(np.array(alist))

def minus_log_likelihood(params,x,y,l): # With two parameters, not used initially
    alpha=params[0]
    beta=params[1]
    a=aii(params,l,x)
    prnon=x*a**(beta-1)
    p=prnon/np.sum(prnon)
    logL=y*np.log(p)
    return -np.sum(logL)

def d(i,l):
    "Return the distancesbetwen point i and vector l in km (input is [lat,lon] in decimals"
    earthradius = 6371.0088 # Mean radius according to Wikipdeia and International Union of Geodesy and Geophysics
    lat=np.radians(np.swapaxes(l,0,1)[0]) # array with latitudes
    lon=np.radians(np.swapaxes(l,0,1)[1]) # array with longitudes
    lati =np.radians(np.repeat(i[0],len(l))) # np array of length entries with latitude of i
    loni = np.radians(np.repeat(i[1],len(l))) # np array of length entries with longitude of i
    
    angles=np.power(np.sin(lat-lati),2)+np.cos(lat)*np.cos(lati)*np.power(np.sin((lon-loni)/2),2)
    return 2*earthradius*np.arcsin(np.sqrt(angles))

def aii(params,l,x):
    "Return the location parameter for city all cities"
    alpha=params[0]
    alist=[]
    for i in l: #For each city, compute the distances
        aii=np.divide(1,1+alpha*np.power(d(i,l),2))
        alist.append(np.sum(aii*x))
    return np.array(alist)

def dmatrix(l):
    "Return matrix of distances between cities"
    result=[]
    for i in l:
        result.append(d(i,l))
    return np.matrix(result)

def amatrix(alpha,l):
    "Return matrix of aii,i.e., 1/(1+alpha (d_i,j)^2)"
    dmat=dmatrix(l)
    return np.divide(1,1+alpha*np.power(dmat,2))

def aModelG(dmat,alpha,gamma=2):
    "Gets a matrix of distances, returns a matrix aii for model gravitation with exponent gamma and extra parameter alpha"
    return np.divide(1,1+alpha*np.power(dmat,gamma))

def aModelE(dmat,alpha):
    "Gets a matrix of distances, returns a matrix aii for model exponential distance"
    return np.exp(-alpha*dmat)

def minus_log_likelihood_beta(beta,x,y,amat):
    "Minus log-likelihood suited for fixed alpha, varying bet (the amat matrix is given, not changing)"
    a = np.array(x*amat) # check with np.matrix times np.array multiplication
    prnon = x*np.power(a,beta-1)
    p=prnon/np.sum(prnon)
    logL=y*np.log(p)
    return -np.sum(logL)

def amatModel(dmat,modelname="no",par=[1,1]):
    "Choose the name of the model, return the distance matrix for fixed alpha; Parameters should contain the parameters of the model, e.g., [alpha,gamma] for the gravitational model"
    if modelname=="no":
        return np.matrix(np.identity(len(dmat)))
    if modelname=="gravitational":
        return aModelG(dmat,par[0],par[1])
    if modelname=="exponential":
        return aModelE(dmat,par[0])
    
    
####################################

