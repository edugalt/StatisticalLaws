# Module for working with constrained power-law surrogates.
# This code is associated with the manuscript "Non-parametric power-law surrogates" which currently (2021-12-14) is under review at PRX.
# 
# The code is provided without warranty or guarantee.
# 
# 
# 
# 
ts_folder_name = 'time-series'
results_folder_name = 'results'
fig_folder_name = 'figures'
# 
# 
# 
# 
# Load modules
# 
# 
# 
# 
import numpy as np# For pointwise and pairwise array operations
np.seterr(over='raise')# Make error when a floating point overflow is detected
import networkx as nx# For discrete zeta function
from sympy.ntheory import factorint# For prime factorisation
import random# For random number generation
from functools import reduce# For finding the product of all elements of a list
from scipy.special import zeta, factorial
from mpmath import gammainc #gammainc(z, a=0, b=inf) computes the (generalized) incomplete gamma function with integration limits [a, b]
from timeit import default_timer as timer#For timing (and also seeding random number generators)
import operator# To check whether or not prime factor dictionaries are equal
from math import log, floor, ceil, erf
from scipy.stats import rv_discrete# Generate from truncated power-law
from scipy.stats import truncnorm# Generate from a normal distribution limited to a finite interval
from scipy.stats import rankdata#For ordinal patterns
import itertools# Find all factors of an integer
import math as math# For rounding
import codecs, json# Saving and loading data in .json format
import matplotlib.pyplot as plt#For plotting
# 
# 
# 
# 
# Code (mostly) for generating surrogate sequences and other time series of various kinds
# 
# 
# 
# 
# 
# 
# 
# 
# Generate integer compositions.
# 
# Adapted (slightly) from
# https://pythonhosted.org/combalg-py/index.html#combalg.subset.random_k_element
def random_k_element(elements, k):
    '''
    Returns a random k-element subset of a set of elements.  This is a clever algorithm, explained in [NW1978]_
    but not actually implemented.  Essentially starting with the first element of the universal set, include
    it in the subset with p=k/n.  If the first was selected, select the next with p=(k-1)/(n-1), if not
    p = k/(n-1), etc.

    :param elements: the input set
    :type elements: list
    :param k: the size of the output set
    :type k: int
    :return: a random k-element subset of the input set
    :rtype: list
    '''
    a = []
    c1 = k
    c2 = len(elements)
    i = 0
    while c1 > 0:
        if random.random() <= float(c1)/c2:
            a.append(elements[i])
            c1 -= 1
        c2 -= 1
        i += 1
    return a
# 
# Adapted (slightly) from
# https://pythonhosted.org/combalg-py/_modules/combalg/combalg.html#composition.random
def rand_comp(n,k):
    '''
    Returns a random composition of n into k parts.

    :param n: integer to compose
    :type n: int
    :param k: number of parts to compose
    :type k: int
    :return: a list of k-elements which sum to n

    Returns random element of :func:`compositions` selected
    `uniformly at random <https://en.wikipedia.org/wiki/Discrete_uniform_distribution>`_.
    '''
    if (k > 1):
        a = random_k_element(range(1, n + k), k - 1)
        r = [0]*k
        r[0] = a[0] - 1
        for j in range(1,k-1):
            r[j] = a[j] - a [j-1] - 1
        r[k-1] = n + k - 1 - a[k-2]
        return r
    elif (k == 1):
        return [n]
    else:
        return []
# 
# 
# 
# 
# Calculate entropies.
# 
### From:
# https://datascience.stackexchange.com/questions/58565/conditional-entropy-calculation-in-python-hyx
# 
##Entropy
def entropy(Y):
    """
    Also known as Shannon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en
# 
#Joint Entropy
def joint_entropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y,X]
    return entropy(YX)
# 
#Conditional Entropy
def cond_entropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return joint_entropy(Y, X) - entropy(X)
#
#
#
#
# Generate power-law with exponential cut-off.
#
# Adapted (slightly) from
# http://tuvalu.santafe.edu/~aaronc/powerlaws/
#for use in Python 3 and to avoid clashes with other packages or names chosen before downloading randht.py. 

# from math import *
# from random import *

# function x=randht(n, varargin)

# RANDHT generates n observations distributed as some continous heavy-
# tailed distribution. Options are power law, log-normal, stretched 
# exponential, power law with cutoff, and exponential. Can specify lower 
# cutoff, if desired.
# 
#    Example:
#       x = randht(10000,'powerlaw',alpha);
#       x = randht(10000,'xmin',xmin,'powerlaw',alpha);
#       x = randht(10000,'cutoff',alpha, lambda);
#       x = randht(10000,'exponential',lambda);
#       x = randht(10000,'lognormal',mu,sigma);
#       x = randht(10000,'stretched',lambda,beta);
#
#    See also PLFIT, PLVAR, PLPVA
#
#    Source: http://www.santafe.edu/~aaronc/powerlaws/


# Version 1.0.2 (2008 April)
# Copyright (C) 2007 Aaron Clauset (Santa Fe Institute)

# Ported to python by Joel Ornstein (2011 August)
# (joel_ornstein@hmc.edu)

# Distributed under GPL 2.0
# http://www.gnu.org/copyleft/gpl.html
# RANDHT comes with ABSOLUTELY NO WARRANTY
# 
# Notes:
# 
def randht(n,*varargin):
    Type   = '';
    xmin   = 1;
    alpha  = 2.5;
    beta   = 1;
    Lambda = 1;
    mu     = 1;
    sigma  = 1;


    # parse command-line parameters; trap for bad input
    i=0; 
    while i<len(varargin): 
        argok = 1; 
        if type(varargin[i])==str: 
            if varargin[i] == 'xmin':
                xmin = varargin[i+1]
                i = i + 1
            elif varargin[i] == 'powerlaw':
                Type = 'PL'
                alpha  = varargin[i+1]
                i = i + 1
            elif varargin[i] == 'cutoff':
                Type = 'PC';
                alpha  = varargin[i+1]
                Lambda = varargin[i+2]
                i = i + 2
            elif varargin[i] == 'exponential':
                Type = 'EX'
                Lambda = varargin[i+1]
                i = i + 1
            elif varargin[i] == 'lognormal':
                Type = 'LN';
                mu = varargin[i+1]
                sigma = varargin[i+2]
                i = i + 2
            elif varargin[i] == 'stretched':
                Type = 'ST'
                Lambda = varargin[i+1]
                beta = varargin[i+2]
                i = i + 2
            else: argok=0
                
        if not argok: 
            print('(RANDHT) Ignoring invalid argument {}', i+1) 
      
        i = i+1 

    if n<1:
        print('(RANDHT) Error: invalid ''n'' argument; using default.\n)')
        n = 10000;

    if xmin < 0.5:#Previously this line read "    if xmin < 1:"
        print('(RANDHT) Error: invalid ''xmin'' argument; using default.\n')
        xmin = 0.5;




    x=[]
    if Type == 'EX':
        x=[]
        for i in range(n):
            x.append(xmin - (1./Lambda)*log(1-random.random()))
    elif Type == 'LN':
        y=[]
        for i in range(10*n):
            y.append(np.exp(mu+sigma*random.normalvariate(0,1)))

        while True:
            y= filter(lambda X:X>=xmin,y)
            y = [val for val in y]
            q = len(y)-n;
            if q==0: break

            if q>0:
                r = [val for val in range(len(y))];
                random.shuffle(r)
                ytemp = []
                for j in range(len(y)):
                    if j not in r[0:q]:
                        ytemp.append(y[j])
                y=ytemp
                break
            if (q<0):
                for j in range(10*n):
                    y.append(np.exp(mu+sigma*random.normalvariate(0,1)))
            
        x = y
        
    elif Type =='ST':
        x=[]
        for i in range(n):
            x.append(pow(pow(xmin,beta) - (1./Lambda)*log(1.-random.random()),(1./beta)))
    elif Type == 'PC':
        
        x = []
        y=[]
        for i in range(10*n):
            y.append(xmin - (1./Lambda)*log(1.-random.random()))
        while True:
            ytemp=[]
            for i in range(10*n):
                if (random.random() <= ((y[i]/float(xmin))**(-alpha))):
                    ytemp.append(y[i])
            y = ytemp
            x = x+y
            q = len(x)-n
            if q==0:
                break;

            if (q>0):
                r = [num for num in range(len(x))]
                random.shuffle(r)

                xtemp = []
                for j in range(len(x)):
                    if j not in r[0:q]:
                        xtemp.append(x[j])
                x=xtemp
                break;
            
            if (q<0):
                y=[]
                for j in range(10*n):
                    y.append(xmin - (1./Lambda)*log(1.-random.random()))


    else:
        x=[]
        for i in range(n):
            x.append(xmin*pow(1.-random.random(),-1./(alpha-1.)))

    return x
# 
# 
# 
# 
# Generate constrained Markov order power-law surrogates.
#
# Adapted from the MATLAB function stirl in whittle_surrogate.m, Shawn Pethel, 2012
# Use Stirling's approximation to calculate (approximately) the log of the exponential function
def stirl(xx):# xx should be a numpy array
    xx = xx.astype(float);
    ii = (xx <= 16);
    jj = (xx > 16);
    aa = xx[ii];
    bb = xx[jj];
    s1 = np.log(factorial(aa))
    s2 = bb*np.log(bb) - bb + 0.5*np.log(2*np.pi*bb) + 1/(12*bb) - 1/(360*bb**3) + 1/(1260*bb**5) - 1/(1680*bb**7);
    s = xx;
    s[ii] = s1;
    s[jj] = s2;
    return s
# 
# Adapted from
# https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python
# Here is a function which implements the above ideas:
# 
#the following code takes a list such as
#[1,1,2,6,8,5,5,7,8,8,1,1,4,5,5,0,0,0,1,1,4,4,5,1,3,3,4,5,4,1,1]
#with states labeled as successive integers starting with 0
#and returns a transition matrix, M,
#where M[i][j] is the probability of transitioning from i to j
# 
# Calculate transition count matrix from a sequence of consecutive integers starting from 0
def transition_matrix(seq):
    n = 1 + max(seq) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(seq,seq[1:]):
        M[i][j] += 1

#     #now convert to probabilities:
#     for row in M:
#         s = sum(row)
#         if s > 0:
#             row[:] = [f/s for f in row]
    return np.array(M)
# 
# Adapted from the MATLAB code whittle_surrogate.m, Shawn Pethel, 2012
# Generate a constrained first order Markov surrogate from a transition count matrix
def count_mat_to_surr(f0, u, v):
    f = np.copy(f0);
    N = len(f);
    sym = np.arange(0, N);
    n = np.sum(f) + 1;
    stlut = stirl(np.arange(0., n + 1));
    seye = np.diag([1.]*N);
    ytrial = np.array([0]*n);
    rsum = np.sum(f, 1);
    term1 = np.sum(stlut[rsum]) - np.sum(np.sum(stlut[f[np.nonzero(f)]]));
    rsum[rsum == 0] = 1;
    sd = np.array(1/rsum);
    rsum = np.sum(f, 1);
    for mm in range(0, n - 2):
        ytrial[mm] = u;
        rsum[u] -= 1;
        if (rsum[u] > 0):
            sd[u] = 1/rsum[u];
        else:
            sd[u] = 0;
        rw = sym[f[u, :] > 0];
        numb = np.zeros([0, 3]);
        cnt = 0;
        for ii in range(0, len(rw)):
            ut = int(rw[ii]);
            g = np.copy(f);
            g[u, ut] -= 1
            fs = seye - np.multiply(sd[:, None], g)
            fs = np.delete(fs, v, 0);
            fs = np.delete(fs, ut, 1);
            cf = abs(np.linalg.det(fs));
            if (cf > 0):
                numb_row = np.zeros([1, 3]);
                numb_row[0, 2] = term1 + stlut[f[u, ut]] - stlut[g[u, ut]];
                numb_row[0, 1] = ut;
                numb_row[0, 0] = numb_row[0, 2] + np.log(cf);
                numb = np.append(numb, numb_row, 0);
                cnt += 1;
        if (cnt == 0):
            numb = np.zeros([1, 3]);
        ec = np.exp(numb[:, 0] - np.max(numb[:, 0]));
        ec /= np.sum(ec);
        rng = np.concatenate(([0], np.cumsum(ec)), axis=0);
#         if sum((rng < 0.999) & (rng > 0.001)):#To see whether there is any randomness
#             print(rng)
        sr = random.random();
        indx = (sr >= rng[0:-1]) & (sr <= rng[1:]);
        ut = int(numb[indx, 1]);
        f[u, ut] -= 1;
        u = ut;
        term1 = numb[indx, 2];
    ytrial[n - 2] = numb[cnt - 1, 1];
    ytrial[-1] = v;
    return ytrial
# 
# Convert a Markov surrogate of order o to a Markov surrogate of order 1
def markov_o_to_1(s, o):
    T = len(s);
    a = [];
    for ii in range(T - (o - 1)):
        a = a + [s[ii:(ii + o)]]
    w, t = np.unique(a, axis=0, return_inverse=True)
    t = t.tolist();
    return (t, w)
# 
# Revert a Markov surrogate of order 1 (previously converted using "markov_o_to_1" from a Markov surrogate of order o) to a Markov surrogate of order o
def markov_1_to_o(t, w):
    n = len(t);
    o = len(w[0]);
    s = w[t[0]].tolist();
    for ii in range(1, n):
        s = s + [w[t[ii], o - 1].tolist()];
    return s
# 
# Generate, from a sequence s, n_surr constrained Markov surrogates of order o
# Assumes states labeled as successive integers starting with 0
def gen_mc_surrogate(s, o, n_surr):
    if (o == 0):
        S = [np.random.permutation(s).tolist() for ii in range(n_surr)]
    else:
        (t, w) =  markov_o_to_1(s, o);
        f = transition_matrix(t);
        u = t[0];
        v = t[-1];
        T = [count_mat_to_surr(f, u, v) for ii in range(n_surr)]
        S = [markov_1_to_o(t, w) for t in T]
    return S
#
#
#
#
# Calculate conditional entropy higher order.
# 
# Conditional entropy of order o:
def cond_entropy_o_n(X, o):
    if (o == 0):
        c_e_o_n = entropy(X)
    else:
        (V, w) = markov_o_to_1(X, o)
        c_e_o_n = cond_entropy(X[o:], V[:-1])
    return c_e_o_n
# 
# Generate correlated power-law power-law sequence with scale exponent gamma and minimum value x_min.
# The first value drawn iid from limiting distribution.
# Subsequently, there is a probability mu of the next proposed value being drawn uniformly at random from iid power-law,
# and a probability (1 - mu) of the next proposed value being drawn uniformly at random from integers in interval [x + a, x + a + (b - 1)],
# where x is the current value.
# Should choose a >= 0. 
def p_l_seq_m_c_1(N, gamma, x_min, a, b, mu):
    x1 = int(nx.utils.zipf_rv(alpha=gamma, xmin=x_min))
    seq = [x1]
    for ii in range(1, N):
        x0 = x1
        q = random.random()
        if q < mu:
            xp = int(nx.utils.zipf_rv(alpha=gamma, xmin=x_min))
        else:
            xp = x0 + random.randint(a, a + b - 1)
        diff = xp - x0
        Anum = 1 + (1 - mu)/mu*(xp/x1)**-gamma*(((-diff) >= a) and ((-diff) <= (a + b - 1)))/b;
        Aden = 1 + (1 - mu)/mu*(x1/xp)**-gamma*(((+diff) >= a) and ((+diff) <= (a + b - 1)))/b;
        A = Anum/Aden
        q = random.random()
        if q < A:
            x1 = xp
        else:
            x1 = x0
        seq = seq + [x1]
    return seq
# 
#
#
#
# Generate power-law sequences with specified Markov order.
# 
# Generate power-law restricted to interval of integers [a, b]:
# Adapted (slightly) from
# https://stackoverflow.com/questions/24579269/sample-a-truncated-integer-power-law-in-python
def truncated_power_law(gam, a, b):
    x = np.arange(a, b + 1, dtype='float')
    pmf = 1/x**gam
    pmf /= pmf.sum()
    pmf /= pmf.sum()# Repeat normalisation in case of numerical issues
    return rv_discrete(values=(range(a, b + 1), pmf))
# 
# 
# Generate correlated power-law sequence of order n or o and with scale exponent gamma and minimum value x_min.
# The first o values drawn iid from limiting distribution.
# Subsequent values have chance mu of being drawn iid from the limiting distribution,
# chance 1 - mu of being iid from limiting distribution restricted to bin of one of n or o most recent observations chosen uniformly at random.
def gen_mc_pl_seq(N, gamma, x_min, b, mu, o=1):
    x1_seq = [int(nx.utils.zipf_rv(alpha=gamma, xmin=x_min)) for ii in range(o)]#Initial values
    bin1_seq = [int(floor(np.log(x1/(x_min - 0.5))/np.log(b))) for x1 in x1_seq]
    bin1_low_lim_seq = [(x_min - 0.5)*(b**bin1) for bin1 in bin1_seq]
    bin1_upp_lim_seq = [(x_min - 0.5)*(b**(bin1 + 1)) for bin1 in bin1_seq]
    power_law1_seq = [truncated_power_law(gamma, int(ceil(bin1_low_lim_seq[ii])), int(floor(bin1_upp_lim_seq[ii]))) for ii in range(o)]
    seq = x1_seq
    for ii in range(o, N):
        q = random.random()
        if (q < mu) or (o == 0):
            x1 = int(nx.utils.zipf_rv(alpha=gamma, xmin=x_min))
            bin1 = int(floor(np.log(x1/(x_min - 0.5))/np.log(b)))
            bin1_low_lim = (x_min - 0.5)*(b**bin1)
            bin1_upp_lim = (x_min - 0.5)*(b**(bin1 + 1))
            power_law1 = truncated_power_law(gamma, int(ceil(bin1_low_lim)), int(floor(bin1_upp_lim)))
        else:
            ii = np.random.randint(0, o)
            power_law1 = power_law1_seq[ii]
            x1 = int(power_law1.rvs(size=1))
            bin1 = bin1_seq[ii]
            bin1_low_lim = bin1_low_lim_seq[ii]
            bin1_upp_lim = bin1_upp_lim_seq[ii]
        seq = seq + [x1]
        bin1_seq = bin1_seq[1:] + [bin1]
        bin1_low_lim_seq = bin1_low_lim_seq[1:] + [bin1_low_lim]
        bin1_upp_lim_seq = bin1_upp_lim_seq[1:] + [bin1_upp_lim]
        power_law1_seq = power_law1_seq[1:] + [power_law1]
    return seq
# 
#
#
#
# Calculate maximum likelihood scale exponent, empirical distribution, KS distance, cut-off:
#
# Code for calculating KS-distance relative to power-law, and lower cut-off which minimises KS-distance from a power-law of best fit:
#
# Code for calculating empirical cdf:
def ecdf(x_list):
    sorted_x_list = np.sort(x_list)
    n = sorted_x_list.size
    emp_cdf = np.arange(1, n + 1)/n
    x_list_unique, indices_in_orig = np.unique(np.flip(sorted_x_list), return_index=True)
    emp_cdf = [emp_cdf[n - 1 - index] for index in indices_in_orig]
    x_list_unique = list(np.sort(x_list_unique))
    return x_list_unique, emp_cdf
#
#Code for calculating KS-distance relative to power-law distribution:
def calc_ks_stat(val_seq, x_min, gamma):
    trunc_val_seq = [deg for deg in val_seq if deg >= x_min]
    unique_trunc_deg_list, emp_cdf = ecdf(trunc_val_seq)
    zeta_gamma = zeta(gamma, x_min)
    exp_cdf = [1 - zeta(gamma, float(deg + 1))/zeta_gamma for deg in unique_trunc_deg_list]
    diff_array = np.subtract(emp_cdf, exp_cdf)
    abs_diff_array = np.absolute(diff_array)
    ks_dist = max(abs_diff_array)
    return ks_dist
# 
# Code for determining maximum likelihood scaling exponent gamma:
def estim_scale_exp(val_seq, x_min):
    gamma_list = np.arange(1.001, 10.000, 0.001);# List of candidate maximum likelihood scale exponents
    zeta_h_list = [zeta(gamma, x_min) for gamma in gamma_list]
    N = len(val_seq)
    log_deg_list = [log(deg) for deg in val_seq]
    log_deg_array = np.array(log_deg_list)
    sum_log_array = np.sum(log_deg_array)
    gamma_array = np.array(gamma_list)
    zeta_h_array = np.array(zeta_h_list)
    log_zeta_h_array = np.log(zeta_h_array)
    neg_log_like_array = np.add(np.multiply(gamma_array, sum_log_array), np.multiply(N, log_zeta_h_array))
    m_l_ind = np.argmin(neg_log_like_array)
    gamma_m_l = gamma_list[m_l_ind]
    return gamma_m_l
#
# Identify cut-off which minimises the KS-distance relative to the maximum likelihood power-law.
def ident_cut_off_const(data):
    N_full = len(data)
    x_min_list = list(set(data));
    x_min_list.sort()
    x_min_list = [int(x_min) for x_min in x_min_list]
    num_x_min = len(x_min_list)
    N_list = np.zeros((num_x_min))
    ks_dist_list = np.zeros((num_x_min))
    gamma_m_l_list = np.zeros((num_x_min))

    for ii_x_min in range(num_x_min):
        x_min = x_min_list[ii_x_min]
        val_seq = [int(val) for val in data if val >= x_min]
        N = len(val_seq)
        N_list[ii_x_min] = N
        gamma_m_l = estim_scale_exp(val_seq, x_min)
        gamma_m_l_list[ii_x_min] = gamma_m_l
        ks_dist = calc_ks_stat(val_seq, x_min, gamma_m_l)
        ks_dist_list[ii_x_min] = ks_dist

    opt_x_min_ind = np.argmin(ks_dist_list)
    opt_x_min = x_min_list[opt_x_min_ind]
    gamma_m_l = gamma_m_l_list[opt_x_min_ind]
    return (opt_x_min, x_min_list, ks_dist_list, gamma_m_l, gamma_m_l_list)
#
# Calculate the empirical pdf within bins
def calc_emp_pdf(val_seq, bins):
    counts, xedges = np.histogram(val_seq, bins=bins)
    bin_widths = (bins[1:] - bins[:-1])
    emp_pdf = counts/bin_widths
    emp_pdf = list(emp_pdf)
    N = len(val_seq)
    emp_pdf = [float(p)/N if p > 0 else float('nan') for p in emp_pdf]
    return emp_pdf
#
#
#
#
# Generate power-law and other types of surrogates.
#
# Code for producing power-law (and more general) surrogates:
# 
def log_bin(val_seq, b, x_min):
    len_val_seq = len(val_seq)
    max_val = max(val_seq)
    num_bins = int(ceil(np.log((max_val + 0.5)/(x_min - 0.5))/np.log(b)))
    bin_edges = [x_min - 0.5] + [(x_min - 0.5)*(b**(p + 1)) for p in range(num_bins)]
    bin_seq = list(np.digitize(val_seq, bin_edges) - 1)
    low_lim_list = [np.ceil(bin_edges[bin_seq[ii]]) for ii in range(len_val_seq)]
    upp_lim_list = [np.floor(bin_edges[bin_seq[ii] + 1]) for ii in range(len_val_seq)]
    return bin_edges, bin_seq, low_lim_list, upp_lim_list
# 
def arb_bin(val_seq, bin_edges):
    len_val_seq = len(val_seq)
    bin_seq = list(np.digitize(val_seq, bin_edges) - 1)
    low_lim_list = [np.ceil(bin_edges[bin_seq[ii]]) for ii in range(len_val_seq)]
    upp_lim_list = [np.floor(bin_edges[bin_seq[ii] + 1]) for ii in range(len_val_seq)]
    return bin_seq, low_lim_list, upp_lim_list
# 
def subsample(seq, N):
    N0 = len(seq)
    if (N == 0):
        N = N0
    if (N > N0):
        n = N//N0
        r = N%N0
        new_seq = []
        for ii in range(n):
            new_seq = new_seq + seq
        new_seq = new_seq + subsample(seq, N=r)
    else:
        new_seq = random.sample(seq, k=N)
    random.shuffle(new_seq)
    return new_seq
# 
def bootstrap(seq, N=0):
    N0 = len(seq)
    if (N == 0):
        N = N0
    return random.choices(seq, k=N)
#
def resample(seq, method='subs', N=0):#Resampling by either: (1) bootstrapping, or (2) sampling without replacement (and replicating where necessary)
    if (method == 'boot'):
        return bootstrap(seq, N)
    else:
        return subsample(seq, N)        
# 
# Generate from a power-law sequence with known parameters
def gen_typ_p_l_surrogate(gamma_m_l, x_min, N):
    surr_val_seq = [int(nx.utils.zipf_rv(alpha=gamma_m_l, xmin=x_min)) for i in range(N)];
    return surr_val_seq
# 
# Generate from a log-normal sequence with known parameters
def gen_log_norm(sigma, x_min, N, mu=0):
    log_val_seq = truncnorm.rvs((np.log(x_min - 0.5) - mu)/sigma, (np.inf - mu)/sigma, loc=mu, scale=sigma, size = N)
    val_seq = [int(round(np.exp(log_deg))) for log_deg in log_val_seq]
    return val_seq
#
# Build a prime factor dictionary:
def build_factor_count_dict(val_seq):
    factor_count_dict = {}
    for deg in val_seq:
        deg_factor_count_dict = factorint(deg)
        for factor in deg_factor_count_dict:
            if not factor in factor_count_dict:
                factor_count_dict[factor] = deg_factor_count_dict[factor]
            else:
                factor_count_dict[factor] += deg_factor_count_dict[factor]
    return factor_count_dict
#
# Adapted slightly from:
# https://stackoverflow.com/questions/1010381/python-factorization?rq=1
#
# Find all factors of an integer
# 
def all_factors(prime_dict):
    series = [[p**e for e in range(maxe+1)] for p, maxe in prime_dict.items()]
    for multipliers in itertools.product(*series):
        yield reduce(operator.mul, multipliers)
#
#
#Work out which instances of each prime factor can be moved without incurring risk that a value will dip below the lower cut-off:
def make_move_factor_dict(obs_factor_dict, val_seq, x_min):
    N = len(val_seq)
    move_factor_dict = obs_factor_dict.copy()
    factor_list, _ = zip(*move_factor_dict.items())
    factor_list = list(factor_list)
    factor_list.sort(reverse=True)
    
    num_dist_factors = len(factor_list)
    min_admiss_factor_ind_list = np.zeros(N).astype(int)
    val_fix_list = [1]*N
    for ii in range(N):
        deg = val_seq[ii]
        factorint_deg = factorint(deg)
        bin_count_list = [factorint_deg.get(factor) for factor in factor_list]
        bin_count_list = [0 if count is None else count for count in bin_count_list]
        current_factor_ind = next((i for i, x in enumerate(bin_count_list) if x), None)
        current_factor = factor_list[current_factor_ind]
        min_admiss_factor_ind_list[ii] = current_factor_ind
        val_fix = current_factor
        bin_count_list[current_factor_ind] -= 1
        move_factor_dict[current_factor] -= 1
        while val_fix < x_min:
            current_factor_ind = next((i for i, x in enumerate(bin_count_list) if x), None)
            current_factor = factor_list[current_factor_ind]
            min_admiss_factor_ind_list[ii] = current_factor_ind
            val_fix = val_fix*current_factor
            bin_count_list[current_factor_ind] -= 1
            move_factor_dict[current_factor] -= 1
        val_fix_list[ii] = val_fix
        min_admiss_factor_ind = next((i for i, x in enumerate(bin_count_list) if x), None)
    return (move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list)
#
# Generate constrained surrogate with x_min = 1 (via random integer composition):
def gen_cons_power_law_surr_1(val_seq, prime_factor_count_dict):
    len_val_seq = len(val_seq)
    surr_val_seq = np.ones(len_val_seq, dtype = np.int64)
    prime_factor_count_list = list(prime_factor_count_dict.items())
    for factor, count in prime_factor_count_list:
        power_list = rand_comp(count, len_val_seq)
        surr_val_seq = np.multiply(surr_val_seq, np.power(factor, power_list))
    surr_val_seq = surr_val_seq.tolist()
    return surr_val_seq
#
# Generate constrained surrogate for any x_min >= 1 (via random integer composition):
def gen_cons_power_law_surr_2(move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list):
    len_val_seq = len(val_fix_list)
    surr_val_seq = val_fix_list
    prime_factor_count_list = list(move_factor_dict.items())
    for factor_ind in range(len(factor_list)):
        factor = factor_list[factor_ind]
        count = move_factor_dict[factor]
        admitt_val_seq_ind = [ii_ind for ii_ind in range(len_val_seq) if (factor_ind >= min_admiss_factor_ind_list[ii_ind])]
        len_admitt_val_seq = len(admitt_val_seq_ind)
        power_list_admitt_deg = rand_comp(count, len_admitt_val_seq)
        power_list_full = np.zeros(len_val_seq, dtype = np.int64)
        for jj_admitt in range(len_admitt_val_seq):
            power_list_full[admitt_val_seq_ind[jj_admitt]] = power_list_admitt_deg[jj_admitt]
        surr_val_seq = np.multiply(surr_val_seq, np.power(factor, power_list_full))
    surr_val_seq = surr_val_seq.tolist()
    return surr_val_seq
#
#
# Randomly split a product of two values as a factor and its quotient without violating user-specified upper and lower limits
# Use when the two values *are not* part of the same ordinal pattern of the given length
def change_vals(from_val, to_val, from_low_lim, to_low_lim, from_upp_lim, to_upp_lim, from_div=[]):
    
    product = from_val*to_val
    from_factor_dict = factorint(product)
    if from_factor_dict:
        factor_list = list(all_factors(from_factor_dict))
    else:
        factor_list = [1]
        return from_val, to_val
    factor_list = np.array(factor_list)
    if not (factor_list.ndim == 1):#For debugging
        print(factor_list)
    quotient_list = factor_list.copy()
    quotient_list = np.flip(quotient_list)
    
    factor_list = factor_list[\
                        (factor_list >= from_low_lim) & \
                        (factor_list <= from_upp_lim) & \
                        (quotient_list >= to_low_lim) & \
                        (quotient_list <= to_upp_lim)]

    factor = random.choice(factor_list)
    from_val = int(round(factor))
    to_val = int(round(product/factor))
    
    return from_val, to_val
#
#
# Randomly split a product of two values as a factor and its quotient without violating user-specified upper and lower limits or changing the ranking of the two values
# Use when the two values *are* part of the same ordinal pattern of the given length
def change_adj_vals_preserve_op(from_val, to_val, from_low_lim, to_low_lim, from_upp_lim, to_upp_lim, from_div=[]):
    
    product = from_val*to_val
    from_factor_dict = factorint(product)
    if from_factor_dict:
        factor_list = list(all_factors(from_factor_dict))
    else:
        factor_list = [1]
        return from_val, to_val
    factor_list = np.array(factor_list)
    if not (factor_list.ndim == 1):#For debugging
        print(factor_list)
    quotient_list = factor_list.copy()
    quotient_list = np.flip(quotient_list)
    
    if (from_val < to_val):
        adj_cond = (factor_list < quotient_list)
    elif (from_val == to_val):
        adj_cond = (factor_list == quotient_list)
        return from_val, to_val
    else:
        adj_cond = (factor_list > quotient_list)
    
    factor_list = factor_list[\
                        adj_cond & \
                        (factor_list >= from_low_lim) & \
                        (factor_list <= from_upp_lim) & \
                        (quotient_list >= to_low_lim) & \
                        (quotient_list <= to_upp_lim)]

    factor = random.choice(factor_list)
    from_val = int(round(factor))
    to_val = int(round(product/factor))
    
    return from_val, to_val
#
#
# Generate constrained Markov power-law surrogate (via Metropolis-Hastings):
def gen_p_l_markov_metr_hast(val_seq, low_lim_list, upp_lim_list, num_trans):
    len_val_seq = len(val_seq)
    surr_val_seq = [int(round(val)) for val in val_seq]
    list_of_bins = [ii for ii in range(len_val_seq)]
    t = 0
    while (t < num_trans):
        t += 1
        from_to_bins = random.sample(list_of_bins, 2)
        if (from_to_bins[0] > from_to_bins[1]):
            from_to_bins = [from_to_bins[1], from_to_bins[0]]
        from_bin = from_to_bins[0]
        to_bin = from_to_bins[1]
        from_val = surr_val_seq[from_bin]
        to_val = surr_val_seq[to_bin]
        from_low_lim = low_lim_list[from_bin]
        to_low_lim = low_lim_list[to_bin]
        from_upp_lim = upp_lim_list[from_bin]
        to_upp_lim = upp_lim_list[to_bin]
        from_val, to_val = change_vals(from_val, to_val, from_low_lim, to_low_lim, from_upp_lim, to_upp_lim)
        surr_val_seq[from_bin] = from_val
        surr_val_seq[to_bin] = to_val
    return surr_val_seq
# 
# Randomly reassign values of one sequence such that sequences of bins does not change
def reassign_vals(surr_val_seq0, bin_seq0, bin_seq):
    unique_bin_list = np.unique(bin_seq, return_index=False, return_inverse=False, return_counts=False)
    unique_bin_list = np.array(unique_bin_list, dtype = np.int64)
    bin_seq0 = np.array(bin_seq0, dtype = np.int64)
    bin_seq = np.array(bin_seq, dtype = np.int64)
    surr_val_seq0 = np.array(surr_val_seq0, dtype = np.int64)
    surr_val_seq = np.array(surr_val_seq0, dtype = np.int64)
    for unique_bin in unique_bin_list:
        surr_val_seq[bin_seq == unique_bin] = np.random.permutation(surr_val_seq0[bin_seq0 == unique_bin])
    surr_val_seq = [int(round(val)) for val in surr_val_seq]
    return surr_val_seq
# 
# Generate surrogate constrained both in likelihood under a power-law model and in transitions between states
def gen_mc_pl_surrogate(val_seq, bin_edges, num_trans, o, n_surr):
    surr_val_seq0 = [int(round(val)) for val in val_seq]
    bin_seq, low_lim_list, upp_lim_list = arb_bin(val_seq, bin_edges)
    unique_bin_list, bin_seq0 = np.unique(bin_seq, return_index=False, return_inverse=True, return_counts=False)# Replace sequence of bin numbers with consecutive integers starting with 0
    surr_val_seq_list = []
    for ii in range(n_surr):
        bin_seq = gen_mc_surrogate(bin_seq0, o, 1)[0]
        surr_val_seq0 = gen_p_l_markov_metr_hast(surr_val_seq0, low_lim_list, upp_lim_list, num_trans)
        surr_val_seq = reassign_vals(surr_val_seq0, bin_seq0, bin_seq)
        surr_val_seq_list = surr_val_seq_list + [surr_val_seq]
    return surr_val_seq_list
# 
#
def opL(seq, L=2):#Work out ordinal patterns of length L (for checking conservation of this property)
    N = len(seq)
    return [list(rankdata(seq[i:i+L])) for i in range(N - L + 1)]
#
#
# Generate Markov power-law surrogate with constrained ordinal patterns of length L (via Metropolis-Hastings) allowing choice within the same ordinal pattern:
def gen_p_l_markov_o_p_metr_hast_L(val_seq, num_trans, x_min, L):
    N = len(val_seq)
    surr_val_seq = [int(round(val)) for val in val_seq]
    list_of_bins = [ii for ii in range(N)]
    t = 0
    while (t < num_trans):
        t += 1
        from_to_bins = random.sample(list_of_bins, 2)
        if (from_to_bins[0] > from_to_bins[1]):#Ensure that j > i
            from_to_bins = [from_to_bins[1], from_to_bins[0]]
        i = from_to_bins[0]
        j = from_to_bins[1]
        x_i = surr_val_seq[i]
        x_j = surr_val_seq[j]
        if not (x_i == x_j):#If x_i == x_j then no change in values is possible
            neighb_i = surr_val_seq[max(0, i - L + 1):i] + surr_val_seq[(i + 1):min(N, i + L)]
            neighb_j = surr_val_seq[max(0, j - L + 1):j] + surr_val_seq[(j + 1):min(N, j + L)]
            low_lim_i = max([x_min] + [x + 1 for x in neighb_i if x < x_i])
            low_lim_j = max([x_min] + [x + 1 for x in neighb_j if x < x_j])
            upp_lim_i = min([+np.Inf] + [x - 1 for x in neighb_i if x > x_i])
            upp_lim_j = min([+np.Inf] + [x - 1 for x in neighb_j if x > x_j])
            if not (x_i in neighb_i) and not (x_j in neighb_j) and not (low_lim_i == upp_lim_i) and not (low_lim_j == upp_lim_j):#If x_i or x_j have the same value as an element of the same pattern then no change is possible
                if (abs(j - i) >= L):#i and j are not part of the same ordinal pattern
                    x_i, x_j = change_vals(x_i, x_j, low_lim_i, low_lim_j, upp_lim_i, upp_lim_j)
                    surr_val_seq[i] = x_i
                    surr_val_seq[j] = x_j
                else:#j- i < L
                    x_i, x_j = change_adj_vals_preserve_op(x_i, x_j, low_lim_i, low_lim_j, upp_lim_i, upp_lim_j)
                    surr_val_seq[i] = x_i
                    surr_val_seq[j] = x_j  
    return surr_val_seq
#
#
#
#
# Code for producing AAFT, IAAFT and STAP surrogates by calling the MATLAB functions AAFTsur, IAAFTsur and STAPsur of the toolkit Measures of Analysis of Time Series (MATS):
# http://www.jstatsoft.org/v33/i05/  
# Kugiumtzis, Dimitris, and Alkiviadis Tsimpiris. "Measures of analysis of time series (MATS): a MATLAB toolkit for computation of multiple measures on time series data bases." Journal of Statistical Software, 33(5), 1 - 30. doi:http://dx.doi.org/10.18637/jss.v033.i05
#
# This depends on both MATLAB and Python version. I am using MATLAB R2020a and Python 3.7.7.
#
# The MATLAB functions AAFTsur, IAAFTsur and STAPsur will need to be on MATLAB's path
# 
#
try:
    import matlab.engine
    eng = matlab.engine.start_matlab()
    use_matlab = True
except:
    use_matlab = False

def nan_seq(val_seq, nsur):#When we cannot use MATLAB, just return sequences of Numpy nans
    surr_val_seq_list = [[np.nan for val in val_seq] for isur in range(nsur)]
    return surr_val_seq_list

def aaft(val_seq, nsur):
    if use_matlab:
        xV = matlab.double(val_seq)
        zM = eng.AAFTsur(xV, nsur)
        zM = eng.transpose(zM)
        surr_val_seq_list = np.array(zM._data).reshape(zM.size, order='F')
        surr_val_seq_list = [[int(np.round(surr_val)) for surr_val in surr_val_seq] for surr_val_seq in surr_val_seq_list]
    else:
        surr_val_seq_list = nan_seq(val_seq, nsur)
    return surr_val_seq_list

def iaaft(val_seq, nsur):
    if use_matlab:
        xV = matlab.double(val_seq)
        zM = eng.IAAFTsur(xV, nsur)
        zM = eng.transpose(zM)
        surr_val_seq_list = np.array(zM._data).reshape(zM.size, order='F')
        surr_val_seq_list = [[int(np.round(surr_val)) for surr_val in surr_val_seq] for surr_val_seq in surr_val_seq_list]
    else:
        surr_val_seq_list = nan_seq(val_seq, nsur)
    return surr_val_seq_list

def stap(val_seq, nsur):
    if use_matlab:
        N = len(val_seq)
        xV = matlab.double(val_seq)
        pol_arm = min(5, float(int(N/3)))
        pol = pol_arm# degree of the polynomial to approximate the sample transform (set to 5 in Fig. 1 of Statically transformed autoregressive process and surrogate data test for nonlinearity)
        arm = pol_arm# order of the AR model to generate the Gaussian time series (set to 5 in Fig. 1 of Statically transformed autoregressive process and surrogate data test for nonlinearity)
        output = eng.STAPsur(xV, pol, arm, nsur, nargout=2)
        zM = output[0]
        errmsg = output[1]
        # print(errmsg)
        zM = eng.transpose(zM)
        surr_val_seq_list = np.array(zM._data).reshape(zM.size, order='F')
        surr_val_seq_list = [[int(np.round(surr_val)) for surr_val in surr_val_seq] for surr_val_seq in surr_val_seq_list]
    else:
        surr_val_seq_list = nan_seq(val_seq, nsur)
    return surr_val_seq_list

# # Quick test:
# val_seq = [1, 4, 3, 2, 2, 4, 5, 1, 6]
# print(aaft(val_seq, 3))
# print(iaaft(val_seq, 3))
# print(stap(val_seq, 3))
# 
# 
# 
# 
# # Generate different kinds of surrogates with a single function
# 
def gen_power_law_surr_list(seq, surr_method='cons', x_min=1, num_surr=1, scale_exp=2.5, b=3, o=1, L=2, num_trans=10**5):#Generate a list of power-law surrogates of the chosen type
    
    N = len(seq)
    
    if (surr_method == 'obse'):#Observed sequence
        surr_list = [seq.copy() for ii in range(num_surr)]
    
    elif (surr_method == 'shuf'):#Shuffle surrogates
        def make_new_shuffled_list(seq):
            new_seq = seq.copy()
            random.shuffle(new_seq)
            return new_seq
        surr_list = [make_new_shuffled_list(seq) for ii in range(num_surr)]
        
    elif (surr_method == 'know'):#Known scale exponent
        surr_list = [gen_typ_p_l_surrogate(scale_exp, x_min, N) for ii in range(num_surr)]
    
    elif (surr_method == 'boot'):#Bootstrap
        surr_list = [bootstrap(seq, N) for ii in range(num_surr)]
    
    elif (surr_method == 'typi'):#Typical surrogates
        m_l_scale_exp = estim_scale_exp(seq, x_min)#Maximum likelihood scale exponent
        surr_list = [gen_typ_p_l_surrogate(m_l_scale_exp, x_min, N) for ii in range(num_surr)]
    
    elif (surr_method == 'mark'):#Constrained Markov order o power-law surrogates using bins of log width 3
        bin_edges, bin_seq, low_lim_list, upp_lim_list = log_bin(seq, b, x_min)
        surr_list = gen_mc_pl_surrogate(seq, bin_edges, num_trans, o, num_surr)
        
    elif (surr_method == 'ordi'):#Constrained ordinal pattern power-law surrogates using ordinal patterns of length L
        surr_list = []
        for ii in range(num_surr):
            surr_seq = gen_p_l_markov_o_p_metr_hast_L(seq, num_trans, x_min, L)
            surr_list = surr_list + [surr_seq]
        
    elif (surr_method == 'aaft'):#Amplitude adjusted Fourier transform surrogates
        surr_list = aaft(seq, num_surr)
        
    elif (surr_method == 'iaaft'):#Iterated amplitude adjusted Fourier transform surrogates
        surr_list = iaaft(seq, num_surr)
            
    else:#Constrained surrogate
        if not (surr_method == 'cons'):
            print('Input surr_method=' + str(surr_method) + ' not recognised. Generating constrained surrogate(s).')
        surr_list = []
        obs_factor_dict = build_factor_count_dict(seq)
        if (x_min == 1):
            for ii in range(num_surr):
                surr_seq = gen_cons_power_law_surr_1(seq, obs_factor_dict)
                surr_factor_dict = build_factor_count_dict(surr_seq)
                while not operator.eq(obs_factor_dict, surr_factor_dict):
                    set1 = set(obs_factor_dict.items())
                    set2 = set(surr_factor_dict.items())
                    print('There is a problem: prime factor counts of surrogate do not match those of original. Difference:')
                    print(set1 ^ set2)
                    print('We will generate a replacement surrogate.')
                    surr_seq = gen_cons_power_law_surr_1(seq, obs_factor_dict)
                    surr_factor_dict = build_factor_count_dict(surr_seq)
                surr_list = surr_list + [surr_seq]
        else:
            (move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list) = make_move_factor_dict(obs_factor_dict, seq, x_min)
            for ii in range(num_surr):
                surr_seq = gen_cons_power_law_surr_2(move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list)
                surr_factor_dict = build_factor_count_dict(surr_seq)
                while not operator.eq(obs_factor_dict, surr_factor_dict):
                    set1 = set(obs_factor_dict.items())
                    set2 = set(surr_factor_dict.items())
                    print('There is a problem: prime factor counts of surrogate do not match those of original. Difference:')
                    print(set1 ^ set2)
                    print('We will generate a replacement surrogate.')
                    surr_seq = gen_cons_power_law_surr_2(move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list)
                    surr_factor_dict = build_factor_count_dict(surr_seq)
                surr_list = surr_list + [surr_seq]
        
    return surr_list
# 
# 
# 
# 
# 
# 
# 
# 
# Functions for generating results
# 
# 
#
# Generate sequences related to the asymmetric tent map
# 
# Generate deterministic sequence from tent map:
def asymm_tent_map(N, x0, a):#N numbers which are successive iterates of the logistic map with parameter r, starting from x0.
    x_seq = [x0]
    x = x0
    for ii in range(1, N):
        if (x <= a):
            x = x/a
        else:
            x = (1 - x)/(1 - a)
        x_seq = x_seq + [x]
    return x_seq

# Convert from probabilities p \in [0, 1] (uniform on [0, 1]) to integer from Zipf distribution:
def cdf_vals_to_zipf_rv(p_seq, gamma):#Convert from a value from the cdf of a Zipf distribution (supported on [1, +inf)]) cdf value to a Zipf realisation
    val_seq = zipf.ppf(p_seq, gamma)
    val_seq = [int(deg) for deg in val_seq]
    return val_seq

# Deterministic sequence of single iterates of asymmetric tent map with parameter a observed after a transformation to integers from zeta distribution
def asymm_tent_map_power_law(N, gamma, min_deg, a=0.90):#Power-law transform of N iterates of the logistic map
    #a = 0.95#Logistic map parameter
    x0 = random.random()#Randomly choose the start point of asymmetric tent map recursion according to its limiting distribution - the stationary distribution of the logistic map with r = 4 is a beta distribution
    x_seq = asymm_tent_map(N, x0, a)
    p_min_deg_m_1 = zipf.cdf(min_deg - 1, gamma)#p-values less than this would violate x >= min_deg
    p_seq = np.array(x_seq)*(1 - p_min_deg_m_1) + p_min_deg_m_1#Linearly shift p-values so that they satisfy x >= min_deg
    val_seq = cdf_vals_to_zipf_rv(p_seq, gamma)
    val_seq = [deg if deg >= min_deg else min_deg for deg in val_seq]#Numerical error can cause a few values x for which x < min_deg; correct these
    return val_seq
# 
# 
# 
# 
# Generate data or load an empirical time series
# 
def gen_orig(dist_code=0, gamma=2.5, x_min=1, N=1024, zero_for_size_one_for_power=0):
    if (dist_code == 10):#Discretised truncated log-normal distribution with mu = 0
        sigma = gamma
        mu = 0
        log_val_seq = truncnorm.rvs((np.log(x_min - 0.5) - mu)/sigma, (np.inf - mu)/sigma, loc=mu, scale=sigma, size = N)
        val_seq = [int(round(np.exp(log_deg))) for log_deg in log_val_seq]
        dist_str = 'l_n_c_dir';
    elif (dist_code == 17):#Power law with exponential cutoff defined by lam = Lambda = lambda = 0.01
        val_seq = randht(N, 'xmin', x_min - 0.5, 'cutoff', gamma, 0.01)
        val_seq = [int(np.round(deg)) for deg in val_seq]
        dist_str = 'pow_law_cuto_2';
    elif (dist_code == 21.5):#Power law truncated at b = 64
        b = 64#When gamma = 2.5 and xMin = 1, the 1/1024 quantile is at k = 63.2388
        stat_dist = truncated_power_law(gamma, x_min, b)
        val_seq = stat_dist.rvs(size=N)
        val_seq = [int(np.round(deg)) for deg in val_seq]
        dist_str = 'trunc_p_l_064';
    elif (dist_code == 22):#Power law truncated at xMax = 184
        b = 184#When gamma = 2.5 and xMin = 12, the 1/64 quantile is at k = 184.644
        stat_dist = truncated_power_law(gamma, x_min, b)
        val_seq = stat_dist.rvs(size=N)
        val_seq = [int(np.round(deg)) for deg in val_seq]
        dist_str = 'trunc_p_l_184';
    elif (dist_code == 25):#Markov order 1 power-law with at least 90% chance of next value being in same log-bin as current value
        b = 3
        mu = 0.1
        val_seq = gen_mc_pl_seq(N, gamma, x_min, b, mu, o=1)
        dist_str = 'c9r_pl_bin_3';
    elif (dist_code == 26):#Markov order 2 power-law with at least 81% chance of proposed value being in same log-bin as current value and  at least 81% chance of proposed value being in same log-bin as preceding value
        b = 3
        mu = 0.1
        val_seq = gen_mc_pl_seq(N, gamma, x_min, b, mu, o=2)
        dist_str = 'c92_pl_bin_3';
    elif (dist_code == 35):#Extract a random sequence of N discretised energies for earthquakes with magnitude at least two in Southern California
        int_energy_list = list(np.loadtxt('../' + ts_folder_name + '/' + 'energy.txt', dtype='int'))
        num_events = len(int_energy_list)
        start_point = random.randint(0, num_events - N)
        val_seq = int_energy_list[start_point:(start_point + N)]
        dist_str = 'earthquakes';
    elif (dist_code == 57):#Extract a random sequence of length N
        file_name_str = 'tail-rescaled-diseases.txt'; dist_str = 'diseases-rescaled-tail'; #x_min = 2317, N = 27, NG = 27
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(np.round(val)) for val in data if val >= x_min]
        num_data = len(val_seq)
        start_point = random.randint(0, num_data - N)
        val_seq = val_seq[start_point:(start_point + N)]
    elif (dist_code == 58):#Extract a random sequence of length N
        file_name_str = 'tail-thousand-blackouts.txt'; dist_str = 'blackouts-tail'; #x_min = 235;, N = 57, NG = 57
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(np.round(val)) for val in data if val >= x_min]
        num_data = len(val_seq)
        start_point = random.randint(0, num_data - N)
        val_seq = val_seq[start_point:(start_point + N)]
    elif (dist_code == 59):#Extract a random sequence of length N
        file_name_str = 'normed-flares.txt'; dist_str = 'flares-normed'; #x_min = 1;, N = 1711, NG = 1711
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(np.round(val)) for val in data if val >= x_min]
        num_data = len(val_seq)
        start_point = random.randint(0, num_data - N)
        val_seq = val_seq[start_point:(start_point + N)]
    elif (dist_code == 93):#Discrete power-law observation of asymmetric tent map with a = 0.95
        val_seq = asymm_tent_map_power_law(N, gamma, x_min, a=0.95)
        dist_str = 'asym_tent_pl_a-0-95';
    elif (dist_code == 94):#Discrete power-law observation of asymmetric tent map with a = 0.98
        val_seq = asymm_tent_map_power_law(N, gamma, x_min, a=0.98)
        dist_str = 'asym_tent_pl_a-0-98';
    elif (dist_code == 95):#Discrete power-law observation of asymmetric tent map with a = 0.99
        val_seq = asymm_tent_map_power_law(N, gamma, x_min, a=0.99)
        dist_str = 'asym_tent_pl_a-0-99';
    else:
        val_seq = gen_typ_p_l_surrogate(gamma, x_min, N)
        dist_str = 'disc_zeta';
    return (val_seq, dist_str)
# 
# 
# 
# 
# Calculate sample statistics
# 
stat_str_dict = {0:'ks_dist', 1:'var_deg', 2:'mean_de', 3:'max_deg', 4:'con_e_1', 20:'coef_var', 21:'co_en_o2', 33:'vmr', 35:'r1_r2_rat'}
stat_factor_dict = {0:+1, 1:-1, 2:-1, 3:-1, 4:-1, 20:-1, 21:-1, 33:-1, 35:-1}#-1: smaller is more extreme, +1: larger is more extreme
def stat_fun_0(val_seq, x_min, gamma):
    return calc_ks_stat(val_seq, x_min, gamma)
def stat_fun_1(val_seq, x_min, gamma):
    return np.var(val_seq);
def stat_fun_2(val_seq, x_min, gamma):
    return np.mean(val_seq);
def stat_fun_3(val_seq, x_min, gamma):
    return np.max(val_seq);
def stat_fun_4(val_seq, x_min, gamma):
    u, val_seq = np.unique(val_seq, return_inverse=True)#This is here because cond_entropy seemed sometimes to have trouble with very large integers.
    val_seq = list(val_seq)
    return cond_entropy(val_seq[1:], val_seq[:-1])# cond_entropy(Y, X) is H(Y|X)
def stat_fun_20(val_seq, x_min, gamma):#Coefficient of variation (using population standard deviation)
    std_dev = np.sqrt(np.var(val_seq))
    mean_val = np.mean(val_seq)
    return (std_dev/mean_val);
def stat_fun_21(val_seq, x_min, gamma):#Conditional entropy of order 2
    u, val_seq = np.unique(val_seq, return_inverse=True)#This is here because cond_entropy seemed sometimes to have trouble with very large integers.
    return cond_entropy_o_n(val_seq, 2)
def stat_fun_33(val_seq, x_min, gamma):#Variance mean ratio
    return np.var(val_seq)/np.mean(val_seq);
def stat_fun_35(val_seq, x_min, gamma):#Ratio of largest value to second largest value
    max_val = np.max(val_seq)
    copy_val_seq = val_seq.copy()
    copy_val_seq.remove(max_val)
    return max_val/np.max(copy_val_seq);

stat_fun_dict = {0:stat_fun_0, 1:stat_fun_1, 2:stat_fun_2, 3:stat_fun_3, 4:stat_fun_4, 20:stat_fun_20, 21:stat_fun_21, 33:stat_fun_33, 35:stat_fun_35}

# stat_fun_a_0 is omitted mainly because it might be inconvenient to recalculate the maximum likelihood scale exponent
def stat_fun_a_1(val_seq):
    return np.var(val_seq);
def stat_fun_a_2(val_seq):
    return np.mean(val_seq);
def stat_fun_a_3(val_seq):
    return np.max(val_seq);
def stat_fun_a_4(val_seq):
    u, val_seq = np.unique(val_seq, return_inverse=True)#This is here because cond_entropy seemed sometimes to have trouble with very large integers.
    val_seq = list(val_seq)
    return cond_entropy(val_seq[1:], val_seq[:-1])# cond_entropy(Y, X) is H(Y|X)
def stat_fun_a_20(val_seq):#Coefficient of variation
    std_dev = np.sqrt(np.var(val_seq))
    mean_val = np.mean(val_seq)
    return (std_dev/mean_val);
def stat_fun_a_21(val_seq):#Conditional entropy of order 2
    u, val_seq = np.unique(val_seq, return_inverse=True)#This is here because cond_entropy seemed sometimes to have trouble with very large integers.
    return cond_entropy_o_n(val_seq, 2)
def stat_fun_a_33(val_seq):#Variance mean ratio
    return np.var(val_seq)/np.mean(val_seq);
def stat_fun_a_35(val_seq):#Ratio of largest value to second largest value
    max_val = np.max(val_seq)
    copy_val_seq = val_seq.copy()
    copy_val_seq.remove(max_val)
    return max_val/np.max(copy_val_seq);

stat_fun_dict_a = {1:stat_fun_a_1, 2:stat_fun_a_2, 3:stat_fun_a_3, 4:stat_fun_a_4, 20:stat_fun_a_20, 21:stat_fun_a_21, 33:stat_fun_a_33, 35:stat_fun_a_35}

def true_fun_gen(gamma, x_min, N, func_flag):
    if (func_flag == 0):#True power-law
        val_seq = gen_typ_p_l_surrogate(gamma, x_min, N)
    elif (func_flag == 2):#Discretised truncated log-normal
        #Generate iid from a discretised truncated log-normal distibution with log-median mu = 0 and log-variance gamma^2 (truncated to [x_min - 0.5, +inf))
        mu = 0
        log_val_seq = truncnorm.rvs((np.log(x_min - 0.5) - mu)/gamma, (np.inf - mu)/gamma, loc=mu, scale=gamma, size = N)
        val_seq = [int(round(np.exp(log_val))) for log_val in log_val_seq]
    elif (func_flag == 7):#Power law with exponential cut-off
        lam7 = 0.01#Exponential cut-off
        val_seq = randht(N, 'xmin', x_min - 0.5, 'cutoff', gamma, lam7)
        val_seq = [int(np.round(deg)) for deg in val_seq]
    elif (func_flag == 21.5):#Power law truncated at b = 64
        b = 64#When gamma = 2.5 and xMin = 1, the 1/1024 quantile is at k = 63.2388
        stat_dist = truncated_power_law(gamma, x_min, b)
        val_seq = stat_dist.rvs(size=N)
        val_seq = [int(np.round(deg)) for deg in val_seq]
        dist_str = 'trunc_p_l_064';
    elif (func_flag == 51):#Upsample or downsample, as required, from data in a text file
        file_name_str = 'terrorism.txt'; dist_str = 'terrorism'; #x_min = 12;#+/-4, N = 9,101, NG = 547
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(val) for val in data if val >= x_min]
        val_seq = subsample(val_seq, N)#Up or downsample, as required
    elif (func_flag == 52):#Upsample or downsample, as required, from data in a text file
        file_name_str = 'words.txt'; dist_str = 'words'; #x_min = 7;#+/-2, N = 18,855, NG = 958
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(val) for val in data if val >= x_min]
        val_seq = subsample(val_seq, N)#Up or downsample, as required
    elif (func_flag == 53):#Upsample or downsample, as required, from data in a text file
        file_name_str = 'rescaled-diseases.txt'; dist_str = 'diseases-rescaled'; #x_min = 2317, N = 72, NG = 27
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(val) for val in data if val >= x_min]
        val_seq = subsample(val_seq, N)#Up or downsample, as required
    elif (func_flag == 55):#Upsample or downsample, as required, from data in a text file
        file_name_str = 'blackouts.txt'; dist_str = 'blackouts'; #x_min = 230;#+/-90, N = 211, NG = 57
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        data = [round(val/1000) for val in data];
        val_seq = [int(val) for val in data if val >= x_min]
        val_seq = subsample(val_seq, N)#Up or downsample, as required
    elif (func_flag == 56):#Upsample or downsample, as required, from data in a text file
        file_name_str = 'flares.txt'; dist_str = 'flares'; #x_min = 323;#N = 12773, NG = 1711
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(val) for val in data if val >= x_min]
        val_seq = subsample(val_seq, N)#Up or downsample, as required
    elif (func_flag == 57):#Upsample or downsample, as required, from data in a text file
        file_name_str = 'tail-rescaled-diseases.txt'; dist_str = 'diseases-rescaled-tail'; #x_min = 2317, N = 27, NG = 27
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(val) for val in data if val >= x_min]
        val_seq = subsample(val_seq, N)#Up or downsample, as required
    elif (func_flag == 58):#Upsample or downsample, as required, from data in a text file
        file_name_str = 'tail-thousand-blackouts.txt'; dist_str = 'blackouts-tail'; #x_min = 235;, N = 57, NG = 57
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(val) for val in data if val >= x_min]
        val_seq = subsample(val_seq, N)#Up or downsample, as required
    elif (func_flag == 59):#Upsample or downsample, as required, from data in a text file
        file_name_str = 'normed-flares.txt'; dist_str = 'flares-normed'; #x_min = 1;, N = 1711, NG = 1711
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(val) for val in data if val >= x_min]
        val_seq = subsample(val_seq, N)#Up or downsample, as required
    elif (func_flag == 62):#Upsample or downsample, as required, from data in a text file
        file_name_str = 'energy.txt';#x_min = 1; N = 59555, NG = 59555
        data = np.loadtxt('../' + ts_folder_name + '/' + file_name_str)
        val_seq = [int(val) for val in data if val >= x_min]
        val_seq = subsample(val_seq, N)#Up or downsample, as required
    return val_seq
# 
# 
# 
# 
# # Calculate empirical distribution of sample statistics
#
num_surr_types_eff = 5
# 
# Consider five types of surrogates/methods of generating sequences - true, typical, constrained, observed and bootstrapped
def calc_val_eff(gamma, stat_fun_dict, stat_code_list, x_min, N, func_flag, num_surr):
    obs_seq = true_fun_gen(gamma, x_min, N, func_flag)
    gamma_m_l = estim_scale_exp(obs_seq, x_min)
    
    num_stats = len(stat_code_list)
    
    # Independent realisations under the true distribution:
    val_list_list_true = [[stat_fun_dict[stat_code](true_fun_gen(gamma, x_min, N, func_flag)) for stat_code in stat_code_list] for ii in range(num_surr)]
    # Typical power-law:
    val_list_list_typi = [[stat_fun_dict[stat_code](gen_typ_p_l_surrogate(gamma_m_l, x_min, N)) for stat_code in stat_code_list] for ii in range(num_surr)]
    # Constrained power-law:
    val_list_list_cond = []
    # Exact reproductions of observed sequence:
    val_list_list_obse = [[stat_fun_dict[stat_code](obs_seq) for stat_code in stat_code_list]]*num_surr
    # Bootstrap:
    val_list_list_boot = [[stat_fun_dict[stat_code](bootstrap(obs_seq)) for stat_code in stat_code_list] for ii in range(num_surr)]

    obs_factor_dict = build_factor_count_dict(obs_seq)
    if (x_min == 1):
        for ii in range(num_surr):
            surr_val_seq = gen_cons_power_law_surr_1(obs_seq, obs_factor_dict)
            surr_factor_dict = build_factor_count_dict(surr_val_seq)
            while not operator.eq(obs_factor_dict, surr_factor_dict):
                set1 = set(obs_factor_dict.items())
                set2 = set(surr_factor_dict.items())
                print('There is a problem: prime factor counts of surrogate do not match those of original. Difference:')
                print(set1 ^ set2)
                print('We will generate a replacement surrogate.')
                surr_val_seq = gen_cons_power_law_surr_1(obs_seq, obs_factor_dict)
                surr_factor_dict = build_factor_count_dict(surr_val_seq)
            val_list_list_cond = val_list_list_cond + [[stat_fun_dict[stat_code](surr_val_seq) for stat_code in stat_code_list]]
    else:
        (move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list) = make_move_factor_dict(obs_factor_dict, obs_seq, x_min)
        for ii in range(num_surr):
            surr_val_seq = gen_cons_power_law_surr_2(move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list)
            surr_factor_dict = build_factor_count_dict(surr_val_seq)
            while not operator.eq(obs_factor_dict, surr_factor_dict):
                set1 = set(obs_factor_dict.items())
                set2 = set(surr_factor_dict.items())
                print('There is a problem: prime factor counts of surrogate do not match those of original. Difference:')
                print(set1 ^ set2)
                print('We will generate a replacement surrogate.')
                surr_val_seq = gen_cons_power_law_surr_2(move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list)
                surr_factor_dict = build_factor_count_dict(surr_val_seq)
            val_list_list_cond = val_list_list_cond + [[stat_fun_dict[stat_code](surr_val_seq) for stat_code in stat_code_list]]
    
    val_list_list_list = [\
                          val_list_list_true,\
                          val_list_list_typi,\
                          val_list_list_cond,\
                          val_list_list_obse,\
                          val_list_list_boot,\
                         ]
    
    val_list_list_list = np.array(val_list_list_list)#[num_surr_types, num_surr, num_stats]
    
    return val_list_list_list
# 
# Consider five types of surrogates/methods of generating sequences - true, observed, and three types of surrogates from the Measures of analysis of time series (MATS) MATLAB toolbox
def calc_val_eff_mats(gamma, stat_fun_dict, stat_code_list, x_min, N, func_flag, num_surr):
    obs_seq = true_fun_gen(gamma, x_min, N, func_flag)
    
    num_stats = len(stat_code_list)
    
    # Independent realisations under the true distribution:
    val_list_list_true = [[stat_fun_dict[stat_code](true_fun_gen(gamma, x_min, N, func_flag)) for stat_code in stat_code_list] for ii in range(num_surr)]
    # AAFT:
    aaft_surr_seq_list = aaft(obs_seq, num_surr)
    val_list_list_aaft = [[stat_fun_dict[stat_code](aaft_surr_seq) for stat_code in stat_code_list] for aaft_surr_seq in aaft_surr_seq_list]
    # IAAFT:
    iaaft_surr_seq_list = iaaft(obs_seq, num_surr)
    val_list_list_iaaft = [[stat_fun_dict[stat_code](iaaft_surr_seq) for stat_code in stat_code_list] for iaaft_surr_seq in iaaft_surr_seq_list]
    # Exact reproductions of observed sequence:
    val_list_list_obse = [[stat_fun_dict[stat_code](obs_seq) for stat_code in stat_code_list]]*num_surr
    ## STAP:
    #stap_surr_seq_list = stap(obs_seq, num_surr)
    #val_list_list_stap = [[stat_fun_dict[stat_code](stap_surr_seq) for stat_code in stat_code_list] for stap_surr_seq in stap_surr_seq_list]
    #Occasionally the STAP surrogate had an error, so do not try to use it here:
    val_list_list_stap = [[np.nan for stat_code in stat_code_list]]*num_surr
    
    val_list_list_list = [\
                          val_list_list_true,\
                          val_list_list_aaft,\
                          val_list_list_iaaft,\
                          val_list_list_obse,\
                          val_list_list_stap,\
                         ]
    
    val_list_list_list = np.array(val_list_list_list)#[num_surr_types, num_surr, num_stats]
    
    return val_list_list_list
#
#
# Make plots to compare original sequences with surrogates:
def plot_surr_seq(surr_seq, obs_seq=[], method_name=[], ax=[], x_label='Time', y_label='Value'):
    N = len(surr_seq)
    if not method_name:
        method_name = 'Surrogate'
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if obs_seq:
        match_seq = [obs_seq[ii] if surr_seq[ii] == obs_seq[ii] else np.nan for ii in range(N)]
        ax.scatter(list(range(N)), obs_seq, marker='o')
        ax.scatter(list(range(N)), surr_seq, marker='o')
        ax.scatter(list(range(N)), match_seq, marker='o')
        leg_labels = ['Observed', method_name, 'Equal']
    else:
        leg_labels = [method_name]
        ax.scatter(list(range(N)), surr_seq, marker='o')
        leg_labels = [method_name]
    ax.set_yscale('log')
    ax.legend(labels=leg_labels, framealpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
#
# Calculate KS-distance relative to maximum likelihood power-law:
def ks_stat(seq):
    x_min = min(seq)
    gamma_m_l = estim_scale_exp(seq, x_min)
    return calc_ks_stat(seq, x_min, gamma_m_l)
#
# Calculate conditional entropy of order one:
def cond_ent_o_1(seq):
    return cond_entropy_o_n(seq, 1)
# 
# Calculate conditional entropy of order two:
def cond_ent_o_2(seq):
    return cond_entropy_o_n(seq, 2)
# 
# Perform hypothesis tests and print output:
def hypothesis_test(obs_seq, method='cons', num_surr = 19, scale_exp=2.5, b=3, o=1, L=2, num_trans=10**5, stat_func_list=[ks_stat]):
    x_min = min(obs_seq)
    surr_seq_list = gen_power_law_surr_list(obs_seq, surr_method=method, x_min=x_min, num_surr=num_surr, scale_exp=scale_exp, b=b, o=o, L=L, num_trans=num_trans)
    quantile_list = []
    p_val_left_tail_list = []
    p_val_right_tail_list = []
    p_val_two_tail_list = []
    print("Using surrogate method: " + str(method) + ':')
    print("Equating lower cut-off parameter *x_min* with minimum value of argument *obs_seq*: x_min*=" + str(x_min))
    for stat_func in stat_func_list:#Iterate over discriminating statistics
        print("\t With statistic: " + stat_func.__name__ + ":")
        #Calculate values of discrimintating statistic for observed sequence and surrogate sequences:
        obs_stat = stat_func(obs_seq)
        stat_val_surr_list = [stat_func(surr_seq) for surr_seq in surr_seq_list]
        #Add small random perturbations to avoid ties:
        abs_obs_stat = abs(obs_stat)
        obs_stat = obs_stat + 10**-6*abs_obs_stat*(np.random.uniform(size=1) - 0.5)
        stat_val_surr_list = [stat_val_surr + 10**-6*abs_obs_stat*(np.random.uniform(size=1) - 0.5) for stat_val_surr in stat_val_surr_list]#Add small random perturbations to avoid ties
        stat_val_surr_list = np.array(stat_val_surr_list)
        #Work out rank of observed statistic (position of statistic corresponding to observed sequence when statistics corresponding to observed sequence and surrogate sequences are ranked from smallest to largest)
        rankMin = 1 + sum(stat_val_surr_list < obs_stat)
        rankMax = 1 + sum(stat_val_surr_list <= obs_stat)
        r = random.randint(rankMin, rankMax)
        q = (r - 0.5)/(num_surr + 1)
        quantile_list = quantile_list + [q]
        p_left = q
        p_val_left_tail_list = p_val_left_tail_list + [p_left]
        p_right = 1 - q
        p_val_right_tail_list = p_val_right_tail_list + [p_right]
        p_two = 2*min([p_left, p_right])
        p_val_two_tail_list = p_val_two_tail_list + [p_two]
        print("\t\t Quantile: " + str(q))
        print("\t\t p-value for left-tailed test: " + str(p_left))
        print("\t\t p-value for right-tailed test: " + str(p_right))
        print("\t\t p-value for two-tailed test: " + str(p_two))
    return quantile_list, p_val_left_tail_list, p_val_right_tail_list, p_val_two_tail_list
