###################################################################
##  Written by Eli Pugh and Ethan Shen                           ##
##  {epugh}, {ezshen} @stanford.edu                              ##
##  Translated from Matlab written by Jiantao Jiao               ##
##  https://github.com/EEthinker/Universal_directed_information  ##
##  Based off of:                                                ##
##    Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman      ##
##    'Universal Estimation of Directed Information'             ##
##    http://arxiv.org/abs/1201.2334                             ##
###################################################################

import numpy as np
import time
from tqdm import tqdm
from ctwalgorithm import ctwalgorithm


# Function 'compute_DI_MI' calculates the directed information I(X^n--> Y^n),
# mutual information I(X^n; Y^n) and reverse directed information I(Y^{n-1}-->X^n)
# for any positive integer n smaller than the length of X and Y.
# Inputs:
# X and Y: two input sequences;
# Nx: the size of alphabet of X, assuming X and Y have the same size of
# alphabets;
# D: the maximum depth of the context tree used in basic CTW algorithm
# alg: indicates one of the four possible estimators proposed in 'Universal Estimation of Directed Information'
# Users can indicate strings 'E1','E2','E3' and 'E4' for corresponding
# estimators.
# start_ratio: indicates how large initial proportion of input data should be ignored when displaying
# the estimated results, for example, if start_ratio = 0.2, then the output DI
# only contains the estimate of I(X^n \to Y^n) for n larger than
# length(X)/5.
def compute_DI_MI(X, Y, Nx, D, start_ratio, prob=None, flag, MI=True):
    XY=X+Nx*Y
    n_data = len(X)

    pxy = ctwalgorithm(XY, Nx**2, D)    # 4x8
    px_xy = np.zeros((Nx,n_data-D))

    for i_x in range(Nx):
        px_xy[i_x,:] = pxy[i_x,:]
        for j in range(1, Nx):
            px_xy[i_x,:] = px_xy[i_x,:] + pxy[i_x+j*Nx,:]

    temp= np.tile(px_xy, (Nx,1))
    py_x_xy = np.divide(pxy, temp)

    # E4
    temp_DI= np.zeros((1,px.shape[1]))
    for iy in range(Nx):
       for ix in range(Nx):
            tmp1 = pxy[ix+iy*Nx,:]
            tmp2 = np.multiply(py[iy,:], px_xy[ix,:])
            temp_DI = temp_DI + np.multiply( tmp1, np.log2(np.divide(tmp1,tmp2)) )
    DI = np.cumsum(temp_DI[int(np.floor(n_data*start_ratio)):])
    return DI


# Function 'compute_mat_px' uses the CTW algorithm to find a universal
# probability assignment for each row of X
# Inputs:
# X: matrix of input sequences
# Nx: Alphabet size of X
# D: Depth of the CTW Algorithm tree
def compute_px(Xn Nx, D):
    Px = []
    for i in tqdm(range(len(X))):
        Px.append(ctwalgorithm(X[i], Nx, D))   # 2x8
    return Px


# Function 'compute_DI_mat' takes in a matrix X and computes pairwise directed
# information between each of the rows of X
# DI[i,j] is the directed information I(X[i]->X[j])
# Inputs:
# X: matrix of input sequences
# Nx: Alphabet size of X
# D: Depth of the CTW Algorithm tree
def compute_DI_mat(X, Nx, D, start_ratio, MI=False):
    X = np.array(X)
    DI = np.zeros((X.shape[0], X.shape[0]))
    Px = compute_mat_px(X, Nx, D)
    for i in tqdm(range(len(X))):
        for j in range(len(X)):
            DI[i][j] = compute_DI_MI(X[i], X[j], Nx, D, start_ratio, prob=[Px[i],Px[j]] MI=MI)[-1]
    return DI
