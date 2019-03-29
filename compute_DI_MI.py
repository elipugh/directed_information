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
from tqdm import tqdm
from ctwalgorithm import ctwalgorithm


# Function 'compute_DI_MI' calculates the directed information I(X^n--> Y^n),
# mutual information I(X^n; Y^n) and reverse directed information I(Y^{n-1}-->X^n)
# for any positive integer n smaller than the length of X and Y.
# Inputs:
# X and Y: two input sequences;
# Nx: the size of alphabet of X, assuming X and Y have the same size of alphabets
# D: the maximum depth of the context tree used in basic CTW algorithm
# alg: indicates one of the four possible estimators proposed in
#   'Universal Estimation of Directed Information'
#   Users can indicate strings 'E1','E2','E3' and 'E4' for corresponding estimators.
# start_ratio: indicates how large initial proportion of input data should be ignored when
#   displaying the estimated results, for example, if start_ratio = 0.2, then the output DI
#   only contains the estimate of I(X^n \to Y^n) for n larger than length(X)/5.
# prob: universal probability assignments, should be an array or tuple (px,py,pxy)
def compute_DI_MI(X, Y, Nx, D, alg, start_ratio, prob=None):
    XY=X+Nx*Y
    n_data = len(X)

    if prob  == None:
        px = ctwalgorithm(X, Nx, D)
        py = ctwalgorithm(Y, Nx, D)
        pxy = ctwalgorithm(XY, Nx**2, D)
    else:
        px, py, pxy = prob

    # px_xy calculates p(x_i|x^{i-1},y^{i-1})
    px_xy = np.zeros((Nx,n_data-D))
    for i_x in range(Nx):
        px_xy[i_x,:] = pxy[i_x,:]
        for j in range(1, Nx):
            px_xy[i_x,:] = px_xy[i_x,:] + pxy[i_x+j*Nx,:]

    # calculate P(y|x,X^{i-1},Y^{i-1})
    temp= np.tile(px_xy, (Nx,1))
    py_x_xy = np.divide(pxy, temp)

    if alg == "E1":
        print("Not yet implemented")

    elif alg == "E2":
        print("Not yet implemented")

    elif alg == "E3":
        print("Not yet implemented")

    elif alg == "E4":
        temp_DI = np.zeros((1,px.shape[1]))
        temp_MI = np.zeros((1,px.shape[1]))
        temp_rev_DI = np.zeros((1,px.shape[1]))
        for iy in range(Nx):
            for ix in range(Nx):
                tmp1 = pxy[ix+iy*Nx,:]
                tmp2 = np.multiply(py[iy,:], px_xy[ix,:])
                temp_DI += np.multiply( tmp1, np.log2(np.divide(tmp1,tmp2)) )
                tmp3 = np.multiply(py[iy,:], px[ix,:])
                temp_MI += np.multiply( tmp1, np.log2(np.divide(tmp1,tmp3)) )
                tmp4 = np.divide( px_xy[ix,:], px[ix,:] )
                temp_rev_DI += np.multiply( tmp1, np.log2(tmp4) )

    else:
        print("Algorithm should be \"E1\", \"E2\", \"E3\", or \"E4\".")
        print(alg, "is not a choice")


    DI = np.cumsum(temp_DI[int(np.floor(n_data*start_ratio)):])
    rev_DI = np.cumsum(temp_rev_DI[int(np.floor(n_data*start_ratio)):])
    # MI = np.cumsum(temp_MI[int(np.floor(n_data*start_ratio)):])
    MI = DI+rev_DI
    return DI, rev_DI, MI


# Function 'compute_mat_px' uses the CTW algorithm to find a universal
# probability assignment for each row of X
# Inputs:
# X: matrix of input sequences
# Nx: Alphabet size of X
# D: Depth of the CTW Algorithm tree
def compute_mat_px(X, Nx, D):
    Px = []
    for i in tqdm(range(len(X))):
        Px.append(ctwalgorithm(X[i], Nx, D))   # 2x8
    return Px


# Function 'compute_mat_pxy' uses the CTW algorithm to find a universal
# probability assignment for each pair of rows of X
# Inputs:
# X: matrix of input sequences
# Nx: Alphabet size of X
# D: Depth of the CTW Algorithm tree
def compute_mat_pxy(X, Nx, D):
    n = len(X)
    Pxy = np.zeros((n,n))
    for i in tqdm(range(n)):
        for j in tqdm(range(n)):
            if i == j:
                continue
            XY=X[i]+Nx*X[j]
            Pxy[i,j] = ctwalgorithm(XY, Nx**2, D)
    return Pxy


# Function 'compute_DI_mat' takes in a matrix X and computes pairwise directed
# information between each of the rows of X
# DI[i,j] is the directed information I(X[i]->X[j])
# Inputs:
# X: matrix of input sequences
# Nx: Alphabet size of X
# D: Depth of the CTW Algorithm tree
def compute_DI_MI_mat(X, Nx, D, start_ratio, MI=False):
    X = np.array(X)
    DI = np.zeros((X.shape[0], X.shape[0]))
    rev_DI = np.zeros((X.shape[0], X.shape[0]))
    MI = np.zeros((X.shape[0], X.shape[0]))
    Px = compute_mat_px(X, Nx, D)
    Pxy = compute_mat_pxy(X, Nx, D)
    for i in tqdm(range(len(X))):
        for j in range(len(X)):
            prob = ( Px[i],Px[j], Pxy[i,j] )
            di, rev_di, mi = compute_DI_MI(X[i], X[j], Nx, D, start_ratio, prob=prob)
            DI[i,j] = di[-1]
            rev_DI[i,j] = rev_di[-1]
            MI[i,j] = mi[-1]
    return DI, rev_DI, MI
