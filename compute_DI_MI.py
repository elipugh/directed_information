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
from ctwentropy import ctwentropy


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
    X = np.array(X)
    Y = np.array(Y)
    assert X.shape == Y.shape
    assert alg in set(["E1", "E2", "E3", "E4"])

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

    # not tested yet
    if alg == "E1":
        rpx = np.arange(0, px.size-Nx+1, Nx)
        rpxy = np.arange(0,pxy.size-Nx**2+1, Nx**2)
        fpx = px.flatten("F")
        fpy = py.flatten("F")
        fpxy = pxy.flatten("F")
        fpx_xy = px_xy.flatten("F")
        temp_MI = -np.log2(fpx[np.add(X[D:],rpx)]) - np.log2(fpy[np.add(Y[D:],rpx)]) + np.log2(fpxy[np.add(XY[D:],rpxy)])
        temp_DI = -np.log2(fpy[Y[D:]+rpx]) + np.log2(fpxy[XY[D:]+rpxy]) - np.log2(fpx_xy[X[D:]+rpx])
        temp_rev_DI = -np.log2(fpx[X[D:]+rpx]) + np.log2(fpx_xy[X[D:]+rpx])

    # not tested yet
    elif alg == "E2":
        temp_MI = ctwentropy(px) + ctwentropy(py) - ctwentropy(pxy)
        temp_DI = ctwentropy(py) - ctwentropy(pxy) + ctwentropy(px_xy)
        temp_rev_DI = ctwentropy(px) - ctwentropy(px_xy)

    elif alg == "E3":
        temp_MI = np.zeros((1,len(px)))
        temp_DI = np.zeros((1,len(px)))
        temp_rev_di = np.zeros((1,len(px)))
        for iy in range(Nx):
            tmp1 = py_x_xy[ X[D:]+(iy-1)*Nx+np.arange(1,len(py_x_xy)-Nx^2+1, Nx**2) ]
            temp_MI +=
            temp_DI +=
            temp_rev_DI +=

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


