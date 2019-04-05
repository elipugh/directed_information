###################################################################
##  Written by Eli Pugh and Ethan Shen                           ##
##  {epugh}, {ezshen} @stanford.edu                              ##
##  This file contains tools to make computing directed          ##
##  information faster with matrices.                            ##
##  Use 'compute_DI_MI_mat()' to compute the DI between each     ##
##  pair of rows in a matrix                                     ##
###################################################################

import numpy as np
from tqdm import tqdm
from .ctwalgorithm import ctwalgorithm
from .ctwentropy import ctwentropy
from .compute_DI_MI import compute_DI_MI

#==============================================================================
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

#==============================================================================
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

#==============================================================================
# Function 'compute_DI_mat' takes in a matrix X and computes pairwise
# directed information between each of the rows of X
# DI[i,j] is the directed information I(X[i]->X[j])
# Inputs:
# X: matrix of input sequences
# Nx: Alphabet size of X
# D: Depth of the CTW Algorithm tree

def compute_DI_MI_mat(X, Nx, D, start_ratio, alg):
    X = np.array(X)
    DI = np.zeros((X.shape[0], X.shape[0]))
    rev_DI = np.zeros((X.shape[0], X.shape[0]))
    MI = np.zeros((X.shape[0], X.shape[0]))
    Px = compute_mat_px(X, Nx, D)
    Pxy = compute_mat_pxy(X, Nx, D)
    for i in tqdm(range(len(X))):
        for j in range(len(X)):
            prob = ( Px[i],Px[j], Pxy[i,j] )
            di, rev_di, mi = compute_DI_MI(X[i], X[j], Nx, D, start_ratio, prob=prob, alg=alg)
            DI[i,j] = di[-1]
            rev_DI[i,j] = rev_di[-1]
            MI[i,j] = mi[-1]
    return DI, rev_DI, MI

