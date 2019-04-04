###################################################################
##  Written by Eli Pugh and Ethan Shen                           ##
##  {epugh}, {ezshen} @stanford.edu                              ##
##  Translated from Matlab written by Jiantao Jiao               ##
##  https://github.com/EEthinker/Universal_directed_information  ##
##  Based off of:                                                ##
##    F. Willems, Y. Shtarkov and T. Tjalkens                    ##
##    'The context-tree weighting method: basic properties'      ##
##    https://ieeexplore.ieee.org/document/382012                ##
###################################################################

import numpy as np
from tqdm import tqdm

#==============================================================================
# Function 'ctwupdate' is an update step in the CTW Algorithm
# Inputs:
# countTree:  countTree[a,:] is the tree for the count of symbol a a=0,...,M
# betaTree:   betaTree[i(s) ] =  Pe^s / \prod_{b=0}^{M} Pw^{bs}(x^{t})
# eta: [ p(X_t = 0|.) / p(X_t = M|.), ..., p(X_t = M-1|.) / p(X_t = M|.)
# xt: the current data

def ctwupdate(countTree, betaTree, eta, index, xt, alpha):
    # size of the alphabet
    Nx = len(eta)
    pw = eta
    pw = pw/np.sum(pw)  # pw(1) pw(2) .. pw(M+1)
    index = int(index)
    pe = (countTree[:,index-1]+0.5)/(np.sum(countTree[:,index-1])+Nx/2)
    temp = betaTree[index-1]
    if temp < 1000:
        eta[:-1] = (alpha*temp * pe[0:Nx-1] + (1-alpha)*pw[0:Nx-1] ) / ( alpha*temp * pe[Nx-1] + (1-alpha)*pw[Nx-1])
    else:
        eta[:-1] = (alpha*pe[0:Nx-1] + (1-alpha)*pw[0:Nx-1]/temp ) / ( alpha*pe[Nx-1] + (1-alpha)*pw[Nx-1]/temp)
    countTree[xt,index-1] = countTree[xt,index-1] + 1
    betaTree[index-1] = betaTree[index-1] * pe[xt]/pw[xt]
    return countTree, betaTree, eta

#==============================================================================
# Function 'ctwalgorithm' outputs the universal sequential probability
# assignments given by the Context Tree Weighting Algorithm
# Inputs:
# X: Input sequence
# Nx: Alphabet size of X
# D: depth of the tree

def ctwalgorithm(x, Nx, D):
    n = len(x)
    countTree = np.zeros( ( Nx, (Nx**(D+1) - 1) // (Nx-1) ))
    betaTree = np.ones( (Nx**(D+1) - 1 ) // (Nx-1) )
    Px_record = np.zeros((Nx,n-D))
    indexweight = Nx**np.arange(D)
    offset = (Nx**D - 1) // (Nx-1) + 1
    for i in range(n-D):
        context = x[i:i+D]
        leafindex = np.dot(context,indexweight)+offset
        xt = x[i+D]
        eta = (countTree[0:Nx,leafindex-1]+0.5)/(countTree[Nx-1,leafindex-1]+0.5)
        eta[-1] = 1
        # update the leaf
        countTree[xt,leafindex-1] = countTree[xt,leafindex-1] + 1
        node = np.floor((leafindex+Nx-2)/Nx)
        while node != 0:
            countTree, betaTree, eta = ctwupdate(countTree, betaTree, eta, node, xt, 1/2)
            node = np.floor((node+Nx-2)/Nx)
        eta_sum = np.sum(eta[:-1])+1
        Px_record[:,i] = eta / eta_sum
    return Px_record

