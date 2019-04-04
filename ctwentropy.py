import numpy as np

#==============================================================================
# Function 'ctwentropy' outputs the (random) conditional entropy of the
# sequential probability assignment given by CTW algorithm.
# Inputs:
# X: a matrix, where for all i, x[:, i] is a probability vector

def ctwentropy(X):
    X = np.array(X).T
    eps = .01
    for i in range(len(X)):
        X[i] = np.clip(X[i], eps, 1-eps)
    return np.sum(np.multiply(-X, np.log2(X)),axis=1)

