import numpy as np

# Function 'ctwentropy' outputs the (random) conditional entropy of the
# sequential probability assignment given by CTW algorithm.
# Inputs:
# X: a matrix, where for all i, x[:, i] is a probability vector
def ctwentropy(X):
    X = np.array(X)
    eps = .01
    X[np.where(x < eps)] = eps
    X[np.where(x > 1-eps)] = eps
    return np.sum(np.multiply(-x, np.log2(x)),axis=0)
