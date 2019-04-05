![](https://img.shields.io/pypi/v/directed_information.svg?style=plastic) ![](https://img.shields.io/pypi/l/directed_information.svg?style=plastic)

# Universal Estimation of Directed Information

Python3 implementation of the universal directed information estimators in Jiantao Jiao, Haim H. Permuter, Lei Zhao, Young-Han Kim, and Tsachy Weissman. "Universal estimation of directed information." IEEE Transactions on Information Theory 59, no. 10 (2013): 6220-6242.

See here: [http://arxiv.org/abs/1201.2334](http://arxiv.org/abs/1201.2334)

Also see [MATLAB implementation](https://github.com/EEthinker/Universal_directed_information)

## Authors
[Eli Pugh](https://github.com/elipugh), [Ethan Shen](https://github.com/ezshen)

# Installation

```sh
pip install directed_information
```
This package currently requires [Python 3](https://www.python.org/downloads/), [numpy](https://github.com/numpy/numpy), and [tqdm](https://github.com/tqdm/tqdm).

# Usage
```python
from directed_information import *
import numpy as np

# to find DI, reverse DI, and MI between
# X and Y, using CTW algorithm depth 3
Nx = 2
D = 3
X = np.random.randint(Nx,size=50)
Y = np.random.randint(Nx,size=50)
DI, rev_DI, MI = compute_DI_MI(X,Y,Nx,D,"E4",0)

# to find DI, reverse DI, and MI between
# each row of X, using CTW algorithm depth 2
# (DI[i,j] is DI between rows i and j of X)
Nx = 2
D = 2
X = np.random.randint(Nx,size=(50,50))
DI, rev_DI, MI = compute_DI_MI_mat(X,Nx,D,0,"E3")
```

