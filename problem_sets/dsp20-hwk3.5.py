# Digital Signal Processing - hwk3 - Part 3.5
# Christos Dimopoulos - 03117037
# PCA SVD

import numpy as np
from numpy import linalg

X = [[1, -0.1, 0.2],
     [1, -0.3, 0.5],
     [0, 0.1, 0.3],
     [5, -1, 1]]

X = np.array(X)

XT = np.transpose(X)

Rx = (1/5)*np.matmul(XT,X)

eigenvalues,eigenvectors = linalg.eig(Rx)

v1 = (eigenvectors[:,0])
a = np.matmul(X,v1)
e = v1
print('e = ',e)
print('a = ',a)