# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

CLASS_NUM = 2
N = 50
np.random.seed(seed = 32)
x = np.random.randn(N, CLASS_NUM)
y = 2 * (x[:,0] > x[:,1]) - 1

if __name__ == "__main__":
    d = int(np.ceil(np.random.rand() * CLASS_NUM)) - 1
    xs = np.sort(x[:,d])
    xi = np.argsort(x[:,d])
    el = np.cumsum(y[xi])
    eu = np.cumsum(y[xi[::-1]])
    e = eu[(N-1):0:-1] - el[0:(N-1)]
    ei = int(np.max(np.abs(e)))
    c = xs[ei]
    
    if d == 0:
        X0 = np.linspace(-3, 3, N)
        Y0 = X0*0 + c
    else:
        Y0 = np.linspace(-3, 3, N)
        X0 = X0*0 + c        

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.plot(x[y==1,0], x[y==1,1], 'bo')
    plt.plot(x[y==-1,0], x[y==-1,1], 'rx')
    plt.plot(X0, Y0, "k-") 
    plt.show()
