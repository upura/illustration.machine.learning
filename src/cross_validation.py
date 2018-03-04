# -*- coding: utf-8 -*-

import numpy as np
from sklearn import cross_validation

n = 50
N = 1000
np.random.seed(seed = 32)

x = np.linspace(-3, 3, n)
x = np.array(x, ndmin = 2).T
X = np.linspace(-3, 3, N)
X = np.array(X, ndmin = 2).T
x2 = x ** 2
xx = np.tile(x2,(1, n)) + np.tile(x2.T,(n, 1)) - 2 * np.dot(x, x.T)

pix = np.pi * x
y = np.sin(pix) / pix + 0.1 * x + 0.2 * np.random.randn(n, 1)

k_fold = cross_validation.KFold(n, n)

l = [0.0001, 0.1, 100]
h = [0.03, 0.3, 3]

min_loss = 1000000
min_loss_l = 0
min_loss_h = 0
for ls in l:
    for hs in h:
        loss = []
        for train, test in k_fold:
            x2 = x[train] ** 2
            x2_test = x[test] ** 2
            hh = 2 * hs ** 2
            k = np.exp(-xx/hh)
            k_test = k[test]
            k_train = k[train]
            #k = np.exp(-(np.tile(x2,(1, n-1)) + np.tile(x2.T,(n-1, 1)) - 2 * np.dot(x[train], x[train].T))/hh)
            #k_test = np.exp(-(np.tile(x2_test,(1, n - 1)) + np.tile(x2_test.T,(n - 1, 1)) - 2 * np.dot(x[test], x[test].T))/hh)
            t = np.linalg.solve((np.dot(k_train.T, k_train) + ls * np.eye(n)), np.dot(k_train.T, y[train]))
            f_train = np.dot(k_train, t)
            f_test = np.dot(k_test, t)
            loss.append((f_test[0]-y[test])**2)
        if (sum(loss) / len(loss)) < min_loss:
            min_loss = sum(loss) / len(loss)
            min_loss_l = ls
            min_loss_h = hs
            
print(min_loss_l, min_loss_h)
