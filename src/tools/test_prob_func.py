#!/usr/bin/env python

'''test_prob_func.py: Test and compare different functions for computing the probability distribution over different actions.'''

################################################################################

import numpy as np
import matplotlib.pyplot as plt

def func(v):
    m = np.max(v)
    importance = np.exp((1.-m*0)/(1.-v))
    s_imp = np.sum(importance)
    p = np.sum(importance*v)/s_imp
    return m, p

colors = ['c', 'b', 'g', 'r', 'k']
dims = [1, 5, 10, 20, 40]

xs=[]
ys=[]

for i in range(5):
    dim = dims[i]
    color = colors[i]
    for j in range(1000000):
        v = np.random.uniform(0,1, size=[dim])
        m, p = func(v)
        xs.append(m)
        ys.append(p)
    plt.plot(xs,ys, color+'+')
    plt.savefig('2_result'+str(dim)+'.png')
    plt.close()
    xs=[]
    ys=[]
