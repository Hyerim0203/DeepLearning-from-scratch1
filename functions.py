#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# sigmoid 함수
def sigmoid(x):
    return 1/(1+np.exp(-x))

# relu 함수
def relu(x):
    return np.maximum(0,x)

# softmax 함수
def softmax(x):
    big = np.max(x, axis=1).reshape(-1,1)
    exp_x = np.exp(x-big)
    under = np.sum(exp_x, axis=1).reshape(-1,1)
    return exp_x/under

def cross_entropy_error(y,t):
    epsilon = 1e-7
    if y.ndim ==1:
        y=y.reshape(1,y.size)
        t=t.reshape(1,t.size)
    batch_size = y.shape[0]
    cee = np.sum(-t * np.log(y+epsilon)) / batch_size
    return cee