import numpy as np
from scipy.stats import norm
from random import random

def biased_weights(N_, W_i, bias_, spread_):
    W_ = np.zeros([N_, N_])
    for ii in range(0, int(np.floor(N_/2))):
        for jj in range(1, N_ + 1):
            W_[jj - 1, np.mod(jj + ii - 1, N_)] = norm.pdf(ii, 0, spread_)
            W_[jj - 1, np.mod(jj - ii - 1, N_)] = norm.pdf(ii, 0, spread_)
    W_ = W_/norm.pdf(0, 0, spread_)
    W_ = bias_ * W_ + W_i[0] + (W_i[1] - W_i[0])*np.random.random([N_, N_])
    W_[W_ < 0] = 0
    return W_


def inverted_biased_weights(N_, W_i, bias_, spread_):
    W_ = np.zeros([N_, N_])
    for ii in range(0, int(np.floor(N_/2))):
        for jj in range(1, N_ + 1):
            W_[jj - 1, np.mod(jj + ii + int(np.floor(N_/2)) - 1, N_)] = norm.pdf(ii, 0, spread_)
            W_[jj - 1, np.mod(jj - ii + int(np.floor(N_/2)) - 1, N_)] = norm.pdf(ii, 0, spread_)
    W_ = W_/norm.pdf(0, 0, spread_)
    W_ = bias_ * W_ + W_i[0] + (W_i[1] - W_i[0])*random()
    W_[W_ < 0] = 0
    return W_
