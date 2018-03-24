import numpy as np


class SGD(object):

    def __init__(self, params, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = {}
        for k, v in params.items():
            self.v[k] = np.zeros(v.shape)

    def update(self, params, grad):
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grad[key]
            params[key] += self.v[key]
        return params


class AdaGrad(object):

    def __init__(self, params, lr=0.001, eps=1e-08):
        self.lr = lr
        self.h = {}
        for k, v in params.items():
            self.h[k] = np.zeros(v.shape) + eps

    def update(self, params, grad):
        for key in params.keys():
            self.h[key] += grad[key] * grad[key]
            params[key] -= self.lr * grad[key] / np.sqrt(self.h[key])
        return params


class Adam(object):

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-08):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        for k, v in params.items():
            self.m[k] = np.zeros(v.shape)
            self.v[k] = np.zeros(v.shape)

    def update(self, params, grad):
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad[key] * grad[key]
            m_t = self.m[key] / (1 - self.beta1)
            v_t = self.v[key] / (1 - self.beta2)
            params[key] -= self.lr * m_t / (np.sqrt(v_t) + self.eps)
        return params
