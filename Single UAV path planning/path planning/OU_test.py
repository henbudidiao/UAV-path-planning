# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2023/8/25 17:29
import numpy as np
import matplotlib.pyplot as plt


class Ornstein_Uhlenbeck_Noise:
    def __init__(self, mu, sigma=1.0, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        '''
        后两行是dXt，其中后两行的前一行是θ(μ-Xt)dt，后一行是σεsqrt(dt)
        '''
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)
ou_noise = Ornstein_Uhlenbeck_Noise(mu=np.zeros(2))
y1 = []
y2 = np.random.normal(0, 1, 1000) #高斯噪声
for _ in range(1000):
        y1.append(ou_noise())
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].plot(y1, c='r')
ax[0].set_title('OU noise')
ax[1].plot(y2, c='b')
ax[1].set_title('Guassian noise')
plt.show()