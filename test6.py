import numpy as np
from spc_utils import SGD, mag, spc_angle, grad_spc_angle, mse, grad_mse, train_model, mixing, grad_mixing
import matplotlib.pyplot as plt

train_index, test_index = np.load("train_index.npy"), np.load("test_index.npy")
texture_clean = np.load("texture-clean.npy")
spc_clean = np.load("spc-clean.npy")
wavelength = np.load("wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]

def expModel(p, clay):
    p1 = p[:,0:1]*np.ones((clay.shape[1],))
    p3c = p[:,2:3]*clay
    exp = np.exp(p3c)
    p2_exp = p[:,1:2]*exp
    return p1 + p2_exp

def grad(p, clay, r):
    p3c = p[:,2:3]*clay
    p2c = p[:,1:2]*clay
    exp = np.exp(p3c)
    p1 = p[:,0:1]*np.ones((clay.shape[1],))
    p2_exp = p[:,1:2]*exp
    err = p1 + p2_exp - r
    out = np.empty((r.shape[0], r.shape[1], 3))
    out[:,:,0] = err
    out[:,:,1] = err*exp
    out[:,:,1] = out[:,:,1]*p2c
    return out.sum(axis=1)

p_0 = np.random.rand(spc_train.shape[1], texture_train.shape[1])
train_x, train_y = texture_train[:,0:1].T, spc_train.T

p, hist = train_model(p_0, expModel, train_x, train_y, mse, grad, 0.0005, 8, 85)

plt.figure()
plt.plot(hist)
plt.grid()
plt.show()

plt.figure(figsize=(9,6))
plt.plot(wavelength, p[:,0], label="bias")
plt.plot(wavelength, p[:,1], label="lin")
plt.xlim((400,2500))
plt.grid()
plt.legend()

plt.figure(figsize=(9,6))
plt.plot(wavelength, p[:,2], label="exp")
plt.legend()
plt.xlim((400,2500))
plt.grid()

plt.show()
