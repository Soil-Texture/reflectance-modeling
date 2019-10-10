import numpy as np
from spc_utils import SGD, mag, spc_angle, grad_spc_angle, mse, rmse, grad_mse, train_model, mixing, grad_mixing, poly_transform
import matplotlib.pyplot as plt

train_index, test_index = np.load("train_index.npy"), np.load("test_index.npy")
texture_clean = np.load("texture-clean.npy")
spc_clean = np.load("spc-clean.npy")
wavelength = np.load("wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]

def grad(endmembers, abundance, reflectance):
    return grad_mse(mixing(endmembers, abundance), reflectance) @ grad_mixing(endmembers, abundance)

train_x, train_y = texture_train.T , spc_train.T
endmembers_0 = np.random.rand(train_y.shape[0], train_x.shape[0])

endmembers, hist = train_model(endmembers_0, mixing, train_x, train_y, mse, grad, 0.002, 16, 100)

print("Init RMSE:", rmse(mixing(endmembers_0, train_x), train_y).mean())
print("Final RMSE:", rmse(mixing(endmembers, train_x), train_y).mean())

plt.figure()
plt.plot(hist)
plt.grid()
plt.show()

plt.figure(figsize=(9,6))
plt.plot(wavelength, endmembers[:,0], label="clay")
plt.plot(wavelength, endmembers[:,1], label="silt")
plt.plot(wavelength, endmembers[:,2], label="sand")
plt.legend()
plt.xlim((400,2500))
plt.grid()

plt.show()