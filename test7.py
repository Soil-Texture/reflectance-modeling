import numpy as np
from spc_utils import SGD, mag, spc_angle, grad_spc_angle, mse, rmse, grad_mse, train_model, mixing, grad_mixing, poly_transform, abse, grad_abse, line, grad_line, unmixing, rpiq
import matplotlib.pyplot as plt
from scipy.signal import medfilt

train_index, test_index = np.load("train_index.npy"), np.load("test_index.npy")
texture_clean = np.load("texture-clean.npy")
spc_clean = np.load("spc-clean.npy")
wavelength = np.load("wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]

def grad_mixing_cost(endmembers, abundance, reflectance):
    return grad_abse(mixing(endmembers, abundance), reflectance) @ grad_mixing(endmembers, abundance)

train_x, train_y = texture_train.T, spc_train.T
endmembers_0 = np.random.rand(train_y.shape[0], train_x.shape[0])
endmembers, hist = train_model(endmembers_0, mixing, train_x, train_y, abse, grad_mixing_cost, 0.0002, 16, 100)

print("Init RMSE:", rmse(mixing(endmembers_0, train_x), train_y).mean())
print("Final RMSE:", rmse(mixing(endmembers, train_x), train_y).mean())

# plt.figure()
# plt.plot(hist)
# plt.grid()
# plt.show()

# plt.figure(figsize=(9,6))
# for i in range(endmembers.shape[1]):
#      plt.plot(wavelength, endmembers[:,i])

# plt.legend()
# plt.xlim((400,2500))
# plt.grid()
# plt.show()

texture, hist = unmixing(spc_train.T, endmembers, error=abse, grad_error=grad_abse)

print("RMSE:", rmse(endmembers@texture, spc_train.T).mean())
print("RPIQ:", rpiq(texture, train_x))

# plt.figure()
# plt.plot(hist)
# plt.grid()
# plt.show()

plt.figure(figsize=(9,6))
plt.plot(texture[0,:], train_x[0,:], '+')
plt.plot(np.linspace(0,1,2), np.linspace(0,1,2))
plt.grid()
plt.show()

# plt.figure(figsize=(9,6))
# sort_i = np.argsort(texture[0,:])
# sort_i2 = np.argsort(train_x[0,:])
# plt.plot(medfilt(train_x[0,sort_i2], 1))
# plt.plot(texture[0,sort_i])
# plt.grid()
# plt.show()

