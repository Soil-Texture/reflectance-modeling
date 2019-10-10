from spc_utils import PMF, mse, grad_mse, rmse, unmixing
import matplotlib.pyplot as plt
import numpy as np
from VCA import vca

train_index, test_index = np.load("train_index.npy"), np.load("test_index.npy")
texture_clean = np.load("texture-clean.npy")
spc_clean = np.load("spc-clean.npy")
wavelength = np.load("wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]


endmembers, indice, proy = vca(spc_train.T, 40)
print(indice)
print(proy.shape)

plt.figure(figsize=(9,6))
plt.grid()
plt.xlim((400,2500))
for i in range(endmembers.shape[1]):
     plt.plot(wavelength, endmembers[:,i])

plt.show()

abundances, hist = unmixing(spc_train.T, endmembers, error=mse, grad_error=grad_mse)

plt.figure()
plt.plot(hist)
plt.grid()
plt.show()

print("RMSE:", rmse(endmembers@abundances, spc_train.T).mean())