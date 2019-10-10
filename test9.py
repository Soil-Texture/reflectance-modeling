from spc_utils import PMF, unmixPMF , rmse, unmixing
import matplotlib.pyplot as plt
import numpy as np

train_index, test_index = np.load("train_index.npy"), np.load("test_index.npy")
texture_clean = np.load("texture-clean.npy")
spc_clean = np.load("spc-clean.npy")
wavelength = np.load("wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]

R = spc_train[0:200,:]

rang = 7
# a, b, hist = PMF(R, rang)
a, b, hist = unmixPMF(R, rang)

print("RMSE R Matrix: {:.4}".format(rmse(a @ b, R)))

plt.figure()
plt.plot(hist)
plt.grid()
plt.show()

plt.figure()
plt.plot(a.sum(axis=1))
plt.grid()
plt.ylim((0,2))
plt.show()

plt.figure(figsize=(9,6))
plt.grid()
plt.xlim((400,2500))
for i in range(b.shape[0]):
     plt.plot(wavelength, b[i,:])
plt.show()

abundances, hist = unmixing(R.T, b.T)

plt.figure()
plt.plot(hist)
plt.grid()
plt.show()

print("RMSE ABUNDANCES: {:.4}".format(rmse(abundances, a.T)))

plt.figure()
plt.plot(abundances.sum(axis=0))
plt.grid()
plt.ylim((0,2))
plt.show()

plt.figure(figsize=(9,6))
for i in range(a.shape[1]):
    plt.plot(abundances.T[:,i], a[:,i], '+')
plt.plot(np.linspace(0,1,2), np.linspace(0,1,2))
plt.grid()
plt.show()
