import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from spc_utils import rpiq

train_index, test_index = np.load("train_index.npy"), np.load("test_index.npy")
texture_clean = np.load("texture-clean.npy")
spc_clean = np.load("spc-clean.npy")
wavelength = np.load("wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]

pls = PLSRegression(n_components=100)
pls.fit(spc_train, texture_train)


texture_pred = pls.predict(spc_train)

print("Train RPIQ: ", rpiq(texture_pred.T, texture_train.T))
print("Test RPIQ: ", rpiq(pls.predict(spc_test).T, texture_test.T))


plt.figure(figsize=(9,6))
plt.plot(texture_pred[:,0], texture_train[:,0], '+')
plt.plot(np.linspace(0,1,2), np.linspace(0,1,2))
plt.grid()
plt.show()
