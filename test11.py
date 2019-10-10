import numpy as np
import matplotlib.pyplot as plt
from spc_utils import elbow


from spc_utils import PMF, unmixPMF , rmse, unmixing
import matplotlib.pyplot as plt
import numpy as np

train_index, test_index = np.load("train_index.npy"), np.load("test_index.npy")
texture_clean = np.load("texture-clean.npy")
spc_clean = np.load("spc-clean.npy")
wavelength = np.load("wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]

R = spc_train[0:100,:]

errors = []
for rang in range(1,20):
    a, b, hist = unmixPMF(R, rang, max_iter=100000)
    errors.append(rmse(a@b, R))
    print("Rang {}, RMSE: {}".format(rang, errors[-1]))

best_rang = elbow(errors) + 1
print(errors)
print(best_rang)
plt.figure()
plt.title("Error vs Rango de factorización")
plt.plot(range(1, len(errors)+1), errors)
plt.plot([best_rang, best_rang], [0, max(errors)])
plt.xlabel("Rango")
plt.ylabel("RMSE")
plt.legend()
plt.grid()
plt.show()

