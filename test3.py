import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
from scipy.signal import medfilt
from util import fitModel

train_index, test_index = np.load("train_index.npy"), np.load("test_index.npy")
texture_clean = np.load("texture-clean.npy")
spc_clean = np.load("spc-clean.npy")
wavelength = np.load("wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]
clay_train, clay_test = texture_train[:,0], texture_test[:,0]
spcb206_train = spc_train[:,206]

def P_C(clay):
    #return 1
    return beta.pdf(clay, 3, 3) 

def P_R_C(r, clay):
    a, b, c = [ 0.03217907,  0.46547123, -0.96546346]
    std = 0.08869425764626322
    #std = 0.1
    return norm.pdf(r, a + b*np.exp(c*clay), std)

def P_R(r):
    #return 1
    return norm.pdf(r, 0.365, 0.1)

def P_C_R(clay, r):
    return P_R_C(r,clay)*P_C(clay)/P_R(r)

def pred_clay(rs):
    clay_space = np.linspace(0,1,100)
    pred_clay = np.zeros(rs.shape)
    std = np.zeros(rs.shape)
    for i in range(rs.shape[0]):
        p_clay = P_C_R(clay_space, rs[i])
        pred_clay[i] = clay_space[np.argmax(p_clay)]
        std[i] = np.std(p_clay)
    return pred_clay, std

clay_space = np.linspace(0,1,100)

rs = [0, 0.25, 0.5, 0.75, 1]
plt.figure()
for r in rs:
    plt.plot(clay_space, P_C_R(clay_space, r), label="r={}".format(r))
plt.legend()
plt.show()

x = spcb206_train
y = clay_train
pred, err_st = pred_clay(x)

plt.figure()
plt.plot(y, pred, '+')
plt.xlabel("Label")
plt.ylabel("Predicción")
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100))

plt.figure()
e_index = np.argsort(err_st)
err = np.abs(pred - y)
plt.plot(err[e_index], err_st[e_index], '+')
plt.xlabel("Label")
plt.ylabel("Predicción")
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100))

plt.show()