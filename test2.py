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
sand_train, sand_test = texture_train[:,2], texture_test[:,2]

clay_psd, cpsd_ranges = np.histogram(clay_train, 100, range=(0,1), density=True)
sand_psd, spsd_ranges = np.histogram(sand_train, 100, range=(0,1), density=True)
 

clay_model = np.load("clay_model.npy")
clay_rmse = np.load("clay_rmse.npy")
band = 206

print("Band {}:".format(band))
print("Clay model:", clay_model[:,band])
print("Clay rmse:", clay_rmse[band])

r_psd, rpsd_ranges = np.histogram(spc_train[:,206], 100, range=(0,1), density=True)

x = cpsd_ranges[0:-1]
def p_model(x, p):
    a, b, c = p[0], p[1], p[2]
    return (beta.pdf(x, a, b) + c)/(1+c)

clay_p = [1.1, 3, 0]
sand_p = [1.1, 8, 2]

plt.figure()
plt.title("Clay PSD")
plt.plot(x,clay_psd, label="Data")
plt.plot(x,p_model(x, clay_p), label="Beta")
plt.legend()
plt.savefig("Clay PSD")

plt.figure()
plt.title("Sand PSD")
plt.plot(x, sand_psd, label="Data") 
plt.plot(x, p_model(x, sand_p), label="Beta")
plt.legend()
plt.savefig("Sand PSD")

plt.figure()
plt.title("R PSD")
plt.plot(x, r_psd, label="Data") 
plt.plot(x, norm.pdf(x, 0.365, 0.1), label="Normal")
plt.legend()
plt.savefig("R PSD")

