import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from util import fitModel
from scipy.signal import medfilt
from sklearn.model_selection import train_test_split

# Ignore Error Messaes
sys.stderr = open(os.devnull)

texture_clean = np.load("texture-clean.npy")
spc_clean = np.load("spc-clean.npy")
wavelength = np.load("wavelength.npy")
index = np.arange(0, spc_clean.shape[0], 1, dtype=int)

train_index, test_index = np.load("train_index.npy"), np.load("test_index.npy")
# train_index, test_index = train_test_split(index, test_size=0.3)
# np.save("train_index.npy", train_index)
# np.save("test_index.npy", test_index)
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]

clay_train, clay_test = texture_train[:,0], texture_test[:,0]
sand_train, sand_test = texture_train[:,2], texture_test[:,2]

# def spectralCorr(component, data):
#     corr_spc = np.zeros((spc.shape[1],))
#     for i in range(corr_spc.shape[0]):
#         corr_spc[i], _ = pearsonr(data, spc[:,i])
#     plt.figure()
#     plt.grid()
#     plt.title("{} Correlation".format(component))
#     plt.plot(wavelength, corr_spc)
#     plt.savefig("corr-{}.png".format(component))
#     return corr_spc

def expModel(x, p):
    return p[0] + p[1]*np.exp(p[2]*x)

def fitExpModel(train_x, train_y, test_x, test_y, p0=[0,1,-1]):
    model_p = fitModel(expModel, train_x, train_y, p0)
    pred_y = expModel(test_x, model_p)
    RMSE = ((pred_y - test_y)**2).mean()**0.5
    IQR = np.percentile(test_y, 75) - np.percentile(test_y, 25)
    RPIQ = IQR/RMSE
    return (model_p, RMSE, RPIQ)

def zeroCrossAreaRate(x):
    sum_a = np.abs(x).sum()
    a_sum = np.abs(x.sum())
    return (sum_a - a_sum)/sum_a

def graphicModel(component, band, model, x, y):
    plt.figure()
    sort_i = np.argsort(x)
    title = "Reflectance-{}nm-vs-{}".format(band, component)
    plt.title(title)
    plt.grid()
    plt.plot(x[sort_i], y[sort_i], '+', label="Reflectance")
    plt.plot(x[sort_i], medfilt(y[sort_i], 51),label="Filtered Reflectance")
    plt.plot(x[sort_i], model[sort_i], label="Model: R = a + b*exp(c*{})".format(component), color='r')
    plt.legend()
    plt.xlabel(component)
    plt.ylabel("reflectance")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.savefig(title + ".png")

def graphicSample(wavelength, sample, clay, sand, y, clay_model_p, sand_model_p):
    x_clay, x_sand = clay[sample], sand[sample]
    clay_pred_y = expModel(x_clay, clay_model_p)
    sand_pred_y = expModel(x_sand, sand_model_p)
    plt.figure()
    plt.grid()
    plt.xlim((400,2500))
    plt.ylim((0,1))
    plt.plot(wavelength, y, label="sample")
    plt.plot(wavelength, clay_pred_y, label="Clay Prediction")
    plt.plot(wavelength, sand_pred_y, label="Sand Prediction")
    plt.xlabel("nm")
    plt.ylabel("reflectance")
    plt.legend()
    plt.savefig("test_sample_{}.png".format(sample))

def graphicZCAR(component, zcar):
    plt.figure()
    plt.grid()
    title = "{}_zcar.png".format(component)
    plt.title(title)
    plt.xlim((0,1))
    plt.hist(zcar, bins=100)
    plt.xlabel("zcar")
    plt.ylabel("ocurrences")
    plt.savefig(title)

clay_model = np.load("clay_model.npy")
clay_rmse = np.load("clay_rmse.npy")
clay_rpiq = np.load("clay_rpiq.npy")
# clay_model = np.empty((3, spc_clean.shape[1]))
# clay_rmse = np.empty((spc_clean.shape[1], ))
# clay_rpiq = np.empty((spc_clean.shape[1], ))
# for i in range(spc_clean.shape[1]):
#     params, rmse, rpiq = fitExpModel(clay_train, spc_train[:,i], clay_test, spc_test[:,i])
#     clay_model[:,i] = params
#     clay_rmse[i] = rmse
#     clay_rpiq[i] = rpiq
# np.save("clay_model.npy", clay_model)
# np.save("clay_rmse.npy", clay_rmse)
# np.save("clay_rpiq.npy", clay_rpiq)

sand_model = np.load("sand_model.npy")
sand_rmse = np.load("sand_rmse.npy")
sand_rpiq = np.load("sand_rpiq.npy")
# sand_model = np.empty((3, spc_clean.shape[1]))
# sand_rmse = np.empty((spc_clean.shape[1], ))
# sand_rpiq = np.empty((spc_clean.shape[1], ))
# for i in range(spc_clean.shape[1]):
#     params, rmse, rpiq = fitExpModel(sand_train, spc_train[:,i], sand_test, spc_test[:,i])
#     sand_model[:,i] = params
#     sand_rmse[i] = rmse
#     sand_rpiq[i] = rpiq
# np.save("sand_model.npy", sand_model)
# np.save("sand_rmse.npy", sand_rmse)
# np.save("sand_rpiq.npy", sand_rpiq)

clay_best_band = np.argmax(clay_rpiq)
clay_best_model = expModel(clay_test, clay_model[:,clay_best_band])
clay_poor_band = np.argmin(clay_rpiq)

graphicModel(   "clay", 
                wavelength[clay_best_band], 
                clay_best_model, 
                clay_test, 
                spc_test[:,clay_best_band])

graphicModel(   "clay", 
                wavelength[clay_poor_band], 
                expModel(clay_test, clay_model[:,clay_poor_band]), 
                clay_test, 
                spc_test[:,clay_poor_band])

sand_best_band = np.argmax(sand_rpiq)
sand_best_model = expModel(sand_test, sand_model[:,sand_best_band])
sand_poor_band = np.argmin(sand_rpiq)

graphicModel(   "sand", 
                wavelength[sand_best_band], 
                sand_best_model, 
                sand_test, 
                spc_test[:,sand_best_band])

graphicModel(   "sand", 
                wavelength[sand_poor_band], 
                expModel(sand_test, sand_model[:,sand_poor_band]), 
                sand_test, 
                spc_test[:,sand_poor_band])

poor_sample = np.argmax(np.abs(clay_best_model - spc_test[:, clay_best_band]) +  np.abs(sand_best_model - spc_test[:, sand_best_band]))
graphicSample(wavelength, poor_sample, clay_test, sand_test, spc_test[poor_sample, :], clay_model, sand_model)

clay_zcar = np.empty((spc_test.shape[0],))
sand_zcar = np.empty((spc_test.shape[0],))
clay_error = np.empty(spc_test.shape)
sand_error = np.empty(spc_test.shape)
for i in range(spc_test.shape[0]):
    y = spc_test[i,:]
    clay_pred_y = expModel(clay_test[i], clay_model)
    sand_pred_y = expModel(sand_test[i], sand_model)
    clay_error[i,:] = y - clay_pred_y
    sand_error[i,:] = y - sand_pred_y
    clay_zcar[i] = zeroCrossAreaRate(clay_error[i,:])
    sand_zcar[i] = zeroCrossAreaRate(sand_error[i,:])
comb_error = sand_error + clay_error
best_zcar = np.argmax(clay_zcar + sand_zcar)
clay_rmse_mean = (clay_error**2).mean()**0.5
sand_rmse_mean = (sand_error**2).mean()**0.5
comb_rmse_mean = (comb_error**2).mean()**0.5
print("rmse mean Errors: clay={:.4}, sand={:.4}, comb={:.4}".format(clay_rmse_mean, sand_rmse_mean, comb_rmse_mean))

graphicZCAR("clay_spectral_error", clay_zcar)
graphicZCAR("sand_spectral_error", sand_zcar)
graphicSample(wavelength, best_zcar, clay_test, sand_test, spc_test[best_zcar, :], clay_model, sand_model)