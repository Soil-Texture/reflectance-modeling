import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spc_utils import PMF, spc_angle, rmse
from scipy.optimize import minimize

source_path='./GDSSL'
spc_libs_path = './USGD'
graphics = './temp_graphics'

train_index, test_index = np.load(source_path+"/train_index.npy"), np.load(source_path+"/test_index.npy")
texture_clean = np.load(source_path+"/texture-clean.npy")
spc_clean = np.load(source_path+"/spc-clean.npy")
wavelength = np.load(source_path+"/wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]
SoilsAndMixtures = pd.read_pickle(spc_libs_path+'/SoilsAndMixtures.pkl')
Minerals = pd.read_pickle(spc_libs_path+'/Minerals.pkl')
OrganicCompounds = pd.read_pickle(spc_libs_path+'/OrganicCompounds.pkl')

def plotEnmbSign(title, wavelength, endmember, signature, path):
    plt.figure()
    plt.plot(wavelength, endmember, label="Endmember")
    plt.plot(wavelength, signature, label="Firma")
    plt.title(title)
    plt.grid()
    plt.xlim((wavelength[0], wavelength[-1]))
    plt.legend()
    plt.savefig(path + "/" + title + ".png")
    plt.close()

def linTransform(a, b):
    cost = lambda p, a, b: np.sum((a*p[0] + p[1] - b)**2)
    r = minimize(cost, [1, 0], args=(a,b))
    if r["success"]:
        p = r["x"]
        return a*p[0] + p[1]
    else :
        raise ValueError("Not posible optimize linear transform")

def find_spc_component(endmember, spc_lib, tol = 1e-3):
    angles = np.array([spc_angle(endmember, spc_lib.iloc[:,i].values) for i in range(spc_lib.shape[1])])
    best_spc = np.argmin(angles)
    if angles[best_spc] > tol:
        return None
    candidate = spc_lib.iloc[:,best_spc].values
    trans = linTransform(endmember, candidate)
    _rmse = rmse(trans, candidate)
    if _rmse <= tol:
        return best_spc, trans
    else:
        return None

libs = [SoilsAndMixtures, Minerals, OrganicCompounds]
C = 16
tol = 1e-5
sum_1 = False
work_path = './SumLess_C_{}'.format(C)

for rang in range(1, 21):
    if not os.path.exists("./temp_graphics/rang_{}".format(rang)):
        os.makedirs("./temp_graphics/rang_{}".format(rang))
    endmembers = np.load(work_path + '/PMF_endmembers_{}_{}.npy'.format(C, rang))
    for i in range(endmembers.shape[0]):

        for lib in libs:
            r = find_spc_component(endmembers[i,:], lib, tol=4e-2)
            if not (r is None):
                sig, trans = r[0], r[1]
                plotEnmbSign(
                    "Endmember {}, Componente: {}".format(i, lib.columns[sig]),
                    wavelength, trans, lib.iloc[:,sig].values, "./temp_graphics/rang_{}".format(rang))
