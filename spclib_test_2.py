import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spc_utils import PMF, spc_angle, rmse, plotHistory
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


def findTransMatrix(endmembers, lib):
    a, _, hist = PMF(endmembers, rang=lib.shape[0], b = lib, alpha=0.00001, tol=1e-2, max_iter=500000)
    return a, hist

lib = pd.concat([SoilsAndMixtures, Minerals, OrganicCompounds], axis = 1).values.T

C = 16
tol = 1e-5
sum_1 = False
work_path = './SumLess_C_{}'.format(C)

for rang in range(1, 21):
    if not os.path.exists("./temp_graphics/rang_{}".format(rang)):
        os.makedirs("./temp_graphics/rang_{}".format(rang))
    endmembers = np.load(work_path + '/PMF_endmembers_{}_{}.npy'.format(C, rang))
    a, hist = findTransMatrix(endmembers, lib)
    plotHistory("Error History", "MSE", hist, "./temp_graphics/rang_{}".format(rang))
    est_enm = a @ lib
    for i in range(endmembers.shape[0]):
        plotEnmbSign(
            "Endmember {}".format(i),
            wavelength, endmembers[i,:], 
            est_enm[i, :], "./temp_graphics/rang_{}".format(rang))
