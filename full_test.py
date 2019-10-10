import numpy as np
import os
import matplotlib.pyplot as plt
from spc_utils import PMF, elbow, unmixPMF , rmse, unmixing

# work_path = "/share1/jorgenarvaez/proyects/full-test"
work_path = "."
train_index, test_index = np.load(work_path+"/train_index.npy"), np.load(work_path+"/test_index.npy")
texture_clean = np.load(work_path+"/texture-clean.npy")
spc_clean = np.load(work_path+"/spc-clean.npy")
wavelength = np.load(work_path+"/wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]

rang_end = 20
if os.path.exists(work_path+'/factorization_error.npy'):
    errors = np.load(work_path+'/factorization_error.npy')
    rang_start = np.argmax(errors == 0) + 1
else :
    errors = np.zeros(rang_end + 1)
    rang_start = 1

for rang in range(rang_start, rang_end + 1):
    print("Working on rang={}".format(rang))
    a, b, hist = PMF(spc_train, rang, max_iter=100000)
    errors[rang - 1] = rmse(a@b, spc_train)
    np.save(work_path+'/factorization_error.npy', errors)
    print("Rango = {}, RMSE = {:.4}".format(rang, errors[rang - 1]))
    plt.figure()
    title = "Historia del error de factorización R=AB (Rango={})".format(rang)
    plt.title(title)
    plt.plot(range(1,len(hist) + 1), hist)
    plt.xlabel("Iteración")
    plt.ylabel("MSE")
    plt.grid()
    plt.savefig(work_path+'/'+title + ".png")


errors = errors[0:-1]
elbow_rang = elbow(errors) + 1
plt.figure()
title="Error de factorización vs Rango"
plt.title(title)
plt.plot(range(1, rang_end+1), errors)
plt.plot([elbow_rang, elbow_rang], [0, np.max(errors)])
plt.xlabel("Rango")
plt.xticks(range(1, rang_end+1))
plt.ylabel("RMSE")
plt.grid()
plt.savefig(work_path+'/'+title + ".png")

groups_test = 2
rang = 12
C=10
if os.path.exists(work_path+'/endmembers_groups_{}_{}.npy'.format(rang, C)):
    endmembers_groups = np.load(work_path+'/endmembers_groups_{}_{}.npy'.format(rang, C))
    abundances_groups = np.load(work_path+'/abundances_groups_{}_{}.npy'.format(rang, C))
    i_start = np.argmax(endmembers_groups.sum(axis=0).sum(axis=0) == 0)
else :
    endmembers_groups = np.zeros((rang, spc_train.shape[1], groups_test+1))
    abundances_groups = np.zeros((spc_train.shape[0], rang, groups_test+1))
    i_start = 0
for i in range(i_start, groups_test):
    print("Working on endmembers group {}".format(i))
    abundances, endmembers_groups[:,:,i], hist = unmixPMF(spc_train, rang, tol=1e-7, C=C)
    np.save(work_path+'/endmembers_groups_{}_{}.npy'.format(rang, C), endmembers_groups)
    np.save(work_path+'/abundances_groups_{}_{}.npy'.format(rang, C), abundances_groups)
    print("Group = {}, RMSE = {:.4}".format(i, rmse(abundances @ endmembers_groups[:,:,i], spc_train)))
    plt.figure()
    title = "Historia del error de factorización C={} Rango={} (Grupo={})".format(C, rang, i)
    plt.title(title)
    plt.plot(range(1,len(hist) + 1), hist)
    plt.xlabel("Iteración")
    plt.ylabel("MSE")
    plt.grid()
    plt.savefig(work_path+'/'+title + ".png")
    plt.figure()
    title = "Suma de las abundancias C={} Rango={} (Grupo={})".format(C, rang, i)
    plt.title(title)
    plt.plot(abundances.sum(axis=1))
    plt.xlabel("Muestra")
    plt.ylabel("Suma")
    plt.grid()
    plt.ylim((0,2))
    plt.savefig(work_path+'/'+title + ".png")
    plt.figure()
    title = "Endmembers C={} Rango={} (Grupo={})".format(C, rang, i)
    plt.title(title)
    for j in range(rang):
        plt.plot(wavelength, endmembers_groups[j,:,i])
    plt.xlabel("Longitud de Onda (nm)")
    plt.ylabel("Reflectancia")
    plt.grid()
    plt.xlim((400,2500))
    plt.savefig(work_path+'/'+title + ".png")
