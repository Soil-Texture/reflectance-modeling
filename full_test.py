import numpy as np
import os
import matplotlib.pyplot as plt
from spc_utils import corrAbsMin, corrAbsMax, corrAbsMean, corrAbsMedian, corrAbs
from spc_utils import PMF, MF, elbow, unmixPMF , rmse, unmixing, rpiq
from spc_utils import plot, plotHistory, plotEndmembers, plotScatter, plotRangesError, plotBoxes
from VCA import vca

# work_path = "/share1/jorgenarvaez/proyects/full-test"
source_path='./GDSSL'
train_index, test_index = np.load(source_path+"/train_index.npy"), np.load(source_path+"/test_index.npy")
texture_clean = np.load(source_path+"/texture-clean.npy")
spc_clean = np.load(source_path+"/spc-clean.npy")
wavelength = np.load(source_path+"/wavelength.npy")
spc_train, spc_test = spc_clean[train_index,:], spc_clean[test_index,:]
texture_train, texture_test = texture_clean[train_index, :], texture_clean[test_index,:]

rang_end = 20
C = 16
tol = 1e-5
sum_1 = False
work_path = './VCA_SumLess_C_{}'.format(C)

if os.path.exists(work_path+'/factorization_error.npy'):
    errors = np.load(work_path+'/factorization_error.npy')
    a_rpiq = np.load(work_path+'/PMF_abundances_rpiq.npy')
    rang_start = np.argmax(errors == 0) + 1
else :
    errors = np.zeros(rang_end + 1)
    a_rpiq = np.zeros(rang_end + 1)
    rang_start = 1


for rang in range(rang_start, rang_end + 1):
    alpha = 0.0001 if rang == 1 else 0.0001
    print("Working on VCA rang={}".format(rang))
    init_emb, _, _ = vca(spc_train.T, rang, verbose=True)
    print("Working on Factorization rang={}".format(rang))
    abundances, endmembers, hist = unmixPMF(spc_train, rang, b_init=init_emb.T, alpha=alpha, tol=tol, C=C, sum_1=sum_1)
    errors[rang - 1] = rmse(abundances @ endmembers, spc_train)
    np.save(work_path+'/PMF_abundances_{}_{}.npy'.format(C, rang), abundances)
    np.save(work_path+'/PMF_endmembers_{}_{}.npy'.format(C, rang), endmembers)
    np.save(work_path+'/factorization_error.npy', errors)
    print("Rango = {}, RMSE = {:.4}".format(rang, errors[rang - 1]))
    plotHistory(
        "Historia del error de factorización R=AB C={} Rango={}".format(C, rang), 
        "MSE", hist, work_path
        )
    plot(
        "Suma de las abundancias C={} Rango={}".format(C, rang), 
        range(spc_train.shape[0]), abundances.sum(axis=1), work_path, 
        "Muestras", "Suma", ylim=(0,2)
        )
    plotEndmembers(
        "Endmembers C={} Rango={}".format(C, rang),
        endmembers, wavelength, work_path
    )
    print("Working on Unmixing C={} rang={}".format(C, rang))
    pred_abundances, hist = unmixing(spc_train.T, endmembers.T, alpha=alpha, tol=tol, C=C, sum_1=sum_1)
    pred_abundances = pred_abundances.T
    np.save(work_path+'/UNMIX_abundances_{}_{}.npy'.format(C, rang), pred_abundances)
    plotHistory(
        "Historia del error de desmezclado C={} Rango={}".format(C, rang), 
        "MSE", hist, work_path
        )
    plotScatter(
        "Dispersión de abundancias C={} rang={}\n Factorización vs Desmezclado".format(C, rang),
        abundances, pred_abundances, "factorización", "desmezclado",  work_path, (0,1), (0,1)
    )
    a_rpiq[rang - 1] = rpiq(pred_abundances, abundances)
    np.save(work_path+'/PMF_abundances_rpiq.npy', a_rpiq)
    
errors = errors[0:-1]
elbow_rang = elbow(errors) + 1
plotRangesError(
    "Error de factorización vs Rango",
    1, errors, "RMSE", work_path, elbow_rang
)

a_rpiq = a_rpiq[0:-1]
plotRangesError(
    "RPIQ vs RANGO",
    1, a_rpiq, "RPIQ", work_path
)

corr_mean=[]
corr_min=[]
corr_max=[]
corr_median=[]
corr = []
for rang in range(2, rang_end+1):
    endmembers = np.load(work_path + '/PMF_endmembers_{}_{}.npy'.format(C, rang))
    corr_mean.append(corrAbsMean(endmembers))
    corr_min.append(corrAbsMin(endmembers))
    corr_max.append(corrAbsMax(endmembers))
    corr_median.append(corrAbsMedian(endmembers))
    corr.append(corrAbs(endmembers))

plotBoxes(
    "Correlaciones vs Rango",
    corr, work_path, xticks=range(2, rang_end+1)
)

corr_mean = np.array(corr_mean)
plotRangesError(
    "Correlación Media vs RANGO",
    2, corr_mean, "Correlación Media", work_path
)

corr_min = np.array(corr_min)
plotRangesError(
    "Correlación Minima vs RANGO",
    2, corr_min, "Correlación Minima", work_path
)

corr_max = np.array(corr_max)
plotRangesError(
    "Correlación Máxima vs RANGO",
    2, corr_max, "Correlación Maxima", work_path
)

corr_median = np.array(corr_median)
plotRangesError(
    "Correlación Mediana vs RANGO",
    2, corr_median, "Correlación Mediana", work_path
)

rang = 30
endmembers = np.load(work_path + '/PMF_endmembers_{}_{}.npy'.format(C, rang))
abundances = np.load(work_path + '/PMF_abundances_{}_{}.npy'.format(C, rang))

print("Working on Texture Matrix")
_, text_matrix, hist = MF(texture_train, a=abundances, alpha=0.001, tol=1e-6)
plotHistory(
    "Historia del error de calculo de Matriz de Texturas  Rango={}".format(rang), 
    "MSE", hist, work_path
    )
print("RPIQ: {:.5f}".format(rpiq(abundances @ text_matrix, texture_train)))

