from scipy.optimize import minimize
import numpy as np

def fit_error(p, model, x, y):
    return (np.abs(y - model(x, p))).mean()

def fitModel(model, x, y, p0):
    r = minimize(fit_error, p0, args=(model, x, y), tol=1e-15, method='SLSQP', options={'maxiter': 100000})
    if r['success']:
        return r['x']
    else:
        raise ValueError("No se pudo optimizar")
