from scipy.optimize import minimize
import numpy as np

def rmse_error(p, model, x, y):
    return ((y - model(x, p))**2).mean()**0.5

def mse_error(p, model, x, y):
    return ((y - model(x, p))**2).mean()

def abs_error(p, model, x, y):
    return (np.abs(y - model(x, p))).mean()

errors = {
    "abs" : abs_error,
    "mse" : mse_error,
    "rmse": rmse_error
}

def fitModel(model, x, y, p0, error="abs"):
    r = minimize(errors[error], p0, args=(model, x, y), tol=1e-9, method='SLSQP', options={'maxiter': 10000})
    if r['success']:
        return r['x']
    else:
        raise ValueError("No se pudo optimizar")
