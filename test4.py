import numpy as np
from spc_utils import SGD, mag, spc_angle, grad_spc_angle, mse, grad_mse, train_model
import matplotlib.pyplot as plt


def model(p, x):
    return p @ np.vstack((np.ones(x.shape[1]), x, x**2, x**3))

def grad_model(p, x):
    return np.vstack((np.ones(x.shape[1]), x, x**2, x**3)).T
    
def grad(p, x, y):
    return grad_mse(model(p, x), y) @ grad_model(p, x)  

train_x = np.expand_dims(np.linspace(-2,2,100), 0)
train_y = 40 + 3*train_x - 4*train_x**2 + 8*train_x**3 + 5*np.random.rand(*train_x.shape)

p0 = np.expand_dims(np.array([1,1,1,1], dtype=float), 0)
print(p0)

p, hist = train_model(p0, model, train_x, train_y, mse, grad, 0.002, 8, 100)

print(p)

plt.figure()
plt.plot(hist)
plt.show()

plt.figure()
plt.plot(train_x[0,:], train_y[0,:])
plt.plot(train_x[0,:], model(p, train_x)[0,:])
plt.grid()
plt.show()