import numpy as np
import matplotlib.pyplot as plt

def mag(x, buff=None):
    buff = np.empty(x.shape) if buff is None else buff
    np.square(x, out=buff)
    return np.sqrt(buff.sum(axis=0))

def abse(a, b, buff=None):
    buff = np.empty(a.shape) if buff is None else buff
    np.subtract(a,b, out=buff)
    np.abs(buff, out=buff)
    return buff.mean(axis=1)

def grad_abse(a,b, buff=None):
    buff = np.empty(a.shape) if buff is None else buff
    np.subtract((a>=b)*1, (a<b)*1, out=buff)
    return buff

def line(a, b, buff=None):
    buff = np.empty(a.shape) if buff is None else buff
    np.subtract(a,b, out=buff)
    return abs(buff.mean(axis=1))

def grad_line(a,b, buff=None):
    buff = np.empty(a.shape) if buff is None else buff
    lin_e = np.subtract(a,b, out=buff).mean(axis=0)
    trans = 1*(lin_e >= 0) - 1*(lin_e < 0)
    buff2 = np.ones_like(a)
    return np.multiply(buff2, trans, buff)

def mse(a, b, buff=None, axis=None):
    buff = np.empty(a.shape) if buff is None else buff
    np.subtract(a,b, out=buff)
    np.square(buff, out=buff)
    return buff.mean(axis=axis)

def grad_mse(a,b, buff=None):
    buff = np.empty(a.shape) if buff is None else buff
    np.subtract(a,b, out=buff)
    return buff

def rmse(a,b, buff=None, axis = None):
    _mse = mse(a, b, buff, axis = axis)
    return np.sqrt(_mse)

def iqr(a, axis=None):
    return np.percentile(a, 75, axis=axis) - np.percentile(a, 25, axis=axis)

def rpiq(a,b, buff=None, axis=None):
    return iqr(b, axis=axis)/rmse(a,b, buff, axis=axis)

def spc_angle(a,b, buff=None):
    buff = np.empty(a.shape) if buff is None else buff
    np.square(a, out=buff)
    a_2 = buff.sum(axis=0)
    np.square(b, out=buff)
    b_2 = buff.sum(axis=0)
    np.multiply(a, b, out=buff)
    a_2 = np.multiply(a_2, b_2)
    a_2 = np.sqrt(a_2)
    return 1 - buff.sum(axis=0)/a_2

def grad_spc_angle(a, b, buff=None):
    buff = np.empty(a.shape) if buff is None else buff
    np.square(a, out = buff)
    mag_a = np.sum(buff, axis = 0)
    np.square(b, out = buff)
    mag_b = np.sum(buff, axis=0)
    buff2 = np.empty(mag_a.shape)
    np.square(mag_a, out=buff2)
    np.multiply(b, buff2, out=buff)
    np.subtract(a, buff, out=buff)
    np.add(buff, a, out=buff)
    np.multiply(buff2, mag_a, out=buff2)
    np.multiply(buff2, mag_b, out=buff2)
    np.divide(buff, buff2, out=buff)
    return buff

def mixing(S, a, buff=None):
    buff = np.empty((S.shape[0], a.shape[1])) if buff is None else buff
    np.dot(S, a, out=buff)
    return buff

def grad_mixing(S, a, buff=None):
    return a.T

def model_error(p, model, x, y, error):
    return error(model(x,p), y)

def gen_batches(x , y, n):
    div, res = divmod(x.shape[1], n)
    out = []
    for i in range(div):
        out.append((x[:, i*n:(i+1)*n], y[:, i*n:(i+1)*n]))
    out.append((x[:, -res:], y[:, -res:]))
    return tuple(out)

def train_model(p0, model, x, y, error, grad, lr, batch_size, epochs=10, tol=1e-2):
    batches = gen_batches(x,y, batch_size)
    history = np.empty((epochs, ))
    p = p0.copy()
    for i in range(epochs):
        p, _ = SGD(grad, p, lr, batches)
        history[i] = error(model(p,x), y).mean()
    return p, history

def unmixing(R, S, alpha=0.001, error=mse, grad_error=grad_mse, C=10, max_iter=100000, tol=1e-7, sum_1=True):
    a = np.random.rand(S.shape[1], R.shape[1])/S.shape[1]
    grad_a = np.empty(a.shape)
    grad_e = np.empty(R.shape)
    buff_e = np.empty(R.shape)
    buff_a = np.empty(a.shape)
    buff_a2 = np.empty(a.shape)
    ones = np.ones(a.shape)
    buff_aS = np.empty(a.shape[1])
    _R_ = np.empty(R.shape)
    history = np.empty((max_iter, ))
    for i in range(max_iter):
        np.dot(S,a, out=_R_)
        history[i] = error(_R_, R, buff_e)
        grad_error(_R_, R, grad_e)
        np.dot(S.T, grad_e, out=grad_a)
        
        np.sum(a, axis=0, out=buff_aS)
        np.subtract(buff_aS, ones, out=buff_a)
        if not sum_1:
            np.greater(buff_aS, ones, out=buff_a2)
            np.multiply(buff_a2, buff_a, out=buff_a)
        np.multiply(C, buff_a, out=buff_a)
        np.add(grad_a, buff_a, out=grad_a)
        np.multiply(alpha, grad_a, out=grad_a)
        # Update a
        np.subtract(a, grad_a, out=a)
        mag_grad_a = np.max(np.abs(grad_a))
        #print("grad-a: {:.7f}".format(mag_grad_a), end='\r')
        if mag_grad_a < tol:
            return a, history[0:i+1]
    print("Warning: Not convergence")
    return a, history

def SGD(grad_f, x, alpha, batches):
    n_batches = len(batches)
    for j in range(n_batches):
        grad = grad_f(x, *batches[j])
        np.multiply(grad, alpha, out=grad)
        np.subtract(x, grad, out=x)
    return x, grad

def poly(n, p):
    cp = [0 for i in range(n-1)] + [p]
    powers = [cp]
    while cp[0] < p:
        cp = cp.copy()
        i = n-1
        while cp[i] == 0:
            i-=1
        cp[i-1], cp[i], cp[n-1] = cp[i-1]+1, 0, cp[i]-1
        powers.append(cp)
    return powers
 
def poly_transform(x, p, pf=1, bias=False):
    p = int(p//pf)
    powers = []
    for g in range(1, p+1):
        powers += poly(x.shape[1], g)
    powers = np.array(powers)*pf
    out = np.ones((x.shape[0], powers.shape[0]))
    buff = np.empty(x.shape)
    for i in range(out.shape[1]):
        np.prod(np.power(x, powers[i,:], out=buff), axis=1, out=out[:,i])
    if bias:
        out = np.hstack((out, np.ones((x.shape[0],1))))
    return out, powers.T

def PMF(R, rang=None, a=None, b=None, error=mse, grad_error=grad_mse, a_init=None, b_init=None, alpha=0.002, max_iter=300000, tol=1e-6):
    optimize_a, optimize_b = False, False
    if rang is None:
        if not (a is None):
            rang = a.shape[1]
        elif not (b is None):
            rang = b.shape[0]
        else:
            raise ValueError("PMF need rang value")
    if rang < 1:
        raise ValueError("PMF should rang >= 1")
    if a is None:
        a = np.random.rand(R.shape[0], rang)/R.shape[0] if a_init is None else a_init.copy()
        optimize_a = True
    if b is None:
        b = np.random.rand(rang, R.shape[1])/R.shape[1] if b_init is None else b_init.copy()
        optimize_b = True
    if a.shape[1] != rang or b.shape[0] != rang or a.shape[0] != R.shape[0] or b.shape[1] != R.shape[1]:
        raise ValueError("Shapes Error: PMF require R.shape=[M,N], a.shape=[M,rang], b.shape=[rang,N]")
    hist = np.empty(max_iter)
    mag_grad_a, mag_grad_b = 0, 0
    for i in range(max_iter):
        _R_ = a @ b
        hist[i] = error(_R_, R)
        if optimize_a:
            grad_a = grad_error(_R_, R) @ b.T + a*(a<0)
            a -= alpha*grad_a
            mag_grad_a = np.max(np.abs(grad_a))
        if optimize_b:
            grad_b = a.T @ grad_error(_R_, R) + b*(b<0)
            b -= alpha*grad_b
            mag_grad_b = np.max(np.abs(grad_b))
        print("grad-a: {:.7f}\t grad-b: {:.7f}".format(mag_grad_a, mag_grad_b), end='\r')
        if max(mag_grad_a, mag_grad_b) < tol:
            return a, b, hist[0:i+1]
    print("Warning: Not convergence")
    return a, b, hist

def MF(R, rang=None, a=None, b=None, error=mse, grad_error=grad_mse, a_init=None, b_init=None, alpha=0.002, max_iter=300000, tol=1e-6):
    optimize_a, optimize_b = False, False
    if rang is None:
        if not (a is None):
            rang = a.shape[1]
        elif not (b is None):
            rang = b.shape[0]
        else:
            raise ValueError("PMF need rang value")
    if rang < 1:
        raise ValueError("PMF should rang >= 1")
    if a is None:
        a = np.random.rand(R.shape[0], rang)/R.shape[0] if a_init is None else a_init.copy()
        optimize_a = True
    if b is None:
        b = np.random.rand(rang, R.shape[1])/R.shape[1] if b_init is None else b_init.copy()
        optimize_b = True
    if a.shape[1] != rang or b.shape[0] != rang or a.shape[0] != R.shape[0] or b.shape[1] != R.shape[1]:
        raise ValueError("Shapes Error: PMF require R.shape=[M,N], a.shape=[M,rang], b.shape=[rang,N]")
    hist = np.empty(max_iter)
    mag_grad_a, mag_grad_b = 0, 0
    for i in range(max_iter):
        _R_ = a @ b
        hist[i] = error(_R_, R)
        if optimize_a:
            grad_a = grad_error(_R_, R) @ b.T
            a -= alpha*grad_a
            mag_grad_a = np.max(np.abs(grad_a))
        if optimize_b:
            grad_b = a.T @ grad_error(_R_, R)
            b -= alpha*grad_b
            mag_grad_b = np.max(np.abs(grad_b))
        if max(mag_grad_a, mag_grad_b) < tol:
            return a, b, hist[0:i+1]
    print("Warning: Not convergence")
    return a, b, hist

def unmixPMF(R, rang, error=mse, grad_error=grad_mse, a_init=None, b_init=None, C=2, alpha=0.001, max_iter=2000000, tol=1e-6, sum_1=True):
    if rang < 1:
        raise ValueError("PMF should rang >= 1")
    a = np.random.rand(R.shape[0], rang)/R.shape[0] if a_init is None else a_init.copy()
    b = np.random.rand(rang, R.shape[1])/R.shape[1] if b_init is None else b_init.copy()
    if a.shape[1] != rang or b.shape[0] != rang or a.shape[0] != R.shape[0] or b.shape[1] != R.shape[1]:
        raise ValueError("Shapes Error: PMF require R.shape=[M,N], a.shape=[M,rang], b.shape=[rang,N]")
    hist = np.empty(max_iter)
    grad_e = np.empty(R.shape)
    grad_a = np.empty(a.shape)
    grad_b = np.empty(b.shape)
    buff_e = np.empty(R.shape)
    buff_a = np.empty(a.shape)
    buff_b = np.empty(b.shape)
    buff_b2 = np.empty(b.shape)
    ones = np.ones(a.T.shape)
    buff_aS = np.empty(a.shape[0])
    buff_aT = np.empty(a.T.shape)
    buff_aT2 = np.empty(a.T.shape)
    _R_ = np.empty(R.shape)
    for i in range(max_iter):
        np.dot(a, b, out=_R_)
        hist[i] = error(_R_, R, buff_e)
        if np.isnan(hist[i]) or np.isinf(hist[i]):
            print("Error: overflow")
            raise ValueError("Overflow")
        # Calc Grad Error
        grad_error(_R_, R, grad_e)
        
        # Calc grad a: grad_a = grad_error(_R_, R, buff1) @ b.T + C*(a*(a<0) + (a.sum(axis = 1)-ones).T[*(a.sum(axis = 1)>1)])
        np.dot(grad_e, b.T, grad_a)
        np.less(a, 0, out=buff_a)
        np.multiply(a, buff_a, out=buff_a)
        np.add(grad_a, buff_a, out=grad_a)
        np.sum(a, axis=1, out=buff_aS)
        np.subtract(buff_aS, ones, out=buff_aT)
        if not sum_1:            
            np.greater(buff_aS, ones, buff_aT2)
            np.multiply(buff_aT2, buff_aT, out=buff_aT)
        np.multiply(buff_aT, C, out=buff_aT)
        np.add(grad_a, buff_aT.T, out=grad_a)
      
        # Calc grad b: grad_b = a.T @ grad_error(_R_, R)
        np.dot(a.T, grad_e, out=grad_b)

        # Calc grad b: grad_b = a.T @ grad_error(_R_, R) + C*(b*(b<0) + (b-1)*(b>1))
        # np.dot(a.T, grad_e, out=grad_b)
        # np.less(b, 0, out=buff_b)
        # np.multiply(b, buff_b, out=buff_b)
        # np.multiply(C, buff_b, out=buff_b)
        # np.add(grad_b, buff_b, out=grad_b)
        # np.greater(b, 1, out=buff_b)
        # np.subtract(b, 1, out=buff_b2)
        # np.multiply(buff_b, buff_b2, out=buff_b)
        # np.multiply(C, buff_b, out=buff_b)
        # np.add(grad_b, buff_b, out=grad_b)
        
        # Update a and b
        np.multiply(alpha, grad_a, out=grad_a)
        np.multiply(alpha, grad_b, out=grad_b)
        np.subtract(a, grad_a, out=a)
        np.subtract(b, grad_b, out=b)

        # Rectify
        # np.maximum(0, b, out=b)
        # np.maximum(0, a, out=a)

        # validate convergence
        np.abs(grad_a, out=buff_a)
        mag_grad_a =  np.max(buff_a)
        np.abs(grad_b, out=buff_b)
        mag_grad_b =  np.max(buff_b)
        print("grad-a: {:.7f}\t grad-b: {:.7f}".format(mag_grad_a, mag_grad_b), end='\r')
        if max(mag_grad_a, mag_grad_b) < tol:
            return a,b, hist[0:i+1]
    print("Warning: Not convergence")
    return a,b, hist

def elbow(_y_):
    y = np.array(_y_)
    y = y*y.shape[0]/np.max(y)
    d_ang = np.empty(len(y)-2)
    for i in range(1, len(y)-2):
        ang1 = np.arctan2(y[i] - y[0], i)
        ang2 = np.arctan2(y[-1] - y[i], len(y)-1 - i)
        d_ang[i-1] = abs(ang1 - ang2)
    return np.argmax(d_ang) + 1

def corrAbs(x):
    cm = np.abs(np.corrcoef(x))
    correlations = []
    for i in range(cm.shape[0]):
        for j in range(i+1,cm.shape[1]):
            correlations.append(cm[i,j])
    return np.array(correlations)

def corrAbsMean(x):
    return np.mean(corrAbs(x))

def corrAbsMedian(x):
    return np.median(corrAbs(x))

def corrAbsMin(x):
    return np.min(corrAbs(x))

def corrAbsMax(x):
    return np.max(corrAbs(x))

def plotHistory(title, y_label, hist, path):
    plt.figure()
    plt.title(title)
    plt.plot(range(1,len(hist) + 1), hist)
    plt.xlabel("Iteración")
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(path + '/' + title + '.png')
    plt.close()

def plot(title, x, y, path, xlabel=None, ylabel=None, xlim=None, ylim=None):
    plt.figure()
    plt.title(title)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid()
    plt.savefig(path + '/' + title + '.png')
    plt.close()

def plotEndmembers(title, endmembers, wavelength, path):
    plt.figure()
    plt.title(title)
    for i in range(endmembers.shape[0]):
        plt.plot(wavelength, endmembers[i,:])
    plt.xlim((np.min(wavelength), np.max(wavelength)))
    plt.xlabel("Longitud de Onda (nm)")
    plt.ylabel("Reflectancia")
    plt.grid()
    plt.savefig(path + '/' + title + '.png')
    plt.close()

def plotRangesError(title, start, errors, ylabel, path, elbow_rang=None):
    plt.figure()
    plt.title(title)
    plt.plot(range(start, errors.shape[0]+start), errors)
    if not (elbow_rang is None):
        plt.plot([elbow_rang, elbow_rang], [0, np.max(errors)])
    plt.xlabel("Rango")
    plt.xticks(range(1, errors.shape[0]+1))
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(path+'/'+title + ".png")
    plt.close()

def plotScatter(title, x, y, xlabel, ylabel, path, xlim=None, ylim=None):
    plt.figure()
    plt.title(title)
    plt.plot([0, 1], [0, 1])
    if len(x.shape) == 1:
        plt.plot(x, y, '+')
    else:
        for i in range(x.shape[1]):
            plt.plot(x[:,i], y[:,i], '+')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(path + '/' + title + '.png')
    plt.close()

def plotBoxes(title, data, path, xticks=None):
    plt.figure()
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data)
    plt.xticks(range(1, len(data) + 1), xticks)
    plt.savefig(path + '/' + title + '.png')
    plt.close()
