import numpy as np
from .utils import *
import matplotlib.pyplot as plt
from .Integral import L2_norm_square


def phi_table(x, interval, m):
    x = np.array(x)  # Ensure x is a NumPy array
    s = (2*x-interval[1]-interval[0])/(interval[1]-interval[0])
    return legtable(s, m)

def DG(f, interval, N, k=5, m=10):
    h = (interval[1]-interval[0])/N
    I = [[n*h+interval[0], (n+1)*h+interval[0]] for n in range(N)]

    c = []

    s, w  = gauleg(m)
    P = legtable(s, k)

    for n in range(N):
        t = variable_transformation(s, I[n])
        cn = np.zeros(k+1)
        for i in range(k+1):
            cn[i] = (2*i+1)/2*np.dot(f(t)*w, P[i])
        c.append(cn)
        
    c = np.array(c)
    def fh(x, n):
        phi = phi_table(x, I[n], k)

        value = np.dot(c[n].reshape(1,-1), phi).reshape(-1,)

        return value

    return fh, I

def computing_DG_method_error(f, fh, intervals, N, m=10):
    error = 0
    for n in range(N):
        I = intervals[n]
        diff = lambda x: f(x)-fh(x, n)
        error += L2_norm_square(diff, I[0], I[1], m)
    error = np.sqrt(error)
    return error

if __name__ == '__main__':
    def f(x):
        return x**3
    interval = [0, 1]
    N = 1
    k = 1
    fh, I = DG(f, interval, N, k)
    x = np.linspace(interval[0], interval[1], 100)
    for n in range(N):
        xn = np.linspace(I[n][0], I[n][1], 100//N)
        plt.plot(xn, fh(xn, n), label='fh' if n == 0 else '', color='red')
    
    plt.plot(x, f(x), label='f', color='blue')
    plt.legend()
    plt.savefig('img/DG.png')    
