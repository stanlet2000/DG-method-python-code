import numpy as np
from .utils import *

def Gaussian_quadratrue_origin(func, degree=3):
    """
    Compute the integral of a function f(x) using Gauss-Legendre quadrature.
    
    Parameters
    ----------
    func : function
        Function to be integrated
    degree : int
        Degree of the Gauss-Legendre quadrature

    Returns
    -------
    value : float
        Integral of the function f(x) using Gauss-Legendre quadrature
    """

    xs, ws = gauleg(degree)
    value = np.dot(func(xs), ws).sum()
    
    return value


def Gaussian_quadratrue(func, a, b, degree=3):
    """
    Compute the integral of a function f(x) over the interval (a, b) using Gauss-Legendre quadrature.

    Parameters
    ----------
    func : function
        Function to be integrated
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval
    degree : int
        Degree of the Gauss-Legendre quadrature

    Returns
    -------
    value : float
        Integral of the function f(x) over the interval (a, b) using Gauss-Legendre quadrature
    """

    new_f = lambda x: func((b-a)/2*x+(a+b)/2)
    value = Gaussian_quadratrue_origin(new_f, degree)

    return (b-a)/2 * value

def L2_norm_square(func, a, b, degree=3):
    """
    Compute the L2 norm square of a function f(x) over the interval (a, b).
    
    Parameters
    ----------
    func : function
        Function to be integrated
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval
    degree : int
        Degree of the Gauss-Legendre quadrature
    
    Returns
    -------
    value : float
        L2 norm square of the function f(x) over the interval (a, b)
    """

    value = Gaussian_quadratrue(lambda x: func(x)**2, a, b, degree)
    return value


if __name__ == '__main__':
    # f = lambda x: x**2
    f = np.cos
    # f = lambda x: legtable(x, 3)[1][1]
    print(f(0))
    print(Gaussian_quadratrue_origin(np.exp,7))
    print(Gaussian_quadratrue(np.exp, 0, 1, 5))