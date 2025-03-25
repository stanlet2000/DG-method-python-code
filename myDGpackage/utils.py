import numpy as np

def legtable(x, m):
    """
    Computes the Legendre polynomial values at given points x up to order m.

    Parameters
    ----------
    x : numpy array
        Row vector of points in (-1,1)
    m : int
        Maximum order of the Legendre polynomials

    Returns
    -------
    P : numpy array
        A (m+1, len(x)) matrix containing the values of the Legendre polynomials at points x
    """
    x = np.array(x)  # Ensure x is a NumPy array
    l = len(x)
    P = np.ones((m + 1, l))  # Initialize matrix P with ones

    if m > 0:
        P[1, :] = x  # Set P1(x) = x

        for i in range(1, m):  # Start from P2(x)
            P[i + 1, :] = ((2 * i + 1) * x * P[i, :] - i * P[i - 1, :]) / (i + 1)

    return P



def gauleg(n):
    """
    Computes the Gauss-Legendre quadrature points and weights.

    Parameters
    ----------
    n : int
        Number of quadrature points

    Returns
    -------
    x : numpy array
        Quadrature points (nodes)
    w : numpy array
        Quadrature weights
    """
    x = np.zeros(n)
    w = np.zeros(n)
    m = (n + 1) // 2  # Only need to compute half due to symmetry

    for i in range(m):
        # Initial guess using Chebyshev roots
        z = np.cos(np.pi * (i + 0.75) / (n + 0.5))
        
        # Newton's method to find root of Legendre polynomial
        while True:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j

            # Compute derivative P_n'(z)
            pp = n * (z * p1 - p2) / (z**2 - 1.0)

            z1 = z
            z = z1 - p1 / pp  # Newton's method update

            if abs(z - z1) < np.finfo(float).eps:  # Convergence check
                break
        
        # Store results with symmetry
        x[i] = -z
        x[n - 1 - i] = z
        w[i] = 2.0 / ((1.0 - z**2) * pp**2)
        w[n - 1 - i] = w[i]

    return x, w

def variable_transformation(x, interval):
    """
    Transforms the variable x from (-1, 1) to (a, b).

    Parameters:
    x : numpy array
        Variable to be transformed
    interval : tuple
        Tuple (a, b) representing the interval (a, b)

    Returns:
    y : numpy array
        Transformed variable
    """
    a, b = interval
    return (b - a) / 2 * x + (a + b) / 2


if __name__ == '__main__':
    x = np.array([-0.5, 0, 0.5])  # Sample points
    m = 3  # Compute Legendre polynomials up to P3(x)
    P = legtable(x, m)
    print(P[1][1])

    n = 2
    x, w = gauleg(n)
    print("Nodes:", x)
    print("Weights:", w)
