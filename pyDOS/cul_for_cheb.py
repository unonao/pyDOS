import numpy as np
import scipy.sparse.linalg as ssla


def cul_for_chebhist(d, X):
    """
	Given a set of first-kind Chebyshev moments, compute the associated density.
	Output a histogram of cumulative density function by default.

	Args:
		d: Vector of Chebyshev moments (on [-1,1])
		X: Evaluation points

	Output:
        Y: Estimated counts on buckets between X points
	"""

    # Compute CDF and bin the difference
    Y = cul_for_chebcum(d, X)
    Y = Y[1:] - Y[:-1]

    return Y


def cul_for_chebcum(d, X):
    """
	Given a (filtered) set of first-kind Chebyshev moments, compute the integral
	of the density:
		int_0^s (2/pi)*sqrt(1-x^2)*( d(0)/2+sum_{n=1}^{N-1}c_nT_n(x) )
	Output a plot of cumulative density function by default.

	Args:
		d: Array of Chebyshev moments (on [-1,1])
		X: Evaluation points (defaults to mesh of 1001 pts)

	Output:
		Y: Estimated cumulative density up to each X point
	"""

    M = len(d)
    txx = np.arccos(X)
    Y = d[0] * (txx - np.pi) / 2
    for idx in np.arange(1, M):
        Y += d[idx] * np.sin(idx * txx) / idx

    Y *= -2 / np.pi

    return Y


def cul_for_cheb_density(d, X):
    """
	Given a set of first-kind Chebyshev moments, compute the associated density.

	Args:
		d: Vector of Chebyshev moments (on [-1,1])
		X: Evaluation points

	Output:
		Y: Density evaluated at X mesh
	"""

    # Run the recurrence
    M = len(d)
    P0 = np.ones(len(X))
    P1 = X
    Y = d[0] + d[1] * X
    print((2 * (X * P1) - P0).shape, Y.shape)
    for idx in np.arange(2, M):
        Pn = 2 * (X * P1) - P0
        Y += d[idx] * Pn
        P0 = P1
        P1 = Pn

    Y = (2 / np.pi) * (Y / (1e-12 + np.sqrt(1 - X**2)))

    Y.reshape([1, -1])
    return Y
