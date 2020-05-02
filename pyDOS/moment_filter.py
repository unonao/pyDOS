import numpy as np
import scipy.sparse as ss


def filter_jackson(c):
    """
	Apply the Jackson filter to a sequence of Chebyshev	moments. The moments
	should be arranged column by column.

	Args:
		c: Unfiltered Chebyshev moments

	Output:
		cf: Jackson filtered Chebyshev moments
	"""

    M = c.shape[0]
    n = np.arange(M)
    tau = np.pi / (M + 1)
    g = ((M - n + 1) * np.cos(tau * n) + np.sin(tau * n) / np.tan(tau)) / (M +
                                                                           1)
    g.shape = (M, 1)
    c = g * c
    return c
