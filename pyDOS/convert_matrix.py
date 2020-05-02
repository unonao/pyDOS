import numpy as np
import copy as cp
import scipy.sparse as ss


def convert_into_laplacian(A, axis=1):
    """
    Convert a weighted adjacency matrix(sparce matrix) into a Laplacian
    return L (= D-A)
    (D is the degree matrix)

    Args:
        A: weighted adjacency matrix
        axis: Compute D over the given axis.
    """

    D = np.asarray(A.sum(axis))  # Compute D over the given axis
    L = -A
    L.setdiag(D.squeeze())
    return L
