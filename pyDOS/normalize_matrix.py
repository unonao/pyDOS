"""
This module is a colletion of functions that shift and rescale the symmetrix matrix H. (the eigenvalue range between -1 and 1)
"""

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssla


def normalize_matrix(A):
    """
	Normalize a weighted adjacency matrix.(Symmetric scaling by the degree)
    D^(-1/2)AD^(-1/2)

	Args:
		A: weighted adjacency matrix
    Outs:
        A: normalized sparse matrix
	"""
    dc = np.asarray(A.sum(0)).squeeze()
    dr = np.asarray(A.sum(1)).squeeze()
    [i, j, wij] = ss.find(A)
    wij = wij / np.sqrt(dr[i] * dc[j])

    A = ss.csr_matrix((wij, (i, j)), shape=A.shape)
    return A


def shift_and_rescale_matrix(H):
    '''
    The spectra of the given graph matrix are changed on the interval [-1,1] by shifting and rescaling.
    [-1,1]を超えるので不採用→なぜ？
    Outs:
        H: normalized matrix(not sparse)
    '''
    max_val = ssla.eigsh(H, k=1, which='LA', return_eigenvectors=False)
    min_val = ssla.eigsh(H, k=1, which='SA', return_eigenvectors=False)
    print(max_val)
    H = (2 * H - (max_val + min_val)) / (max_val - min_val)
    return H
