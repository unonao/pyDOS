import os
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import scipy.sparse as ss


def load_graph(filepath):
    """
    Load a graph matrix
    Args:
        filepath: filepath. It does not matter whether file is 'smat', 'mat' or 'txt'.
    """
    (filepath, ext) = os.path.splitext(filepath)
    if ext == '.smat':
        return load_graph_from_smat(filepath)
    elif ext == '.mat':
        return load_graph_from_mat(filepath)
    elif ext == '.txt':
        return load_graph_from_txt(filepath)


def load_graph_from_smat(filepath):
    """
    Load a graph matrix from graph-eids-v1
    Args:
        filepath: filepath without extension(.mat)
        mname: matrix name
    """
    # load graph
    data = np.loadtxt(filepath + '.smat')
    data = data[1:, :]
    row = np.append(data[:, 0], data[:, 1])
    col = np.append(data[:, 1], data[:, 0])
    row = row.astype(int)
    col = col.astype(int)
    if data.ndim == 3:  # weighted graph
        weight = np.append(data[:, 2], data[:, 2])
    else:  # unweighted graph
        weight = np.ones(2 * data.shape[0])
    H = ss.csr_matrix((weight, (row, col)))

    # load normalized eigenvalues (if exist)
    eig_vals = None
    if os.path.isfile(filepath + '.smat.normalized.eigs'):
        eig_vals = np.loadtxt(filepath + '.smat.normalized.eigs')

    # compute eigenvalues (if not exist & possible)
    if eig_vals is None:
        if N > 1e3:
            print('Graph size is too large to compute true eigenvalues.')
        else:
            eig_vals = LA.eigvalsh(H.toarray())
    else:
        eig_vals = 1 - eig_vals  # eig_vals are eigenvalues of the Normalized Laplacian Matrix

    return H, eig_vals


def load_graph_from_mat(filepath, mname='A'):
    """
    Load a graph matrix from Matlab file
    Args:
        filepath: filepath without extension(.mat)
        mname: matrix name
    """
    data = sio.loadmat(filepath)
    return ss.csr_matrix(data[mname]), data["lambda"].flatten()


def load_graph_from_txt(filepath):
    """
    Load a graph matrix from adjacency list(txt)
    The ajacency list must be 0-indexed!
    """
    data = np.loadtxt(filepath + '.txt')
    row = np.append(data[:, 0], data[:, 1])
    col = np.append(data[:, 1], data[:, 0])
    row = row.astype(int)
    col = col.astype(int)
    if data.ndim == 3:  # weighted graph
        weight = np.append(data[:, 2], data[:, 2])
    else:  # unweighted graph
        weight = np.ones(2 * data.shape[0])
    return ss.csr_matrix((weight, (row, col))), None
