import os
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import scipy.sparse as ss
from scipy.sparse import SparseEfficiencyWarning
import warnings
#warnings.simplefilter('ignore', SparseEfficiencyWarning)


def load_graph(filepath):
    """
    Load a graph matrix as an undirected graph
    Args:
        filepath: filepath. It does not matter whether file extention is 'smat', 'mat' or 'txt'.

    """
    head_tail = os.path.split(filepath)
    (name, ext) = os.path.splitext(head_tail[-1])
    if ext == '.smat':
        return load_graph_from_smat(filepath)
    elif ext == '.mat':
        return load_graph_from_mat(filepath)
    elif ext == '.txt':
        return load_graph_from_txt(filepath)
    elif ext == '.csv':
        return load_graph_from_csv(filepath)
    else:
        if name == 'out':  # for konect data
            return load_graph_from_konect(filepath)


def load_graph_from_smat(filepath):
    """
    Load a graph matrix from graph-eids-v1
    Args:
        filepath: filepath without extension(.mat)
        mname: matrix name
    """
    # load graph
    data = np.loadtxt(filepath, dtype=int)
    N = data[0, 0]
    data = data[1:, :]
    row = data[:, 0]
    col = data[:, 1]
    if len(data[0]) == 3:  # weighted graph
        weight = data[:, 2]
    else:  # unweighted graph
        weight = np.ones(data.shape[0])
    A = ss.csr_matrix((weight, (row, col)), shape=(N, N)).tolil()
    rows, cols = A.nonzero()
    A[cols, rows] = A[rows, cols]
    A = A.tocsr()
    A = remove_zero_entries(A)

    # load normalized eigenvalues (if exist)
    eig_vals = None
    if os.path.isfile(filepath + '.normalized.eigs'):
        eig_vals = np.loadtxt(filepath + '.normalized.eigs')

    # compute eigenvalues (if not exist & possible)
    if eig_vals is None:
        if N > 1e3:
            print('Graph size is too large to compute true eigenvalues.')
        else:
            eig_vals = LA.eigvalsh(H.toarray())
    else:
        eig_vals = 1 - eig_vals  # eig_vals are eigenvalues of the Normalized Laplacian Matrix

    return A, eig_vals


def load_graph_from_mat(filepath, mname='A'):
    """
    Load a graph matrix from Matlab file
    Args:
        filepath: filepath without extension(.mat)
        mname: matrix name
    """
    (filepath, ext) = os.path.splitext(filepath)
    data = sio.loadmat(filepath)

    # matrix
    if "Problem" in data.keys():  # for The SuiteSparse Matrix Collection
        H = ss.csr_matrix(data["Problem"][0, 0]["A"])
    else:  # for snap
        H = ss.csr_matrix(data[mname])

    # lambda
    if "lambda" in data.keys():
        return H, data["lambda"].flatten()
    else:
        return H, None


def load_graph_from_txt(filepath):
    """
    Load a graph matrix from adjacency list(txt)
    The ajacency list must be 0-indexed!
    """
    data = np.loadtxt(filepath, dtype=int, comments='#')
    row = data[:, 0]
    col = data[:, 1]
    N = max(row.max(), col.max()) + 1
    if len(data[0]) == 3:  # weighted graph
        weight = data[:, 2]
    else:  # unweighted graph
        weight = np.ones(data.shape[0])
    A = ss.csr_matrix((weight, (row, col)), shape=(N, N)).tolil()
    rows, cols = A.nonzero()
    A[cols, rows] = A[rows, cols]
    A = A.tocsr()
    A = remove_zero_entries(A)
    return A, None


def load_graph_from_csv(filepath):
    """
    Load a graph matrix from adjacency list(txt)
    The ajacency list must be 0-indexed!
    """
    data = np.loadtxt(filepath, dtype=int, delimiter=',', skiprows=1)
    row = data[:, 0]
    col = data[:, 1]
    N = max(row.max(), col.max()) + 1
    if len(data[0]) == 3:  # weighted graph
        weight = data[:, 2]
    else:  # unweighted graph
        weight = np.ones(data.shape[0])
    A = ss.csr_matrix((weight, (row, col)), shape=(N, N)).tolil()
    rows, cols = A.nonzero()
    A[cols, rows] = A[rows, cols]
    A = A.tocsr()
    A = remove_zero_entries(A)
    return A, None


def load_graph_from_konect(filepath):
    """
    Load a graph matrix (konect)
    The ajacency list must be 1-indexed!
    """
    data = np.loadtxt(filepath, dtype=int, comments='%')
    row = data[:, 0]
    col = data[:, 1]
    row = row - 1
    col = col - 1
    N = max(row.max(), col.max()) + 1
    if len(data[0]) == 3:  # weighted graph
        weight = data[:, 2]
    else:  # unweighted graph
        weight = np.ones(data.shape[0])
    A = ss.csr_matrix((weight, (row, col)), shape=(N, N)).tolil()
    rows, cols = A.nonzero()
    A[cols, rows] = A[rows, cols]
    A = remove_zero_entries(A)
    return A, None


def remove_zero_entries(A):
    A = A.tocsr()
    A = A[A.getnnz(1) > 0]
    A = A[:, A.getnnz(0) > 0]
    return A


if __name__ == "__main__":
    data = np.array([[0, 1], [0, 2], [0, 3], [1, 0], [1, 3], [2, 0], [3, 0],
                     [3, 1], [10, 1], [20, 3]])
    row = data[:, 0]
    col = data[:, 1]
    N = max(row.max(), col.max()) + 1
    if len(data[0]) == 3:  # weighted graph
        weight = data[:, 2]
    else:  # unweighted graph
        weight = np.ones(data.shape[0])
    A = ss.csr_matrix((weight, (row, col)), shape=(N, N)).tolil()
    rows, cols = A.nonzero()
    A[cols, rows] = A[rows, cols]
    A = A.tocsr()
    A = remove_zero_entries(A)
    print(A.toarray())

    data = np.array([[0, 1, 2], [0, 2, 2], [0, 3, 2], [1, 3, 2]])
    row = data[:, 0]
    col = data[:, 1]
    if len(data[0]) == 3:  # weighted graph
        weight = data[:, 2]
    else:  # unweighted graph
        weight = np.ones(data.shape[0])
    A = ss.csr_matrix((weight, (row, col)), shape=(N, N)).tolil()
    rows, cols = A.nonzero()
    A[cols, rows] = A[rows, cols]
    A = A.tocsr()
    A = remove_zero_entries(A)
    print(A.toarray())
