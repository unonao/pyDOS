import scipy.io as sio
from scipy.sparse import dok_matrix


def load_graph_from_mat(path, filename, mname='A'):
    """
    Load a graph matrix from Matlab file
    """
    data = sio.loadmat(path+filename)
    return data[mname]


def load_graph_from_list(path, filename, is_weight=False, mname='A'):
    """
    Load a graph matrix from adjacency list(txt)
    The ajacency list must be 0-indexed!
    """
    filename += '.txt'
    if is_weight:
        G = []
        with open(path+filename) as f:
            for line in f.readlines():
                G.append(list(map(int, line.split())))
        dct = {}
        N = 0
        for u, v, weight in G:
            dct[(u, v)] = weight
            dct[(v, u)] = weight
            if N < u:
                N = u
            if N < v:
                N = v
        N += 1
        smat = dok_matrix((N, N))
        for (u, v), w in dct.items():
            smat[u, v] = w
    else:
        G = []
        with open(path+filename) as f:
            for line in f.readlines():
                G.append(list(map(int, line.split())))
        dct = {}
        N = 0
        for u, v in G:
            dct[(u, v)] = 1
            dct[(v, u)] = 1
            if N < u:
                N = u
            if N < v:
                N = v
        N += 1
        smat = dok_matrix((N, N))
        for (u, v), w in dct.items():
            smat[u, v] = w
    return smat
