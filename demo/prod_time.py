import contextlib
import time
import scipy.sparse.linalg as ssla
import numpy.linalg as LA
import numpy as np
import os
import sys
sys.path.append('../')
args = sys.argv


@contextlib.contextmanager
def simple_timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.03f} s')


if __name__ == '__main__':
    from pyDOS import load_graph, normalize_matrix
    filepath = '../data/graph-eigs-v1/Erdos02-cc.smat' if len(
        args) <= 1 else args[1]
    # load graph network
    (H, true_eig_vals) = load_graph(filepath)
    N = H.shape[0]
    print("file: " + filepath)
    print('V:\t{}'.format(N))
    print('E:\t{}'.format(H.count_nonzero() // 2))
    print('average degree:\t{}'.format(H.count_nonzero() / N))

    # normalize matrix
    H = normalize_matrix(H)

    with simple_timer('H2'):
        H2 = H*H
    print('V:\t{}'.format(N))
    print('E:\t{}'.format(H2.count_nonzero() // 2))
    print('average degree:\t{}'.format(H2.count_nonzero() / N))
    print('eigenvalues^2:', H2.diagonal().sum())

    with simple_timer('H4'):
        H4 = H2*H2
    print('V:\t{}'.format(N))
    print('E:\t{}'.format(H4.count_nonzero() // 2))
    print('average degree:\t{}'.format(H4.count_nonzero() / N))
    print('eigenvalues^4:', H4.diagonal().sum())

    with simple_timer('H8'):
        H8 = H4*H4
    print('V:\t{}'.format(N))
    print('E:\t{}'.format(H8.count_nonzero() // 2))
    print('average degree:\t{}'.format(H8.count_nonzero() / N))
    print('eigenvalues^8:', H4.diagonal().sum())
