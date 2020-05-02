import os
import sys
sys.path.append('../')
args = sys.argv
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as ssla
import matplotlib.pyplot as plt

from pyDOS import load_graph, shift_and_rescale_matrix, normalize_matrix
from pyDOS import cul_for_chebhist, cul_for_cheb_density
from pyDOS import filter_jackson
from pyDOS.moments import dos_by_cheb

if __name__ == '__main__':
    # command line args
    filepath = '../data/graph-eigs-v1/Erdos02-cc.smat' if len(
        args) <= 1 else args[1]
    method = 'cheb' if len(args) <= 2 else str(args[2])
    Nz = 20 if len(args) <= 3 else int(args[3])
    moment_num = 500 if len(args) <= 4 else int(args[4])
    bin_num = 50 if len(args) <= 5 else int(args[5])
    is_filter = True if len(args) <= 6 else bool(args[6])
    print(method)
    # load graph network
    (H, true_eig_vals) = load_graph(filepath)
    N = H.shape[0]
    print('n: {}'.format(N))
    if N > 1e3:
        print('Graph size might be too large for exact computation.')

    # normalize matrix
    H = normalize_matrix(H)

    # c[m] = tr(T_m(H)) (m = 0 to n-1)
    # d[m] = 1/N * tr(T_m(H)) (m = 0 to n-1)
    if method == 'cheb':
        c, cstd = dos_by_cheb(H, N, Nz, moment_num)
        d = c / N

    # filter
    if is_filter:
        df = filter_jackson(d)
    else:
        df = cf

    # plot

    lmin = -1
    lmax = 1
    plt.figure()
    if true_eig_vals is not None:
        lmin = max(min(true_eig_vals), -1)
        lmax = min(max(true_eig_vals), 1)
        plt.hist(true_eig_vals, bins=bin_num)

    # plot with prob function (Gibbs oscillations)
    """
    bin_size = N // bin_num
    X = np.linspace(lmin + 1e-8, lmax - 1e-8, 100)
    Y = cul_for_cheb_density(df, X) * bin_size
    plt.plot(X, Y)

    # the same graph in the papar
    """
    X = np.linspace(lmin, lmax, bin_num + 1)
    Xmid = (X[0:-1] + X[1:]) / 2
    plt.plot(Xmid, cul_for_chebhist(df, X) * N, 'r.', 60)

    # setting
    plt.xlim(lmin, lmax)
    #plt.ylim(0, 500)
    plot_title = 'DOS (Nz:{}, M:{}, filter:{})'.format(
        Nz, moment_num, 'Jackson' if is_filter else 'None')
    plt.title(plot_title)
    plt.xlabel('λ')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
