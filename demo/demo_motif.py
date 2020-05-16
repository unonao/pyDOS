import os
import sys
sys.path.append('../')
args = sys.argv
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as ssla
import matplotlib.pyplot as plt

from pyDOS import load_graph, normalize_matrix
from pyDOS import cal_for_chebhist, cal_for_cheb_density
from pyDOS import filter_jackson
from pyDOS import zero_filter
from pyDOS.moments import dos_by_cheb

if __name__ == '__main__':
    # command line args
    filepath = '../data/HepTh.mat' if len(args) <= 1 else args[1]
    method = 'cheb' if len(args) <= 2 else args[2]
    Nz = 20 if len(args) <= 3 else int(args[3])
    moment_num = 100 if len(args) <= 4 else int(args[4])
    bin_num = 51 if len(args) <= 5 else int(
        args[5])  # bin_num should be odd (to avoid splitting true λ=0)
    is_filter = True if len(args) <= 6 else bool(args[6])

    # load graph network
    H, true_eig_vals = load_graph(filepath)
    N = H.shape[0]

    print("method:\t{}".format(method))
    print('Nz:\t{}'.format(Nz))
    print('moments:\t{}'.format(moment_num))
    print('bins:\t{}'.format(bin_num))
    print('filter:\t{}'.format('Jackson' if is_filter else 'None'))
    print('N:\t{}'.format(N))
    print('M:\t{}'.format(H.count_nonzero()))

    if N > 1e3:
        print('Graph size might be too large for exact computation.')

    # normalize matrix
    H = normalize_matrix(H)

    # zero filter
    Q = zero_filter(H)

    # c[m] = tr(T_m(H)) (m = 0 to n-1)
    # d[m] = 1/N * tr(T_m(H)) (m = 0 to n-1)
    if method == 'cheb':
        c, cstd = dos_by_cheb(H, N, Nz, moment_num, Q)
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
    X = np.linspace(lmin, lmax, bin_num + 1)
    if true_eig_vals is not None:
        #lmin = max(min(true_eig_vals), -1)
        #lmax = min(max(true_eig_vals), 1)
        plt.hist(true_eig_vals, bins=X)

    Xmid = (X[0:-1] + X[1:]) / 2
    Y = cal_for_chebhist(df, X) * N
    # add zero motif
    zero_ind = np.argmax(X > 0) - 1
    Y[zero_ind] += Q.shape[1]
    plt.plot(Xmid, Y, 'r.', 60)

    plt.xlim(lmin, lmax)
    #plt.ylim(0, 500)
    plot_title = 'DOS (Nz:{}, M:{}, filter:{})'.format(
        Nz, moment_num, 'Jackson' if is_filter else 'None')
    plt.title(plot_title)
    plt.xlabel('λ')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
