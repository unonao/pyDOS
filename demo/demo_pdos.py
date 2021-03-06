import os
import sys
sys.path.append('../')
args = sys.argv
import numpy as np
from scipy import linalg
import scipy.sparse.linalg as ssla
import matplotlib.pyplot as plt

from pyDOS import load_graph, normalize_matrix
from pyDOS import filter_jackson
from pyDOS.moments import pdos_by_cheb


def save_pdos(filepath, method, Nz, moment_num, bin_num, is_filter, is_show=False, save_dir='../plot/'):
    # load graph network
    (H, true_eig_vals) = load_graph(filepath)
    N = H.shape[0]

    print("file: " + filepath)
    print("method:\t{}".format(method))
    print('Nz:\t{}'.format(Nz))
    print('moments:\t{}'.format(moment_num))
    print('bins:\t{}'.format(bin_num))
    print('filter:\t{}'.format('Jackson' if is_filter else 'None'))
    print('V:\t{}'.format(N))
    print('E:\t{}'.format(H.count_nonzero() // 2))
    print('average degree:\t{}'.format(H.count_nonzero() / N))

    if N > 1e3:
        print('Graph size might be too large for exact computation.')

    # normalize matrix
    H = normalize_matrix(H)

    # c[m] = tr(T_m(H)) (m = 0 to n-1)
    # d[m] = 1/N * tr(T_m(H)) (m = 0 to n-1)
    if method == 'cheb':
        cl, cstd = pdos_by_cheb(H, N, Nz, moment_num)
        d = cl / N

    # filter
    if is_filter:
        df = filter_jackson(d)
    else:
        df = cf

    df = df * N

    # plot ldos
    # Run the recurrence to compute CDF
    X = np.linspace(-1 + 1e-8, 1 - 1e-8, bin_num + 1).reshape(1, bin_num + 1)
    tX = np.arccos(X)
    Y = df[0, :].reshape(N, 1) * (tX - np.pi) / 2
    for i in range(1, moment_num):
        Y = Y + df[i, :].reshape(N, 1) * np.sin(i * tX) / i

    Y = -2 / np.pi * Y

    # Difference the CDF to compute histogram
    Y = Y[:, 1:] - Y[:, 0:-1]

    U, _, _ = linalg.svd(Y, full_matrices=False)
    idx = np.argsort(U[:, 0])
    bot = np.min(Y)
    top = np.max(Y)

    extent = -1, 1, N, 0
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 20

    plt.imshow(Y[idx, :], cmap='jet', aspect='auto', extent=extent)
    plt.xlabel('λ')
    plt.ylabel('Node Index')

    base_name = os.path.basename(filepath).replace(".", "").replace(" ", "")
    """
    plot_title = 'PDOS: {} (Nz:{}, M:{}, bins:{})'.format(
        base_name, Nz, moment_num, bin_num)
    plt.title(plot_title)
    """
    plt.savefig(save_dir + base_name + '_pdos.png', bbox_inches='tight', pad_inches=0.)
    if is_show:
        plt.show()
    plt.close()

if __name__ == '__main__':
    # command line args
    filepath = '../data/HepTh.mat' if len(args) <= 1 else args[1]
    method = 'cheb' if len(args) <= 2 else str(args[2])
    Nz = 20 if len(args) <= 3 else int(args[3])
    moment_num = 1000 if len(args) <= 4 else int(args[4])
    bin_num = 51 if len(args) <= 5 else int(args[5])
    is_filter = True if len(args) <= 6 else bool(args[6])


    save_pdos(filepath, method, Nz, moment_num, bin_num, is_filter, is_show=True)
