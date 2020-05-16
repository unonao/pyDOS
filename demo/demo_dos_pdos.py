import os
import sys
sys.path.append('../')
args = sys.argv
import numpy as np
from scipy import linalg
import scipy.sparse as ss
import scipy.sparse.linalg as ssla
import matplotlib.pyplot as plt

from pyDOS import load_graph, normalize_matrix
from pyDOS import cal_for_chebhist, cal_for_cheb_density
from pyDOS import filter_jackson
from pyDOS.moments import dos_by_cheb
from pyDOS.moments import pdos_by_cheb

if __name__ == '__main__':
    # command line args
    filepath = '../data/facebook_combined.txt' if len(args) <= 1 else args[1]
    method = 'cheb' if len(args) <= 2 else str(args[2])
    Nz = 20 if len(args) <= 3 else int(args[3])
    moment_num = 500 if len(args) <= 4 else int(args[4])
    bin_num = 51 if len(args) <= 5 else int(
        args[5])  # bin_num should be odd (to avoid splitting true λ=0)
    is_filter = True if len(args) <= 6 else bool(args[6])

    # load graph network
    (H, true_eig_vals) = load_graph(filepath)

    N = H.shape[0]

    print("method:\t{}".format(method))
    print('Nz:\t{}'.format(Nz))
    print('moments:\t{}'.format(moment_num))
    print('bins:\t{}'.format(bin_num))
    print('filter:\t{}'.format('Jackson' if is_filter else 'None'))
    print('N:\t{}'.format(N))
    print('nonzero:\t{}'.format(H.count_nonzero()))

    # normalize matrix
    H = normalize_matrix(H)
    # check size & compute eigenvalues
    if N > 400:
        print('Graph size might be too large for exact computation.')
    else:
        if true_eig_vals is None:
            true_eig_vals = LA.eigvalsh(H.toarray())

    # for plot
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    #
    # DOS
    #
    '''
    c[m] = tr(T_m(H)) (m = 0 to n-1)
    d[m] = 1/N * tr(T_m(H)) (m = 0 to n-1)
    '''
    if method == 'cheb':
        c, cstd = dos_by_cheb(H, N, Nz, moment_num)
        d = c / N
    # filter
    if is_filter:
        df = filter_jackson(d)
    else:
        df = cf

    lmin = -1
    lmax = 1
    X = np.linspace(lmin, lmax, bin_num + 1)
    if true_eig_vals is not None:
        #lmin = max(min(true_eig_vals), -1)
        #lmax = min(max(true_eig_vals), 1)
        ax1.hist(true_eig_vals, bins=X)
    Xmid = (X[0:-1] + X[1:]) / 2
    Ymid = cal_for_chebhist(df, X) * N
    ax1.plot(Xmid, Ymid, 'r.', 60)
    #### setting
    ax1.set_xlim(lmin, lmax)
    ax1.set_ylim(0)
    ax1.set_title('DOS')
    ax1.set_xlabel('λ')
    ax1.set_ylabel('Count')

    #
    # PDOS
    #
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
    X = np.linspace(-1 + 1e-8, 1 - 1e-8, bin_num).reshape(1, bin_num)
    tX = np.arccos(X)
    Y = df[0, :].reshape(N, 1) * (tX - np.pi) / 2
    for i in range(1, moment_num):
        Y = Y + df[i, :].reshape(N, 1) * np.sin(i * tX) / i
    Y = -2 / np.pi * Y

    # Difference the CDF to compute histogram
    Y = Y[:, 1:] - Y[:, 0:-1]
    # sort by SVD
    U, _, _ = linalg.svd(Y, full_matrices=False)
    idx = np.argsort(U[:, 0])
    bot = np.min(Y)
    top = np.max(Y)

    extent = -1, 1, N, 0
    ax2.imshow(Y[idx, :], cmap='jet', aspect='auto', extent=extent)
    ax2.set_title("PDOS")
    ax2.set_xlabel('λ')
    ax2.set_ylabel('Node Index')

    #
    # fig show
    #
    base_name = os.path.basename(filepath)

    plot_title = '{} (Nz:{}, M:{}, bins:{})'.format(base_name, Nz, moment_num,
                                                    bin_num)
    fig.suptitle(plot_title)
    plt.savefig('../plot/' + base_name + '_dos_pdos.png')
    plt.show()
