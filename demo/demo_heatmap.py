'''
If you want to use this script, change "par_dir" & "files".
'''

import os
import sys
sys.path.append('../')
args = sys.argv
import numpy as np
import numpy.linalg as LA
import pandas as pd
import scipy.sparse.linalg as ssla
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns

from pyDOS import load_graph, normalize_matrix
from pyDOS import cal_for_chebhist, cal_for_cheb_density
from pyDOS import filter_jackson
from pyDOS.moments import dos_by_cheb

if __name__ == '__main__':
    # command line args
    method = 'cheb' if len(args) <= 1 else str(args[1])
    Nz = 20 if len(args) <= 2 else int(args[2])
    moment_num = 500 if len(args) <= 3 else int(args[3])
    bin_num = 51 if len(args) <= 4 else int(
        args[4])  # bin_num should be odd (to avoid splitting true λ=0)
    is_filter = True if len(args) <= 5 else bool(args[5])

    par_dir = '../data/'
    files = [
        #sparse
        "graph-eigs-v1/Erdos02-cc.smat",
        "HepTh.mat",
        # social
        "graph-eigs-v1/marvel-chars-cc.smat",
        "snap/facebook_combined.txt",
        "konect/ca-AstroPh/out.ca-AstroPh",
        # hyper-link
        "konect/web-Stanford/out.web-Stanford",
        # road network
        "minnesota.mat",
        "snap/roadNet-CA.txt",
        "konect/dimacs9-NY/out.dimacs9-NY",
        # infrastructure
        "konect/openflights/out.openflights",
        "konect/opsahl-powergrid/out.opsahl-powergrid",
        # models
        "gen_model/Erdos-Renyi_random_graph_3000_30000.txt",
        "gen_model/scale-free_3000_5.txt",
        "gen_model/small_world_5000_5_0.100.txt",
        "gen_model/forest_fire_5000_0.45_0.30.txt",
        "gen_model/grid_graph_200_200.txt",
        "gen_model/grid_graph_deleted_200_200.txt"
    ]

    print("method:\t{}".format(method))
    print('Nz:\t{}'.format(Nz))
    print('moments:\t{}'.format(moment_num))
    print('bins:\t{}'.format(bin_num))
    print('filter:\t{}'.format('Jackson' if is_filter else 'None'))

    num = len(files)
    dos = np.empty((0, bin_num), float)
    label = []
    for i in range(num):
        filepath = par_dir + files[i]
        if os.path.isfile(filepath) is False:
            continue
        print(filepath)
        # load graph network
        (H, true_eig_vals) = load_graph(filepath)
        N = H.shape[0]
        # normalize matrix
        H = normalize_matrix(H)
        # check size & compute eigenvalues
        if N < 400:
            if true_eig_vals is None:
                true_eig_vals = LA.eigvalsh(H.toarray())

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
        X = np.linspace(lmin, lmax, bin_num + 1)
        if true_eig_vals is not None:
            eig_hist, _, _ = plt.hist(true_eig_vals, bins=X)
        Xmid = (X[0:-1] + X[1:]) / 2
        Ymid = cal_for_chebhist(df, X) * N
        plt.plot(Xmid, Ymid, 'r.', 60)
        # setting
        plt.xlim(lmin, lmax)
        plt.ylim(0)

        base_name = os.path.basename(filepath)

        plot_title = 'DOS: {} (Nz:{}, M:{}, bins:{} )'.format(
            base_name, Nz, moment_num, bin_num)
        plt.title(plot_title)
        plt.xlabel('λ')
        plt.ylabel('Count')
        plt.savefig('../plot/' + base_name + '_dos.png')
        plt.close()

        # store dos
        if true_eig_vals is not None:
            dos = np.append(dos, eig_hist.reshape(1, bin_num), axis=0)
        else:
            dos = np.append(dos, Ymid.reshape(1, bin_num), axis=0)
        label.append(base_name)

    num = len(dos)
    dist = np.ndarray((num, num))
    for i in range(num):
        for j in range(num):
            dist[i, j] = distance.cosine(dos[i], dos[j])
    df = pd.DataFrame(data=dist, index=label, columns=label)
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.2)
    sns.heatmap(df, square=True, vmax=1, vmin=0, annot=True)
    plt.title("cosine distances")
    plt.savefig('../heatmap.png')
    plt.show()