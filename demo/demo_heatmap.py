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
    kind = 0 if len(args) <= 1 else int(
        args[1])  # 0: unsort, 1: sort by degree
    bin_num = 51 if len(args) <= 2 else int(
        args[2])  # bin_num should be odd (to avoid splitting true λ=0)

    method = 'cheb' if len(args) <= 3 else args[3]
    Nz = 30 if len(args) <= 4 else int(args[4])
    moment_num = 1000 if len(args) <= 5 else int(args[5])
    is_filter = True if len(args) <= 6 else bool(args[6])

    par_dir = '../data/'
    db_dir = '../moments_db/'
    files = [
        # human social networks
        "konect/moreno_health/out.moreno_health_health",
        # social networks
        "graph-eigs-v1/marvel-chars-cc.smat",
        "snap/facebook_combined.txt",
        "snap/soc-Epinions1.txt",
        # collaboration networks
        "konect/ca-AstroPh/out.ca-AstroPh",
        # communication networks
        "snap/Email-Enron.txt",
        "snap/Email-EuAll.txt",  # Autonomous systems
        "graph-eigs-v1/Erdos02-cc.smat",  #(Erdos02-cc - Pajek's Erdos sample file, largest connected component.as-caida20060911 - SNAP)
        # hyper-link(web graph)
        "konect/web-NotreDame/out.web-NotreDame",
        "konect/web-Stanford/out.web-Stanford",
        # infrastructure
        "konect/openflights/out.openflights",
        "konect/opsahl-usairport/out.opsahl-usairport",
        "konect/opsahl-powergrid/out.opsahl-powergrid",
        # road network
        "minnesota.mat",
        "snap/roadNet-CA.txt",
        "konect/dimacs9-NY/out.dimacs9-NY",
        # models
        "gen_model/Erdos-Renyi_random_graph_3000_3000.txt",
        "gen_model/Erdos-Renyi_random_graph_3000_30000.txt",
        "gen_model/scale-free_3000_1.txt",
        "gen_model/scale-free_3000_5.txt",
        "gen_model/small_world_5000_5_0.010.txt",
        "gen_model/small_world_5000_5_0.100.txt",
        "gen_model/forest_fire_5000_0.40_0.30.txt",
        "gen_model/forest_fire_5000_0.45_0.30.txt",
        "gen_model/copying_model_5000_0.50.txt",
        "gen_model/grid_graph_200_200.txt",
        "gen_model/grid_graph_deleted_200_200.txt"
    ]

    print("method:\t{}".format(method))
    print('Nz:\t{}'.format(Nz))
    print('moments:\t{}'.format(moment_num))
    print('bins:\t{}'.format(bin_num))
    print('filter:\t{}'.format('Jackson' if is_filter else 'None'))

    num = len(files)
    for i in range(num):
        filepath = par_dir + files[i]
        if os.path.isfile(filepath) is False:
            print("WARN:", files[i], " not found.")

    doss = np.empty((0, bin_num), float)
    labels = np.empty(0)
    sort_vals = np.empty(0)
    for i in range(num):
        base_name = os.path.basename(files[i])
        dbpath = db_dir + base_name + '_{}.npy'.format(moment_num)
        filepath = par_dir + files[i]

        print("file: " + filepath, end="\t")
        df = None
        if os.path.isfile(filepath) is False:
            continue

        (H, true_eig_vals) = load_graph(filepath)
        N = H.shape[0]
        if (N < 300) and (true_eig_vals is None):
            true_eig_vals = LA.eigvalsh(H.toarray())

        if kind == 1:  # sort by degree
            sort_vals = np.append(sort_vals, H.sum() / N)

        dos = None
        if true_eig_vals is not None:
            print("use true eigenvalues")
            eig_hist, _, _ = plt.hist(true_eig_vals, bins=X)
            dos = eig_hist.reshape(1, bin_num)
        else:
            if os.path.isfile(
                    dbpath):  # If moments have already been calculated
                print("already calculated")
                df = np.load(dbpath)
            else:
                print("calculating...")
                # load graph network
                # normalize matrix
                H = normalize_matrix(H)

                # calculate moment
                if method == 'cheb':
                    c, cstd = dos_by_cheb(H, N, Nz, moment_num)
                    d = c / N
                # filter
                if is_filter:
                    df = filter_jackson(d)
                else:
                    df = cf
                np.save(dbpath, df)

            # plot
            lmin = -1
            lmax = 1
            plt.figure()
            X = np.linspace(lmin, lmax, bin_num + 1)
            Xmid = (X[0:-1] + X[1:]) / 2
            Ymid = cal_for_chebhist(df, X) * N
            plt.plot(Xmid, Ymid, 'r.', 60)
            # setting
            plt.xlim(lmin, lmax)
            plt.ylim(0)
            plot_title = 'DOS: {} (Nz:{}, M:{}, bins:{} )'.format(
                base_name, Nz, moment_num, bin_num)
            plt.title(plot_title)
            plt.xlabel('λ')
            plt.ylabel('Count')
            plt.savefig('../plot/' + base_name + '_dos.png')
            plt.close()

            dos = Ymid.reshape(1, bin_num)

        doss = np.append(doss, dos, axis=0)
        labels = np.append(labels, base_name)

    num = len(doss)
    dist = np.ndarray((num, num))
    if kind > 0:
        ids = np.argsort(sort_vals)
        doss = doss[ids]
        labels = labels[ids]
        print(sort_vals[ids])
    for i in range(num):
        for j in range(num):
            dist[i, j] = distance.cosine(doss[i], doss[j])
    df = pd.DataFrame(data=dist, index=labels, columns=labels)
    plt.close()
    plt.figure(figsize=(num + 5, num + 5))
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.2)
    sns.heatmap(df, square=True, vmax=1, vmin=0, annot=True)
    plt.title("cosine distances")
    plt.savefig('../heatmap_{}_{}.png'.format(kind, bin_num))
    plt.show()
