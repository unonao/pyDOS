import numpy as np
import scipy.sparse as ss
import numpy.linalg as LA
np.random.seed(0)


def zero_filter(A):
    '''
    Construct the filter for node duplicate which leads to eigenvalue 0.
    It is the most common motif.

    Out: A matrix with eigenvectors of the target eigenvalue(λ=0) as columns.
    '''
    same_hash_list = vertex_hashing(A, 1)
    sqd = np.sqrt(A.sum(0))  # sqrt of degree
    n = A.shape[0]
    return filter_construct(same_hash_list, sqd, n)


def vertex_hashing(A, pattern=1):
    if pattern == 1:
        hashfun = lambda x: A * x
    elif pattern == 2:
        hashfun = lambda x: A * x + x
    else:
        raise Exception("No such hash pattern")

    n = A.shape[0]
    w = hashfun(np.random.randn(n, 1))
    w = w.flatten()

    # uniquetol of w
    TOL = 1e-12
    idc = np.argsort(w)
    w.sort()
    d = np.append(0, np.diff(w))
    k = np.count_nonzero(d > TOL)

    same_hash_list = [[] for i in range(k)]
    now = []
    dt = 0
    for i in range(n):
        if d[i] > TOL:
            if len(now) > 0:
                now.sort()
                same_hash_list[dt] = now
                dt += 1
            now = []
        now.append(idc[i])
    if len(now) > 1:
        same_hash_list[dt] = now

    return same_hash_list


def filter_construct(same_hash_list, sqd, n):
    '''
    Input:
        vinfo: Sets of nodes that form the filter.
        sqd:   Square root of degrees.
        n:     Number of vertices.
    Out:
        A matrix with eigenvectors of the target eigenvalue(λ=0) as columns.
    '''
    nf = len(same_hash_list)  # the same as k in vertex hasing
    vfl = []
    for i in range(len(same_hash_list)):
        vfl.append(len(same_hash_list[i]))

    nnzvf = np.sum(vfl)
    npair = 2 * (nnzvf - nf)
    ind = np.zeros((npair, 3))
    cnt1 = 0  # 対称的な頂点が2つ以上k個存在する時、k-1回加算
    cnt2 = 0
    for i in range(nf):
        for j in range(1, len(same_hash_list[i])):
            ind[cnt2, 0] = cnt1
            ind[cnt2, 1] = same_hash_list[i][0]
            ind[cnt2, 2] = sqd[0, same_hash_list[i][0]]
            ind[cnt2 + 1, 0] = cnt1
            ind[cnt2 + 1, 1] = same_hash_list[i][j]
            ind[cnt2 + 1, 2] = -sqd[0, same_hash_list[i][j]]
            cnt1 += 1
            cnt2 += 2

    # sparse kernel
    Q = ss.csr_matrix((ind[:, 2], (ind[:, 1], ind[:, 0])), (n, nnzvf - nf))
    Q = Q.toarray()
    ind2 = np.append(0, np.cumsum(vfl - np.ones(nf, dtype=int), dtype=int))
    for j in range(1, nf):
        if ind2[j - 1] < ind2[j]:
            Q[:, ind2[j - 1]:(ind2[j] - 1)], _ = LA.qr(
                Q[:, ind2[j - 1]:(ind2[j] - 1)])
    return Q
