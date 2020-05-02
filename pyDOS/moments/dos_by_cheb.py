import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssla


def dos_by_cheb(H, N, Nz, M, Q=None):
    """
    Compute a column vector of Chebyshev moments of the form c(m) = tr(T_m(H))
	for k = 0 to N-1. This routine does no scaling; the spectrum of A should
	already lie in [-1,1].
    The traces are computed via a stochastic estimator with Nz probe. Probe vectors are np.sign(np.random.randn(N,Nz))

    Args:
        H: normalized matrix
        N: H is N*N matrix
        Nz: probe number
        M: max momennt number
        Q: motif filter
    """
    if isinstance(H, np.ndarray):
        H = ss.csr_matrix(H)
    Hfun = lambda z: H * z
    Z = np.sign(np.random.randn(N, Nz))
    if Q is not None:
        Z = Z - np.dot(Q, np.dot(Q.T, Z))

    cs = np.zeros((M, Nz))

    # Run three-term recurrence to compute moments
    T = [Z, Hfun(Z)]
    cs[0] = np.sum(Z * T[0], 0)
    cs[1] = np.sum(Z * T[1], 0)
    for i in range(2, M):
        Tm = 2 * Hfun(T[1]) - T[0]
        T[0] = T[1]
        T[1] = Tm
        cs[i] = sum(Z * T[1], 0)
    c = np.mean(cs, 1)
    cstd = np.std(cs, 1, ddof=1) / np.sqrt(Nz)
    c = c.reshape([M, -1])
    cstd = cstd.reshape([M, -1])
    return c, cstd


def ldos_by_cheb(H, N, Nz, M):
    """
    Compute a column vector (or vectors) of Chebyshev moments of the form c(k,j) = [T_k(A)]_jj for k = 0 to N-1. This routine does no scaling; the spectrum of A should
	already lie in [-1,1].

    Args:
        H: normalized matrix
        N: H is N*N matrix
        Nz: probe number
        M: max momennt number
    """
    if isinstance(H, np.ndarray):
        H = ss.csr_matrix(H)
    Hfun = lambda z: H * z
    Z = np.sign(np.random.randn(N, Nz))

    c = np.zeros((M, N))
    cstd = np.zeros((M, N))

    # Run three-term recurrence to compute moments
    T = [Z, Hfun(Z)]
    X = Z * T[0]
    c[0, :] = np.mean(X, 1).T
    cstd[0, :] = np.std(X, 1).T

    X = Z * T[1]
    c[1, :] = np.mean(X, 1).T
    cstd[1, :] = np.std(X, 1).T

    for i in range(2, M):
        Tm = 2 * Hfun(T[1]) - T[0]
        T[0] = T[1]
        T[1] = Tm
        X = Z * T[1]
        c[i, :] = np.mean(X, 1).T
        cstd[i, :] = np.std(X, 1).T

    csrd = cstd / np.sqrt(Nz)
    return c, cstd
