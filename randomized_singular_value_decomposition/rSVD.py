import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse.linalg

def rSVD(H, rank, n_oversamples=15, n_iter=7):
    """Computes the singular value decomposition of a matrix H using
    a randomized algorithm.

    Parameters
    ----------
    H : {sparse matrix, ndarray, LinearOperator}
        m-by-n matrix
    rank : int
        Number of singular components to compute
    n_oversamples : int, optional
        In order to compute an accurate estimate oversampling may be needed.
    n_iter : int, optional
        Number of power iterations to perform.

    Returns
    -------
    U : (M, rank) ndarray
        Left singular vectors
    s : (rank,) ndarray
        Singular values
    V : (rank, N) ndarray
        Right singular vectors
    """
    H = sp.sparse.linalg.aslinearoperator(H)
    # Stage A: Compute ||A - Q Q^\dagger|| \leq \epsilon

    k = rank + n_oversamples
    # Reconstructs a random matrix for Q. Note that the choice of random number
    # generator here doesn't really matter.
    Qx = np.random.randn(H.shape[1], k).astype(H.dtype)
    for i in range(n_iter):
        Qy, _ = sp.linalg.qr(H.matmat(Qx), mode='economic')
        Qx, _ = sp.linalg.qr(H.rmatmat(Qy), mode='economic')

    Qy, _ = sp.linalg.qr(H.matmat(Qx), mode='economic')

    # Stage B:
    # Since you cannot do a Dense @ LinearOperator, we instead
    # do (LinearOperator^\dagger @ Dense^\dagger)^\dagger

    # Perform Q.H @ H = (H.H @ Q).H
    B = H.H @ Qy
    B = B.T.conj() # Note .H property isn't defined for ndarrays
    Uhat, s, V = sp.linalg.svd(B, full_matrices=False)

    del B
    U = Qy @ Uhat

    return U[:, :rank], s[:rank], V[:rank, :]
