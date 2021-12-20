import numpy as np
import scipy as sp
import scipy.sparse.linalg

def WB(H, f, a=None, max_iter=1000000, atol=1e-6):
    """Computes the null space component for a single vector for a large,
    sparse, imaging operator using the Wilson-Barrett method.

    Parameters
    ----------
    H : {sparse matrix, ndarray, LinearOperator}
        Representation of m,n imaging operator.
    f : array_like
        Input vector.
    a : float, optional
        Step size used.
    max_iter : int, optional
        Maximum number of iterations to run. This function will terminate if the
        maxmimum number of iterations is exceeded.
    atol : float, optional
        Absolute tolerance used for early stopping. This function will terminate
        if atol < \frac{1}{2 * n} ||H f_null ||_2^2.

    Returns
    -------
    f_null : array_like
        The null space projection to the input vector f.
    """

    H = sp.sparse.linalg.aslinearoperator(H)

    f_null = f.copy()
    r = H.matvec(f_null)

    # Estimate a value for a if not given. Any value less than 2 / \sigma_0^2, where
    # \sigma_0 is a largest singular value, should converge given enough
    # iterations. In numerical simulations, 1.9 / \sigma_0^2 seems to yield good
    # results, however your results may vary.
    if not a:
        s = sp.sparse.linalg.svds(H, k=1, return_singular_vectors=False)
        a = 1.9 / s[0]**2

    for _ in range(max_iter):
        res = 0.5 * (1 / f_null.shape[0]) * np.dot(r.T, r)

        if res < atol:
            return f_null
        g = -H.rmatvec(r)
        f_null += a * g
        r = H.matvec(f_null)

    return f_null
