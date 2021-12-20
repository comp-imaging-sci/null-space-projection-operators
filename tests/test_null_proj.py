import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.sparse
import pytest

from null_space_projection_operators import rSVD, WB, LPO

class Test:
    def setup_method(self):
        # Set seed for reproducibility.
        tf.random.set_seed(0)
        np.random.seed(0)

        self.N = 100
        self.M = 40
        self.rank = 10

        # Generate a random sparse matrix with fixed rank
        row = np.arange(self.rank)
        col = np.arange(self.rank)
        data = np.random.uniform(1e-3, 100, size=(self.rank, )).astype(np.float32)
        self.H_tf = tf.SparseTensor(indices=list(zip(row, col)),
            values=data,
            dense_shape=(self.N, self.M))
        self.H_sp = sp.sparse.coo_matrix((data, (row, col)), shape=(self.N,
            self.M))

        # Generate a testing vector
        self.f = np.random.randn(self.M).astype(np.float32)
        self.f /= np.linalg.norm(self.f)

    def test_rSVD(self):
        u, s, vh = rSVD(self.H_sp, self.rank)
        f_null_rSVD = self.f - vh.T @ vh @ self.f
        res = self.H_sp @ f_null_rSVD
        assert res.T @ res < 1e-6

    def test_WB(self):
        f_null_WB = WB(self.H_sp, self.f, atol=1e-12)
        res = self.H_sp @ f_null_WB
        assert res.T @ res < 1e-6

    def test_LPO(self):
        w = LPO(self.H_tf, self.rank, lr=1e-3, num_iter=10000, num_epochs=2,
                batch_size=8)
        f_null_LPO = self.f - w @ w.T @ self.f
        res = self.H_sp @ f_null_LPO
        assert res.T @ res < 1e-6
