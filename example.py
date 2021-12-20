import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.sparse

from learned_projection_operator import *
from randomized_singular_value_decomposition import *
from wilson_barrett import *

# Set seed for reproducibility.
tf.random.set_seed(0)
np.random.seed(0)

N = 100
M = 40
rank = 10

# Generate a random sparse matrix with fixed rank
row = np.arange(rank)
col = np.arange(rank)
data = np.random.uniform(1e-3, 100, size=(rank, )).astype(np.float32)
H_tf = tf.SparseTensor(indices=list(zip(row, col)),
        values=data,
        dense_shape=(N, M))
H_sp = sp.sparse.coo_matrix((data, (row, col)), shape=(N, M))

# Generate a testing vector
f = np.random.randn(M).astype(np.float32)
f /= np.linalg.norm(f)

u, s, vh = rSVD(H_sp, rank)
f_null_rSVD = f - vh.T @ vh @ f
assert np.linalg.norm(H_sp @ f_null_rSVD) < 1e-4

f_null_WB = WB(H_sp, f, atol=1e-12)
assert np.linalg.norm(H_sp @ f_null_WB) < 1e-4

w = LPO(H_tf, rank)
f_null_LPO = f - w @ w.T @ f
assert np.linalg.norm(H_sp @ f_null_LPO) < 1e-4
