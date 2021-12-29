import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib.pyplot as plt
from null_space_projection_operators import LPO

H = sp.sparse.load_npz('RT_64x64_30_views.npz')
f = np.load('USCT_impedance_64x64.npy')

# The rank of this operator is 2347
w = LPO(H, 2347, lr=1e-3, num_iter=10000, num_epochs=5, batch_size=8)
f_null = f - w @ w.T @ f

plt.figure()
plt.imshow(f_null.reshape([64, 64]))
plt.xlabel([])
plt.ylabel([])
plt.title('$\mathbf{f}_{\text{null}}$ LPO')
