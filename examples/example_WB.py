import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib.pyplot as plt
from null_space_projection_operators import WB

H = sp.sparse.load_npz('RT_64x64_30_views.npz')
f = np.load('USCT_impedance_64x64.npy')

f_null = WB(H, f, atol=1e-12)

plt.figure()
plt.imshow(f_null.reshape([64, 64]))
plt.xlabel([])
plt.ylabel([])
plt.title('$\mathbf{f}_{\text{null}}$ WB')
