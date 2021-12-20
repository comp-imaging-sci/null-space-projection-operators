import numpy as np
import scipy as sp
import scipy.sparse
import tensorflow as tf

from .StiefelSGD import *
from .Dataloader import *
from .Projection import *

class Model:
    def __init__(self, H, rank, projection, lr):
        self.H = H
        self.rank = rank
        self.projection = projection
        self.lr=lr
        self.model = self.create_model()

    def _lpo_loss(self, y_actual, y_pred):
        val = tf.sparse.sparse_dense_matmul(self.H, y_pred, adjoint_b=True)
        return (0.5 / val.shape[0]) * tf.norm(val)**2

    def create_model(self):
        input_data = tf.keras.layers.Input(self.H.shape[1])
        kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0)

        dense = self.projection(num_outputs=self.rank,
                kernel_initializer=kernel_initializer)

        x = tf.keras.layers.Flatten()(input_data)
        x = dense(x)

        f = StiefelSGD(learning_rate=self.lr, beta=0.9)

        model = tf.keras.models.Model(inputs=input_data, outputs=x)
        model.compile(optimizer=f,
                loss=self._lpo_loss,
                metrics=[])

        return model

    def train(self, train_dataset, val_dataset, num_epochs):
        self.model.fit(train_dataset,
                epochs=num_epochs,
                validation_data=val_dataset,
                validation_freq=1,
                callbacks=[self.ProjectWeightsCallback(rank=self.rank)])

    def get_w(self):
        return self.model.layers[2].get_weights()[0]

    class ProjectWeightsCallback(tf.keras.callbacks.Callback):
        def __init__(self, rank, **kwargs):
            super().__init__(**kwargs)
            self.rank = rank

        def on_epoch_end(self, epoch, logs=None):
            w = self.model.layers[2].get_weights()
            w = w[0]
            [w, _] = np.linalg.qr(w)
            self.model.set_weights([w])

def LPO(H, r, lr=1e-3, num_iter=10000, num_epochs=5, batch_size=8):
    """Computes the null space project operator for a large, sparse, imaging
    operator.

    This function solves either ``min \frac{1}{2} ||\mathbf{H} \mathbf{W}
    \mathbf{W}^\star f||^2 \, s.t. \mathbf{W}^star \mathbf{W} = I_{n - r}`` for null
    space directly or ``min \frac{1}{2} ||\mathbf{H} (I - \mathbf{W} \mathbf{W})
    f||^2 \, s.t. \mathbf{W}^\star \mathbf{W} = I_{r}``

    Parameters
    ----------
    H : SparseTensor
        Representation of an m,n imaging operator.
    r : int
        Numerical rank of H.
    lr : float, optional
       Learning rate used for the LPO method.
    num_iter : int, optional
        Iterations to run per epoch
    num_epochs : int, optional
        Number of epochs to run.
    batch_size : int, optional
        Batch size used per iteration.

    Returns
    -------
    w : {ndarray}
        The final null space projection operator.
    is_null : bool
        This returns ``true`` when P_{\text{null}} is represented as w w^\star and
        ``false`` when P_{\text{null}} is represented as (I - w w^\star).
    """

    m, n = H.shape
    is_null = r > n - r

    if is_null:
        arch = NullSpaceProjection
        k = n - r
    else:
        arch = InformedSubspaceProjector
        k = r
    dl = Dataloader(num_iter, 1, 1, n, batch_size)
    train_ds, val_ds, test_ds = dl.load_data()

    model = Model(H, k, arch, lr)

    model.train(train_ds, val_ds, num_epochs=num_epochs)
    w = model.get_w()

    return w
