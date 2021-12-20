import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

class StiefelSGD(optimizer.Optimizer):
    def __init__(self, learning_rate=1e-1, beta=0.0, eps=1e-8, q=0.5, s=2, use_locking=False, name='StiefelSGD'):
        super(StiefelSGD, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta = beta
        self._eps = eps
        self._q = q
        self._s = s
        self._step = 0

        # Tensor version, created in prepare
        self._lr_t = None
        self._beta_t = None
        self._eps_t = None
        # Don't create a tensor for s since it should be an int

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        self._lr_t = ops.convert_to_tensor(lr, name='learning_rate')
        self._beta_t = ops.convert_to_tensor(self._beta, name='beta')
        self._eps_t = ops.convert_to_tensor(self._eps, name='epsilon')

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "m", self._name)

    def _resource_apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        eps_t = math_ops.cast(self._eps_t, var.dtype.base_dtype)
        self._step += 1

        m = self.get_slot(var, "m")
        m_t = beta_t * m - grad

        def W_matmat(M, X, Y):
            WAA = tf.matmul(M, tf.matmul(X, Y, transpose_a=True))
            WAB = tf.matmul(X, WAA, transpose_a=True)
            WAB = tf.matmul(X, WAB)

            WBA = tf.matmul(X, tf.matmul(M, Y, transpose_a=True))
            WBB = tf.matmul(X, tf.matmul(X, Y, transpose_a=True))
            WBB = tf.matmul(X, tf.matmul(M, WBB, transpose_a=True))
            return WAA - 0.5 * WAB - WBA + 0.5 * WBB

        m_t = W_matmat(m_t, var, var)

        m_t = state_ops.assign(m, m_t)

        y = var + lr_t * m_t
        alpha = 0.5 * lr_t
        for _ in range(self._s):
            y = var + alpha * W_matmat(m_t, var, (var + y))

        var_update = state_ops.assign(var, y)

        return control_flow_ops.group(*[var_update, m_t])
