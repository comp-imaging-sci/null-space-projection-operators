import tensorflow as tf

class NullSpaceProjection(tf.keras.layers.Layer):
    def __init__(self, num_outputs, kernel_initializer, **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.kernel = self.add_weight(
                shape=[input_shape[-1], self.num_outputs],
                initializer=self.kernel_initializer,
                trainable=True)

    def call(self, f):
        return tf.matmul(tf.matmul(f, self.kernel), self.kernel, transpose_b=True)

class InformedSubspaceProjector(tf.keras.layers.Layer):
    def __init__(self, num_outputs, kernel_initializer):
        super(InformedSubspaceProjector, self).__init__()
        self.num_outputs = num_outputs
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.kernel = self.add_weight(
                shape=(input_shape[-1], self.num_outputs),
                initializer=self.kernel_initializer,
                trainable=True)

    def call(self, f):
        return f - tf.matmul(tf.matmul(f, self.kernel), self.kernel, transpose_b=True)
