import tensorflow as tf
from functools import partial

class Dataloader:
    def __init__(self, train_size, val_size, test_size, input_shape, batch_size):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.input_shape = input_shape
        self.batch_size = batch_size

    def load_data(self):
        train, val, test = self._load_images()
        train = train.batch(self.batch_size)
        val = val.batch(self.batch_size)
        test = test.batch(self.batch_size)
        return train, val, test

    def _load_images(self):
        def gen_real(batch_size):
            for _ in range(batch_size):
                s = tf.random.normal((self.input_shape, ),
                        dtype=tf.dtypes.float32)
                s = tf.math.l2_normalize(s)
                yield (s, s)

        x_train = tf.data.Dataset.from_generator(partial(gen_real,
            batch_size=self.train_size),
            (tf.float32, tf.float32))

        x_val = tf.data.Dataset.from_generator(partial(gen_real,
            batch_size=self.train_size),
            (tf.float32, tf.float32))

        x_test = tf.data.Dataset.from_generator(partial(gen_real,
            batch_size=self.train_size),
            (tf.float32, tf.float32))

        return x_train, x_val, x_test
