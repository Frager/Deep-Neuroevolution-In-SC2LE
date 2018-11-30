import tensorflow as tf


class BaseModel:
    def __init__(self, scope):
        self.weights_in_out = []
        self.weights = []
        self.biases = []
        self.scope = scope

    def create_weight(self, shape, layer_in, layer_out):
        w = tf.get_variable('w', dtype=tf.float32, shape=shape, trainable=False)
        self.weights.append(w)
        self.weights_in_out.append((layer_in, layer_out))
        return w

    def create_bias(self, shape):
        b = tf.get_variable('b', dtype=tf.float32, shape=shape, trainable=False)
        self.biases.append(b)
        return b

    def conv(self, x, name, kernel_size, num_outputs, stride=1, padding="SAME", bias=True):
        assert len(x.get_shape()) == 4  # Batch x Height x Width x Channel
        with tf.variable_scope(name):
            w = self.create_weight(shape=(1, kernel_size, kernel_size, int(x.get_shape()[-1].value), num_outputs),
                                   layer_in=x.get_shape()[-1].value, layer_out=num_outputs)
            w = tf.reshape(w, [-1, kernel_size * kernel_size * int(x.get_shape()[-1].value), num_outputs])

            patches = tf.extract_image_patches(x, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1],
                                               rates=[1, 1, 1, 1], padding=padding)
            final_shape = (
                tf.shape(x)[0], patches.get_shape()[1].value, patches.get_shape()[2].value, num_outputs)
            patches = tf.reshape(patches, [tf.shape(x)[0],
                                           -1,
                                           kernel_size * kernel_size * x.get_shape()[-1].value])

            ret = tf.matmul(patches, w)
            ret = tf.reshape(ret, final_shape)

            if bias:
                b = self.create_bias(shape=(1, 1, 1, num_outputs))
                return ret + b
            else:
                return ret

    def dense(self, x, name, size, bias=True):
        with tf.variable_scope(name):
            w = self.create_weight(shape=(x.get_shape()[-1].value, size), layer_in=x.get_shape()[-1].value, layer_out=size)
            ret = tf.matmul(x, w)

            if bias:
                b = self.create_bias((1, size,))
                return ret + b
            else:
                return ret

    # def create_set_function(self):
    #     for weight in self.weights:
