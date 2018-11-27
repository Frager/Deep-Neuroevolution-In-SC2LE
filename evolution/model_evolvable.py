import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from common.enums import ModelDataFormat as DataFormat


class ModelEvolvable:

    def __init__(self, scope):
        # seeds : (sigma, seed)
        self.start_seed = None
        self.evolve_seeds = []
        self.scope = scope

        self.add_tensors = {}
        self.variables_collection = []
        self.weights_collection = []
        self.data_format = None

        self.mutate_input = []
        self.assign_op = None
        self.add_op = None

    def spacial_block(self, spacial_dims, channel_dims):
        # TODO: do I actually need batch dimension?
        block_input = tf.placeholder(tf.float32, [None, len(channel_dims), spacial_dims, spacial_dims])
        if self.data_format == DataFormat.NHWC:
            channel_axis = 3
            block = tf.transpose(block_input, [0, 2, 3, 1])  # NHWC -> NCHW
            block = tf.split(block, len(channel_dims), axis=channel_axis)
        else:
            channel_axis = 1
            block = tf.split(block_input, len(channel_dims), axis=channel_axis)
        for i, d in enumerate(channel_dims):
            if d > 1:
                block[i] = tf.one_hot(tf.to_int32(tf.squeeze(block[i], axis=channel_axis)), d, axis=channel_axis)
                # TODO: check if convolution of one_hot encoded spacial input is correct
                block[i] = layers.conv2d(block[i], num_outputs=max(1, round(np.log2(d))), kernel_size=1
                                         , data_format=self.data_format.value, variables_collections=[self.scope])
            else:
                block[i] = tf.log(block[i] + 1.0)
        block = tf.concat(block, axis=channel_axis)

        # TODO: should I set reuse=True (for all variables in this scope)
        conv1 = layers.conv2d(block, num_outputs=16, kernel_size=5, data_format=self.data_format.value,
                              variables_collections=[self.scope])
        conv2 = layers.conv2d(conv1, num_outputs=32, kernel_size=3, data_format=self.data_format.value,
                              variables_collections=[self.scope])
        return conv2, block_input

    def non_spacial_block(self, dim, spacial_size):
        block_input = tf.placeholder(tf.float32, [None, dim])
        broadcast_block = self.broadcast_flat_feature(block_input, spacial_size, self.data_format)
        return broadcast_block, block_input

    def broadcast_flat_feature(self, block_input, spacial_size, data_format):
        if data_format == DataFormat.NCHW:
            block = tf.tile(tf.expand_dims(tf.expand_dims(block_input, 2), 3), [1, 1, spacial_size, spacial_size])
        else:
            block = tf.tile(tf.expand_dims(tf.expand_dims(block_input, 1), 2), [1, spacial_size, spacial_size, 1])
        return block

    def fully_conv(self, model_config):
        spacial_size = model_config.size
        self.data_format = model_config.model_format
        feature_inputs = model_config.feature_inputs
        arg_outputs = model_config.arg_outputs
        num_functions = model_config.num_functions
        if not isinstance(self.data_format, DataFormat):
            raise ValueError('variable data_format is not of type ModelDataFormat')
        block_inputs = {}
        blocks = []
        for feature_input in feature_inputs:
            with tf.variable_scope(feature_input.get_feature_names_as_scope()):
                if feature_input.is_spacial:
                    block, spacial_input = self.spacial_block(feature_input.get_spacial_dimensions(),
                                                              feature_input.get_channel_dimensions())
                    block_inputs[feature_input.input_name] = spacial_input
                    blocks.append(block)
                else:
                    # TODO: what about features of variable dimensions? like multi_select, alerts, build_queue ...
                    block, non_spacial_input = self.non_spacial_block(np.sum(feature_input.get_channel_dimensions()),
                                                                      spacial_size)
                    block_inputs[feature_input.input_name] = non_spacial_input
                    blocks.append(block)
        if self.data_format == DataFormat.NCHW:
            state = tf.concat(blocks, axis=1)
        else:
            state = tf.concat(blocks, axis=3)

        # from https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/networks/fully_conv.py#L131-L137
        # state to non_spacial action policy and value
        flat_state = layers.flatten(state)
        fully_connected = layers.fully_connected(flat_state, num_outputs=256, activation_fn=tf.nn.relu,
                                                 variables_collections=[self.scope])

        with tf.variable_scope("value"):
            value = layers.fully_connected(fully_connected, num_outputs=1, activation_fn=None,
                                           variables_collections=[self.scope])
            # tf.reshape(t, -1) flattens t
            value = tf.reshape(value, [-1])

        with tf.variable_scope("fn_out"):
            # fully_connected to action id
            fn_out, available_actions_input = self.function_id_output(fully_connected, num_functions)

        with tf.variable_scope("args_out"):
            # state to non-spacial/spatial action arguments (for each action type)
            args_out = {}
            for arg_output in arg_outputs:
                if arg_output.is_spacial:
                    arg_out = self.spatial_output(state)
                else:
                    arg_out = self.non_spatial_output(fully_connected, arg_output.arg_size)
                args_out[arg_output.arg_type] = arg_out

        policy = (fn_out, args_out)
        self.variables_collection = tf.get_collection(self.scope)

        # TODO: this probably belongs in another class
        assign_ops = []
        add_ops = []
        for variable in self.variables_collection:
            if 'weights' in variable.name:
                self.weights_collection.append(variable)
                placeholder = tf.placeholder(dtype=tf.float32, shape=variable.shape)
                self.mutate_input.append(placeholder)
                assign_ops.append(variable.assign(placeholder))
                add_ops.append(variable.assign_add(placeholder))
                tf.add_to_collection('mutate_input', placeholder)
        self.assign_op = tf.group(*assign_ops)
        self.add_op = tf.group(*add_ops)

        return block_inputs, policy, value, self.mutate_input, available_actions_input

    def function_id_output(self, x, channels):
        logits = layers.fully_connected(x, num_outputs=channels, activation_fn=None, variables_collections=[self.scope])
        available_actions_input = tf.placeholder(dtype=tf.float32, shape=logits.shape)
        masked_logits = tf.multiply(logits, available_actions_input)
        return tf.math.argmax(masked_logits, axis=1), available_actions_input

    # from https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/networks/fully_conv.py#L76-L78
    def non_spatial_output(self, x, channels):
        logits = layers.fully_connected(x, num_outputs=channels, activation_fn=None, variables_collections=[self.scope])
        return tf.math.argmax(logits, axis=1)

    # from https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/networks/fully_conv.py#L80-L84
    def spatial_output(self, x):
        logits = layers.conv2d(x, num_outputs=1, kernel_size=1, stride=1, activation_fn=None,
                               data_format=self.data_format.value, variables_collections=[self.scope])
        if self.data_format == DataFormat.NHWC:
            # return output back to NCHW format
            logits = layers.flatten(tf.transpose(logits, [0, 3, 1, 2]))
        else:
            logits = layers.flatten(logits)
        return tf.math.argmax(logits, axis=1)

    def assign_tensors(self):
        return self.assign_op

    def assign_add_tensors(self):
        return self.add_op

    def compress(self):
        return CompressedModel(self.start_seed, self.evolve_seeds)


def random_seed():
    # TODO: create random seed (independent from tf)
    sigma = 0.5
    seed = 123
    return sigma, seed


class CompressedModel:
    def __init__(self, start_seed=None, evolve_seeds=None, scope='unnamed_model'):
        self.start_seed = start_seed if start_seed is not None else random_seed()
        self.evolve_seeds = evolve_seeds if evolve_seeds is not None else []
        self.scope = scope

    def evolve(self, sigma, rng_state=None):
        self.evolve_seeds.append((sigma, rng_state if rng_state is not None else random_seed()))
