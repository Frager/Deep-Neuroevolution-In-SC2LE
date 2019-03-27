import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from common.enums import ModelDataFormat as DataFormat
from evolution.model_base import BaseModel
from common.random_util import RandomUtil

class ModelEvolvable(BaseModel):

    def __init__(self, scope):
        super(ModelEvolvable, self).__init__(scope)

        self.start_seed = None
        self.evolve_seeds = []
        self.use_biases = False
        self.data_format = None     # NHWC or NCHW

        self.weight_placeholders = []
        self.bias_placeholders = []
        self.initialize_op = None
        self.add_op = None
        self.reset_bias_op = None

    def spacial_block(self, name,  spacial_dims, channel_dims):
        # Creates spacial block:
        # Pre-processing of spatial features:
        #   If Categorical:
        #       One hot encoding in channel dimension. Then 1x1 convolution with log2(channel_dimension) filters
        #   Else:
        #       Rescaling using logarithmic transformation
        # Concatenates all spatial features in channel dimension
        # Two subsequent convolutional layers (filter sizes 5 and 3, Strides 1 and 1)
        # returns second convolutional layers output and spatical feature input
        block_input = tf.placeholder(tf.float32, [None, len(channel_dims), spacial_dims, spacial_dims])

        # Preprocess observations
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
                block[i] = self.conv(block[i], name + '/one_hot_conv_' + str(i), kernel_size=1, num_outputs=max(1, int(round(np.log2(d)))), stride=1,
                                     padding="SAME", bias=self.use_biases)
            else:
                block[i] = tf.log(block[i] + 1.0)
        block = tf.concat(block, axis=channel_axis)

        conv1 = self.conv(block, name + '/conv_1', kernel_size=5, num_outputs=16, stride=1, padding="SAME", bias=self.use_biases)
        conv2 = self.conv(conv1, name + '/conv_2', kernel_size=3, num_outputs=32, stride=1, padding="SAME", bias=self.use_biases)
        return conv2, block_input

    def non_spacial_block(self, dim, spacial_size):
        # Creates non spatial block and input for flat features
        # Broadcasts flat features along channel dimension
        # returns broadcasted features and flat feature input

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
        # creates model equivalent to DeepMind's FullyConv architecture
        # model_config contains feature input names and dimensions and output structures
        # returns inputs, policy and (unused) value outputs as well as lists of all created weights and biases

        spacial_size = model_config.size
        self.data_format = model_config.model_format
        self.use_biases = model_config.use_biases
        feature_inputs = model_config.feature_inputs
        arg_outputs = model_config.arg_outputs
        num_functions = model_config.num_functions
        if not isinstance(self.data_format, DataFormat):
            raise ValueError('variable data_format is not of type ModelDataFormat')
        block_inputs = {}
        blocks = []

        # Create spatial and non spatial block
        for feature_input in feature_inputs:
            if feature_input.is_spacial:
                block, spacial_input = self.spacial_block(feature_input.input_name,
                                                          feature_input.get_spacial_dimensions(),
                                                          feature_input.get_channel_dimensions())
                block_inputs[feature_input.input_name] = spacial_input
                blocks.append(block)
            else:
                block, non_spacial_input = self.non_spacial_block(np.sum(feature_input.get_channel_dimensions()),
                                                                  spacial_size)
                block_inputs[feature_input.input_name] = non_spacial_input
                blocks.append(block)

        # concatenate blocks to full state representation
        if self.data_format == DataFormat.NCHW:
            state = tf.concat(blocks, axis=1)
        else:
            state = tf.concat(blocks, axis=3)

        # from https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/networks/fully_conv.py#L131-L137
        # state to action id and value
        flat_state = layers.flatten(state)
        fully_connected = tf.nn.relu(self.dense(flat_state, 'flat_state', size=256, bias=self.use_biases))
        value = self.dense(fully_connected, 'value', size=1, bias=self.use_biases)
        value = tf.reshape(value, [-1])
        fn_out, available_actions_input = self.function_id_output(fully_connected, num_functions)

        # state to non-spacial/spatial action arguments (for each action type)
        spacial = 0
        non_spacial = 0
        args_out = {}
        for arg_output in arg_outputs:
            if arg_output.is_spacial:
                arg_out = self.spatial_output(state, 'arg_out_spacial_' + str(spacial))
                spacial += 1
            else:
                arg_out = self.non_spatial_output(fully_connected, 'arg_out_non_spacial_' + str(non_spacial),
                                                  arg_output.arg_size)
                non_spacial += 1
            args_out[arg_output.arg_type] = arg_out

        policy = (fn_out, args_out)
        self.create_assign_operations()

        return block_inputs, policy, value, self.weight_placeholders, self.bias_placeholders, available_actions_input

    def function_id_output(self, x, channels):
        # returns action id output and (one-hot-encoded) available actions input
        logits = self.dense(x, 'func_id', size=channels, bias=self.use_biases)

        # mask function id outputs
        available_actions_input = tf.placeholder(dtype=tf.float32, shape=logits.shape)
        masked_logits = tf.multiply(logits, available_actions_input)
        return tf.math.argmax(masked_logits, axis=1), available_actions_input

    # from https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/networks/fully_conv.py#L76-L78
    def non_spatial_output(self, x, name, channels):
        # returns non spatial action argument output

        logits = self.dense(x, name, size=channels, bias=self.use_biases)
        return tf.math.argmax(logits, axis=1)

    # from https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/networks/fully_conv.py#L80-L84
    def spatial_output(self, x, name):
        # returns spatial action argument output

        logits = self.conv(x, name, kernel_size=1, num_outputs=1, stride=1, padding="SAME", bias=self.use_biases)
        if self.data_format == DataFormat.NHWC:
            # return output back to NCHW format
            logits = layers.flatten(tf.transpose(logits, [0, 3, 1, 2]))
        else:
            logits = layers.flatten(logits)
        return tf.math.argmax(logits, axis=1)

    def initialize_tensors(self):
        return self.initialize_op

    def add_tensors(self):
        return self.add_op

    def create_assign_operations(self):
        # Creates tf operations initialize_op and add_op
        # initialize_op: sets all weights to placeholder values and sets biases to 0
        # add_op: adds all placeholder values to weights and biases

        assign_weight_ops = []
        add_ops = []
        reset_bias_ops = []
        for weight in self.weights:
            placeholder = tf.placeholder(dtype=tf.float32, shape=weight.shape)
            self.weight_placeholders.append(placeholder)
            assign_weight_ops.append(weight.assign(placeholder))
            add_ops.append(weight.assign_add(placeholder))
        for bias in self.biases:
            placeholder = tf.placeholder(dtype=tf.float32, shape=bias.shape)
            self.bias_placeholders.append(placeholder)
            add_ops.append(bias.assign_add(placeholder))
            reset_bias_ops.append(bias.assign(tf.zeros(bias.shape, dtype=tf.float32)))
        self.initialize_op = tf.group(*assign_weight_ops, *reset_bias_ops)
        self.add_op = tf.group(*add_ops)

    def compress(self):
        return CompressedModel(self.start_seed, self.evolve_seeds)


def random_seed():
    seed = RandomUtil.get_random_seed()
    return seed


class CompressedModel:
    # Compressed representation of model.
    # Consists of only seeds (and optional scope name)

    def __init__(self, start_seed=None, evolve_seeds=None, scope='unnamed_model'):
        self.start_seed = start_seed if start_seed is not None else random_seed()
        self.evolve_seeds = evolve_seeds if evolve_seeds is not None else []
        self.scope = scope

    def evolve(self, evolve_seed=None):
        self.evolve_seeds.append(evolve_seed if evolve_seed is not None else random_seed())

