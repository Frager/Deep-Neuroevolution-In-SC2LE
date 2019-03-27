from common.random_util import RandomUtil as Random
import numpy as np
from evolution.model_evolvable import ModelEvolvable as Model


class NeuroevolutionAgent:
    # The agent acting in SC2LE.
    # Uses a neural network model to decide on actions when calling step()
    # setup_model() needs to be called to initialize the model first

    def __init__(self, session, model_config, variables_initializer):
        self.session = session
        self.model = None
        self.block_inputs = self.policy = self.value = self.weight_placeholders = self.bias_placeholders = None
        self.model_config = model_config
        self.variables_initializer = variables_initializer
        self.available_actions_input = None

    def init_variables(self):
        self.session.run(self.variables_initializer())

    def setup_model(self, start_seed, evolve_seeds=None):
        # sets up model (fully_conv) if not already done
        # (re-)initializes model parameters using start_seed and then mutates them using evolve seeds

        if self.model is None:
            self.model = Model(scope=self.model_config.scope)
            self.block_inputs, self.policy, self.value, self.weight_placeholders, self.bias_placeholders, self.available_actions_input = self.model.fully_conv(self.model_config)
            self.init_variables()

        self.model_initialize(start_seed)
        if evolve_seeds is not None:
            for seed in evolve_seeds:
                self.model_evolve(seed)
        print('model({}) evolved with {} evolve seeds'.format(start_seed, len(evolve_seeds)))
        return

    def model_initialize(self, start_seed):
        # (re-)initializes model weights using normalized_columns_initializer with random values from a RNG seeded
        # with start_seed
        # model biases are set to 0

        feed_dict = {}
        Random.set_seed(start_seed)
        for placeholder in self.weight_placeholders:
            feed_dict[placeholder] = Random.normalized_columns_initializer(placeholder.shape)
        self.session.run(self.model.initialize_tensors(), feed_dict=feed_dict)

    def model_evolve(self, evolve_seed):
        # mutates model weights and biases by adding values chosen by a RNG seeded with evolve_seed from a
        # pre-initialized table to them

        feed_dict = {}
        Random.set_seed(evolve_seed)
        for placeholder in self.weight_placeholders:
            feed_dict[placeholder] = Random.get_random_values(placeholder.shape)
        for placeholder in self.bias_placeholders:
            feed_dict[placeholder] = Random.get_random_values(placeholder.shape)
        self.session.run(self.model.add_tensors(), feed_dict=feed_dict)

    def compress_model(self):
        # returns CompressedModel class containing start seed and evolve seeds
        # currently unused

        return self.model.compress()

    def decompress_model(self, compressed_model):
        # sets up model according to start seed and evolve seeds in compressed_model (of type CompressedModel)

        start_seed = compressed_model.start_seed
        evolve_seeds = compressed_model.evolve_seeds
        self.setup_model(start_seed, evolve_seeds=evolve_seeds)
        return

    def step(self, obs, available_actions):
        # feeds processed observations obs and one-hot-encoded available_actions to model
        # returns model output

        feed_dict = self.input_to_feed_dict(obs)
        feed_dict[self.available_actions_input] = available_actions
        action_id, action_args, value_estimate = self.session.run(
            [self.policy[0], self.policy[1], self.value],
            feed_dict=feed_dict
        )
        # if model action_id output is invalid overwrite it with action_id of no_op (= 0)
        if available_actions[0][action_id[0]] == 0:
            action_id[0] = 0   # no_op
        return [action_id, action_args], value_estimate

    def print_parameter_count(self):
        # prints number of models weights and biases

        num_biases = 0
        num_weights = 0
        for variable in self.bias_placeholders:
            num_biases += np.prod(variable.get_shape().as_list())
        for variable in self.weight_placeholders:
            num_weights += np.prod(variable.get_shape().as_list())
        num_vars = num_biases + num_weights
        print("Parameter count: {} (w: {} ,b: {} )".format(num_vars, num_weights, num_biases))

    def input_to_feed_dict(self, obs):
        # turns observations to feed_dict needed by session.run

        feed_dict = {}
        for input_name, value in obs.items():
            feed_dict[self.block_inputs[input_name]] = [value]
        return feed_dict

