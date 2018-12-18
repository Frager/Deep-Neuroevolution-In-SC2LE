from common.random_util import RandomUtil as Random
import numpy as np
from evolution.model_evolvable import ModelEvolvable as Model
import time


class TestAgent:
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
        if self.model is None:
            self.model = Model(scope=self.model_config.scope)
            self.block_inputs, self.policy, self.value, self.weight_placeholders, self.bias_placeholders, self.available_actions_input = self.model.fully_conv(self.model_config)
            self.init_variables()

        # # To Time evolution:
        # for i in range(100):
        #     start = time.clock()
        #     self.model_initialize(start_seed[1])
        #     end = time.clock()
        #     print("{}: time: {}".format(i, (end-start)))

        self.model_initialize(start_seed)
        if evolve_seeds is not None:
            for seed in evolve_seeds:
                self.model_evolve(seed)
        print('model({}) evolved with {} evolve seeds'.format(start_seed, len(evolve_seeds)))
        # for variable in self.model.variables_collection:
        #     print(variable.eval())
        return

    def model_initialize(self, start_seed):
        feed_dict = {}
        # xavier initialize weights
        Random.set_seed(start_seed)
        for placeholder in self.weight_placeholders:
            feed_dict[placeholder] = Random.xavier_initializer(placeholder.shape)
        self.session.run(self.model.initialize_tensors(), feed_dict=feed_dict)

    def model_evolve(self, evolve_seed):
        feed_dict = {}
        Random.set_seed(evolve_seed)
        for placeholder in self.weight_placeholders:
            feed_dict[placeholder] = Random.get_random_values(placeholder.shape)
        for placeholder in self.bias_placeholders:
            feed_dict[placeholder] = Random.get_random_values(placeholder.shape)
        self.session.run(self.model.add_tensors(), feed_dict=feed_dict)
        # TODO: what about biases?

    def compress_model(self):
        return self.model.compress()

    def decompress_model(self, compressed_model):
        start_seed = compressed_model.start_seed
        evolve_seeds = compressed_model.evolve_seeds
        self.setup_model(start_seed, evolve_seeds=evolve_seeds)
        return

    def step(self, obs, available_actions):
        feed_dict = self.input_to_feed_dict(obs)
        feed_dict[self.available_actions_input] = available_actions
        action_id, action_args, value_estimate = self.session.run(
            [self.policy[0], self.policy[1], self.value],
            feed_dict=feed_dict
        )
        # # To get parameter count
        # num_vars = 0
        # for variable in self.bias_placeholders + self.weight_placeholders:
        #     num_vars += np.prod(variable.get_shape().as_list())
        # print(num_vars)
        if available_actions[0][action_id[0]] == 0:
            action_id[0] = 0   # no_op
        return [action_id, action_args], value_estimate

    def input_to_feed_dict(self, obs):
        feed_dict = {}
        for input_name, value in obs.items():
            # TODO: [value] is only a temporary solution for batched observations (which I don't think I need)
            feed_dict[self.block_inputs[input_name]] = [value]
        return feed_dict

