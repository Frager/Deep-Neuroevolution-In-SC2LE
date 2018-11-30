from common.random_util import RandomUtil as Random
import numpy as np
from evolution.model_evolvable import ModelEvolvable as Model
import time


class TestAgent:
    def __init__(self, session, model_config, variables_initializer):
        self.session = session
        self.model = None
        self.block_inputs = self.policy = self.value = self.weight_placeholders = None
        self.model_config = model_config
        self.variables_initializer = variables_initializer
        self.available_actions_input = None

    def init_variables(self):
        self.session.run(self.variables_initializer())

    def setup_model(self, start_seed, evolve_seeds=None):
        if self.model is None:
            self.model = Model(scope=self.model_config.scope)
            self.block_inputs, self.policy, self.value, self.weight_placeholders, self.available_actions_input = self.model.fully_conv(self.model_config)
            self.init_variables()

        # # To Time evolution:
        # for i in range(100):
        #     start = time.clock()
        #     self.model_initialize(start_seed[1])
        #     end = time.clock()
        #     print("{}: time: {}".format(i, (end-start)))

        self.model_initialize(start_seed[1])
        if evolve_seeds is not None:
            for seed in evolve_seeds:
                self.model_evolve(seed[1])

        # for variable in self.model.variables_collection:
        #     print(variable.eval())
        return

    def model_initialize(self, start_seed):
        feed_dict = {}
        # xavier initialize weights
        for placeholder, n_in_out in zip(self.weight_placeholders, self.model.weights_in_out):
            feed_dict[placeholder] = Random.xavier_initializer(placeholder.shape, n_in_out[0], n_in_out[1], start_seed)
        self.session.run(self.model.assign_tensors(), feed_dict=feed_dict)

        # set biases to zeros
        self.session.run(self.model.reset_bias_op)
        # TODO: delete when finished testing
        print(self.model.biases[0].eval(session=self.session))

    def model_evolve(self, evolve_seed):
        feed_dict = {}
        for placeholder in self.weight_placeholders:
            feed_dict[placeholder] = Random.get_random_values(placeholder.shape, evolve_seed)
        self.session.run(self.model.assign_add_tensors(), feed_dict=feed_dict)
        # TODO: what about biases?
        # TODO: delete when finished testing
        print(self.model.weights[0].eval(session=self.session))

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
        return [action_id, action_args], value_estimate

    def input_to_feed_dict(self, obs):
        feed_dict = {}
        for input_name, value in obs.items():
            # TODO: [value] is only a temporary solution for batched observations (which I don't think I need)
            feed_dict[self.block_inputs[input_name]] = [value]
        return feed_dict

