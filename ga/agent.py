from common.random_util import RandomUtil as Random
import numpy as np
from ga.model_evolvable import ModelEvolvable as Model
import time


class TestAgent:
    def __init__(self, session, model_config, variables_initializer):
        self.session = session
        self.model = None
        self.block_inputs = self.policy = self.value = self.mutate_inputs = None
        self.model_config = model_config
        self.variables_initializer = variables_initializer
        self.available_actions_input = None

    def init_variables(self):
        self.session.run(self.variables_initializer())

    def setup_model(self, start_seed, evolve_seeds=None):
        self.model = Model(scope=self.model_config.scope)
        self.block_inputs, self.policy, self.value, self.mutate_inputs, self.available_actions_input = self.model.fully_conv(self.model_config)
        self.init_variables()

        # # To Time evolution:
        # for i in range(100):
        #     start = time.clock()
        #     self.model_assign_all((1, i), do_assign_add=True)
        #     end = time.clock()
        #     print("{}: time: {}".format(i, (end-start)))

        self.model_assign_all(start_seed)
        if evolve_seeds is not None:
            for seed in evolve_seeds:
                self.model_assign_all(seed, do_assign_add=True)

        # for variable in self.model.variables_collection:
        #     print(variable.eval())
        return

    def model_assign_all(self, start_seed, do_assign_add=False):
        sigma = start_seed[0]
        seed = start_seed[1]
        feed_dict = {}
        for mutate_input in self.mutate_inputs:
            feed_dict[mutate_input] = Random.get_random_values(mutate_input.shape, seed)
        if do_assign_add:
            self.session.run(self.model.assign_add_tensors(), feed_dict=feed_dict)
        else:
            self.session.run(self.model.assign_tensors(), feed_dict=feed_dict)
        print(self.model.variables_collection[0].eval(session=self.session))

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

