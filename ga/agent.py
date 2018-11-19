
import tensorflow as tf
import numpy as np
from ga.model_evolvable import CompressedModel
from ga.model_evolvable import ModelEvolvable as Model


class TestAgent:
    def __init__(self, session, model_config, variables_initializer):
        self.session = session
        self.model = None
        self.block_inputs = self.policy = self.value = self.mutate_inputs = None
        self.model_config = model_config
        self.variables_initializer = variables_initializer
        self.available_actions_tensor = tf.placeholder(dtype=tf.float32,
                                                       shape=[None, model_config.num_functions],
                                                       name='input_available_actions')

    def init_variables(self):
        self.session.run(self.variables_initializer())

    def setup_model(self, start_seed, evolve_seeds=[]):
        self.model = Model(scope=self.model_config.scope)
        self.block_inputs, self.policy, self.value, self.mutate_inputs = self.model.fully_conv(self.model_config)
        self.init_variables()

        self.model_assign_all(start_seed)
        for seed in evolve_seeds:
            self.model_assign_all(seed, do_assign_add=True)

        # for variable in self.model.variables_collection:
        #     print(variable.eval())
        return

    def model_assign_all(self, start_seed, do_assign_add=False):
        sigma = start_seed[0]
        seed = start_seed[1]
        feed_dict = {}
        np.random.seed(seed)
        for mutate_input in self.mutate_inputs:
            feed_dict[mutate_input] = np.random.normal(0, sigma, mutate_input.shape)
        if do_assign_add:
            self.session.run(self.model.assign_add_tensors(), feed_dict=feed_dict)
        else:
            self.session.run(self.model.assign_tensors(), feed_dict=feed_dict)
        print(self.model.variables_collection[0].eval())

    def compress_model(self):
        return self.model.compress()

    def decompress_model(self, compressed_model):
        start_seed = compressed_model.start_seed
        evolve_seeds = compressed_model.evolve_seeds
        self.setup_model(start_seed, evolve_seeds=evolve_seeds)
        return

    def step(self, obs, available_actions):
        feed_dict = self.input_to_feed_dict(obs)
        feed_dict[self.available_actions_tensor] = available_actions
        p_action_id, p_action_args, value_estimate, available_actions = self.session.run(
            [self.policy[0], self.policy[1], self.value, self.available_actions_tensor],
            feed_dict=feed_dict
        )
        print(p_action_id)
        action_id, action_args = sample_actions(p_action_id, p_action_args, available_actions)

        return [action_id, action_args], value_estimate

    def input_to_feed_dict(self, obs):
        feed_dict = {}
        for input_name, value in obs.items():
            # TODO: [value] is only a temporary solution for batched observations (which I don't think I need)
            feed_dict[self.block_inputs[input_name]] = [value]
        return feed_dict


def arg_max(id_probabilities):
    return np.argmax(id_probabilities, 1)


def mask_unavailable_actions(available_actions, fn_pi):
    fn_pi *= available_actions
    return fn_pi


def sample_actions(p_action_id, p_action_args, available_actions):
    masked_ids = mask_unavailable_actions(available_actions, p_action_id)
    action_id = arg_max(masked_ids)
    action_args = dict()
    for arg_type, arg_pi in p_action_args.items():
        action_args[arg_type] = arg_max(arg_pi)

    return action_id, action_args
