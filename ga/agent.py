
import tensorflow as tf
import numpy as np
from ga.model_evolvable import ModelEvolvable as Model


class TestAgent:
    def __init__(self, session, model_config, variables_initializer):
        self.session = session
        self.model = None
        self.block_inputs = self.policy = self.value = None
        self.model_config = model_config
        self.feature_names_lists = [feature.feature_names_list for feature in model_config.feature_inputs]
        self.variables_initializer = variables_initializer
        self.available_actions_tensor = tf.placeholder(dtype=tf.float32,
                                                       shape=[None, model_config.num_functions],
                                                       name='input_available_actions')

    def init(self):
        self.session.run(self.variables_initializer())

    def setup_model(self, model_config):
        self.model = Model(scope=model_config.scope)
        self.block_inputs, self.policy, self.value = self.model.fully_conv(model_config)
        self.init()
        self.model_config = model_config
        self.feature_names_lists = [feature.feature_names_list for feature in model_config.feature_inputs]
        return

    def reset_model(self, start_seed):
        # TODO: reset Model with start seed
        return

    def evolve_model(self, evolve_seed):
        # TODO: evolve model with seed
        return

    def compress_model(self):
        # TODO: compress model and return compressed model
        return

    def decompress_model(self, compressed_model):
        # TODO: decompress compressed_model and save it in self.model
        return

    def step(self, obs, available_actions):
        feed_dict = self.input_to_feed_dict(obs)
        feed_dict[self.available_actions_tensor] = available_actions
        p_action_id, p_action_args, value_estimate, available_actions = self.session.run(
            [self.policy[0], self.policy[1], self.value, self.available_actions_tensor],
            feed_dict=feed_dict
        )
        action_id, action_args = sample_actions(p_action_id, p_action_args, available_actions)
        #TODO:
        # ValueError: Wrong number of values
        # for argument of 4 / select_control_group (4 / control_group_act[5]; 5 / control_group_id[10]),
        # got: [ array([0.2119934, 0.18735965, 0.14499538, 0.30552408, 0.15012754], dtype=float32),
        # array([0.09692005, 0.10003376, 0.07594538, 0.1871254, 0.10363444,
        #        0.09784229, 0.07564336, 0.07490392, 0.10570957, 0.08224183], dtype=float32)]

        return [action_id, action_args], value_estimate

    def input_to_feed_dict(self, obs):
        feed_dict = {}
        for i, o in enumerate(obs):
            feed_dict[self.block_inputs[i].name] = o
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