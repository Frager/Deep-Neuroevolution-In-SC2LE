
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

    def setup_model(self, model_config, sigma, start_seed):
        self.model = Model(scope=model_config.scope, )
        self.block_inputs, self.policy, self.value = self.model.fully_conv(model_config)
        self.init()
        self.reset_model(sigma, start_seed)
        self.model_config = model_config
        self.feature_names_lists = [feature.feature_names_list for feature in model_config.feature_inputs]
        # for variable in self.model.variables_collection:
        #     print(variable.eval())
        return

    def reset_model(self, sigma, start_seed):
        self.session.run(self.model.initialize_with_seed(sigma, start_seed))
        return

    def evolve_model(self, sigma, evolve_seed):
        self.session.run(self.model.evolve(sigma, evolve_seed))
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