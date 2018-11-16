import numpy as np
import itertools
from pysc2.lib.actions import FUNCTIONS, FUNCTION_TYPES, TYPES, FunctionCall
from common.feature_dimensions import is_spacial_action as is_spacial

class Preprocessor:
    def __init__(self, model_config):
        self.model_config = model_config

    def preprocess_observations(self, obs):
        return [self.preprocess_observation(obs, feature_input) for feature_input in self.model_config.feature_inputs]

    def preprocess_observation(self, obs, feat_input):
        processed_observations = []
        for o in obs:
            # if feature_input.is_spacial:
            processed_observations.append(o[feat_input.feature_names_list[0]])
            # TODO: test is this works now
            # else:
            #     concatenated_features = itertools.chain([obs[feature_name]
            #                                              for feature_name in feature_input.feature_names_list])
            #     processed_observations.append(concatenated_features)

        return processed_observations

    def preprocess_available_actions(self, available_actions_raw):
        available_actions = np.zeros(self.model_config.num_functions, dtype=np.float32)
        available_actions[tuple(available_actions_raw)] = 1
        return available_actions

    def preprocess_action(self, actions):
        action_ids, args = actions[0], actions[1]
        calls = []
        for env_index, ids in enumerate(action_ids):
            calls.append(self.to_sc2_action(ids, args))
        return calls

    def to_sc2_action(self, action_id, action_args):
        chosen_function = FUNCTIONS[action_id]
        f_type = chosen_function.function_type
        function_args = FUNCTION_TYPES[f_type]
        processed_args = []
        for arg in function_args:
            action_arg = action_args[arg]
            x = action_arg % self.model_config.size
            y = action_arg // self.model_config.size
            # if action_arc is_spacial
            if is_spacial[arg]:
                action_arg = [action_arg % self.model_config.size.x,
                               action_arg // self.model_config.size.y]
            processed_args.append(action_arg)
        return FunctionCall(chosen_function.id, processed_args)
