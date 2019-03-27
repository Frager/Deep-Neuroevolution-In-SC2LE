import numpy as np
from pysc2.lib.actions import FUNCTIONS, FUNCTION_TYPES, FunctionCall
from common.feature_dimensions import is_spacial_action as is_spacial


class Preprocessor:
    # pre-processes data coming from SC2Env to the agents model and vice versa
    # used by EnvWrapper class

    def __init__(self, model_config):
        self.model_config = model_config

    def preprocess_observations(self, obs):
        observations = {}
        for feature_input in self.model_config.feature_inputs:
            observations[feature_input.input_name] = self.concatinate_features(obs, feature_input.feature_names_list)
        return observations

    def concatinate_features(self, obs, feat_names):
        processed_observations = []
        for o in obs:
            # if feature_input is spacial is not handled
            for feat_name in feat_names:
                if feat_name == "last_actions":
                    if len(o[feat_name]) == 1:
                        processed_observations.append(o[feat_name])
                    else:
                        processed_observations.append([0])
                else:
                    processed_observations.append(o[feat_name])
        return np.concatenate(processed_observations)

    def preprocess_available_actions(self, available_actions_raw):
        available_actions = np.zeros(self.model_config.num_functions, dtype=np.float32)
        available_actions[tuple(available_actions_raw)] = 1
        return available_actions

    def preprocess_action(self, actions):
        # converts model output to FunctionCall for SC2Env

        action_ids, args = actions[0], actions[1]
        function_calls = []
        for ids in action_ids:
            function_calls.append(self.to_sc2_action(ids, args))
        return function_calls

    def to_sc2_action(self, action_id, action_args):
        # creates FunctionCall
        chosen_function = FUNCTIONS[action_id]
        f_type = chosen_function.function_type
        function_args = FUNCTION_TYPES[f_type]
        processed_args = []
        for arg in function_args:
            action_arg = action_args[arg]
            if is_spacial[arg]:
                x = action_arg % self.model_config.size
                y = action_arg // self.model_config.size
                action_arg = [x, y]
            processed_args.append(action_arg)
        return FunctionCall(chosen_function.id, processed_args)
