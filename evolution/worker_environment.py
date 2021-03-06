from absl import flags
from celery import Task
import tensorflow as tf
from pysc2.env import sc2_env
import importlib
from evolution.model_input import ModelInput
from evolution.model_output import ModelOutput
from evolution.model_config import ModelConfig
from common.env_wrapper import EnvWrapper
from common.feature_dimensions import NUM_FUNCTIONS
from common.enums import ModelDataFormat as DataFormat
import common.feature_dimensions as feature_dims
from common.random_util import RandomUtil


class WorkerEnvironment(Task):
    # Base class for managing the RL-Environments.
    # (Only) env_params(params) needs to be called before it can be used.
    # When env_params(params) is called to pass environment parameters, (re-)initializes
    #   - SC2 environment
    #   - agent (and neural network model)
    # or reuses them when already initialized and params are the same.

    _agent = None
    _env = None
    _sess = None
    _model_config = None
    _env_params = dict()

    def __init__(self):
        tf.reset_default_graph()
        # PySC2 needs flags parsed or it will cause exceptions
        flags.FLAGS(['distributed_main.py'])

    @property
    def sess(self):
        if self._sess is None:
            self._sess = tf.Session()
        return self._sess

    @sess.setter
    def sess(self, value):
        self._sess = value

    @property
    def model_config(self):
        if self._model_config is None:
            self._model_config = self.setup_model_config()
        return self._model_config

    @model_config.setter
    def model_config(self, value):
        self._model_config = value

    @property
    def env(self):
        if self._env is None:
            self._env = self.setup_environment()
        return self._env

    @property
    def agent(self):
        if self._agent is None:
            self._agent = self.setup_agent()
        return self._agent

    @agent.setter
    def agent(self, value):
        self._agent = value

    @property
    def env_params(self):
        return self._env_params

    @env_params.setter
    def env_params(self, params):
        # Setter for environment parameters.
        # Will set up environment and agent according to parameters in params

        params_changed = False
        # Check if parameters changed
        for key, value in params.items():
            if key in self._env_params.keys():
                if self._env_params[key] == value:
                    continue
            self._env_params[key] = value
            params_changed = True
        # If changed: re-initializes random table, environment and agent
        if params_changed:
            RandomUtil.reinitialize_random_table(size=params['random_table_size'],
                                                 sigma=params['random_table_sigma'],
                                                 seed=params['random_table_seed'])
            self.setup_model_config()
            self.setup_environment()
            self.setup_agent()

    def shut_down_env(self):
        if self._env is not None:
            self._env.close()
        if self._sess is not None:
            self._sess.close()
            self._sess = None
        tf.reset_default_graph()

    def setup_agent(self):
        # Sets up Agent (incl. model configuration for Input/outputs)
        # Returns Agent class

        if self._sess is not None:
            self._sess.close()
            self._sess = None
        tf.reset_default_graph()
        agent_module, agent_name = self._env_params['agent'].rsplit(".", 1)
        agent_cls = getattr(importlib.import_module(agent_module), agent_name)
        self._agent = agent_cls(self.sess, self.model_config, tf.global_variables_initializer)
        return self._agent

    def setup_environment(self):
        # Set up wrapped SC2 environment.
        # Returns EnvWrapper class.

        if self._env is not None:
            self._env.close()
        players = list()
        players.append(sc2_env.Agent(sc2_env.Race[self._env_params['agent_race']]))
        sc2_environment = sc2_env.SC2Env(map_name=self._env_params['map_name'],
                                         players=players,
                                         step_mul=self._env_params['step_mul'],
                                         agent_interface_format=sc2_env.parse_agent_interface_format(
                                             feature_screen=self._env_params['screen_size'],
                                             feature_minimap=self._env_params['screen_size'],
                                             rgb_screen=self._env_params['rgb_screen_size'],
                                             rgb_minimap=self._env_params['rgb_screen_size'],
                                             action_space=self._env_params['action_space'],
                                             use_feature_units=self._env_params['use_feature_units']))
        self._env = EnvWrapper(sc2_environment, self.model_config)
        return self._env

    def setup_model_config(self):
        # creates ModelConfig (used to set up neural network inputs/outputs and ) according to data in environment
        # parameters
        # Returns ModelConfig class

        feature_inputs = list()
        flat_feature_names = self._env_params['features_flat']
        flat_feature_names = flat_feature_names.split(',')
        feature_inputs.append(ModelInput('flat', flat_feature_names,
                                         feature_dims.get_flat_feature_dims(flat_feature_names)))
        if self._env_params['use_minimap']:
            size = self._env_params['screen_size']
            feature_inputs.append(ModelInput('minimap', ['feature_minimap'], feature_dims.get_minimap_dims(), size))
        size = self._env_params['screen_size']
        feature_inputs.append(ModelInput('screen', ['feature_screen'], feature_dims.get_screen_dims(), size))

        arg_outputs = []
        for arg_type in feature_dims.ACTION_TYPES:
            arg_outputs.append(ModelOutput(arg_type, arg_type.sizes[0], feature_dims.is_spacial_action[arg_type]))

        scope = "test"

        self._model_config = ModelConfig(feature_inputs, arg_outputs, size, NUM_FUNCTIONS, DataFormat.NHWC, scope, self._env_params['use_biases'])
        return self._model_config
