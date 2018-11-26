
import sys
from absl import flags
from celery import Task
import tensorflow as tf
import common.env as environmrnt
from pysc2.env import sc2_env
from common.env_parameters import *
import importlib
from ga.model_input import ModelInput
from ga.model_output import ModelOutput
from ga.model_config import ModelConfig
from common.env_wrapper import EnvWrapper
from common.feature_dimensions import NUM_FUNCTIONS
from common.enums import ModelDataFormat as DataFormat
import common.feature_dimensions as feature_dims

class WorkerEnvoronment(Task):
    _agent = None
    _env = None
    _sess = None
    _model_config = None

    def __init__(self):
        tf.reset_default_graph()
        flags.FLAGS(['main.py'])

    @property
    def sess(self):
        if self._sess is None:
            # TODO: is this correct? Do I need to worry about closing the session?
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

    @env.setter
    def env(self, value):
        self._env = value

    @property
    def agent(self):
        if self._agent is None:
            self._agent = self.setup_agent()
        return self._agent

    @agent.setter
    def agent(self, value):
        self._agent = value

    def setup_agent(self):
        agent_module, agent_name = agent_class.rsplit(".", 1)
        agent_cls = getattr(importlib.import_module(agent_module), agent_name)
        agent = agent_cls(self.sess, self.model_config, tf.global_variables_initializer)
        return agent

    def setup_environment(self):
        players = list()
        players.append(sc2_env.Agent(sc2_env.Race[agent_race]))
        sc2_environment = environmrnt.make_env(map_name=map_name,
                                               players=players,
                                               agent_interface_format=sc2_env.parse_agent_interface_format(
                                                   feature_screen=feature_screen_size,
                                                   feature_minimap=feature_minimap_size,
                                                   rgb_screen=rgb_screen_size,
                                                   rgb_minimap=rgb_minimap_size,
                                                   action_space=action_space,
                                                   use_feature_units=use_feature_units))
        env_wrapper = EnvWrapper(sc2_environment, self.model_config)
        return env_wrapper

    def setup_model_config(self):
        flat_feature_names = ['player', 'score_cumulative']
        flat_feature_input = ModelInput('flat', flat_feature_names,
                                        feature_dims.get_flat_feature_dims(flat_feature_names))
        size = feature_minimap_size
        minimap_input = ModelInput('minimap', ['feature_minimap'], feature_dims.get_minimap_dims(), size)
        size = feature_screen_size
        screen_input = ModelInput('screen', ['feature_screen'], feature_dims.get_screen_dims(), size)
        feature_inputs = [minimap_input, screen_input, flat_feature_input]

        arg_outputs = []
        for arg_type in feature_dims.ACTION_TYPES:
            arg_outputs.append(ModelOutput(arg_type, arg_type.sizes[0], feature_dims.is_spacial_action[arg_type]))

        scope = "test"

        model_config = ModelConfig(feature_inputs, arg_outputs, size, NUM_FUNCTIONS, DataFormat.NHWC, scope)
        return model_config
