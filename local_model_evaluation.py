from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from evolution.model_input import ModelInput
from evolution.model_output import ModelOutput
import tensorflow as tf
import common.feature_dimensions as feature_dims
from common.enums import ModelDataFormat as DataFormat
from evolution.model_config import ModelConfig
from common.env_wrapper import EnvWrapper
from common.feature_dimensions import NUM_FUNCTIONS
from common.file_writer import *
from evolution.model_evaluator import run_loop
from common.random_util import RandomUtil
import importlib


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("render", True, "Whether to render with pygame.")
    point_flag.DEFINE_point("feature_screen_size", "32",
                            "Resolution for screen feature layers.")
    point_flag.DEFINE_point("feature_minimap_size", "32",
                            "Resolution for minimap feature layers.")
    point_flag.DEFINE_point("rgb_screen_size", None,
                            "Resolution for rendered screen.")
    point_flag.DEFINE_point("rgb_minimap_size", None,
                            "Resolution for rendered minimap.")
    flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                      "Which action space to use. Needed if you take both feature "
                      "and rgb observations.")
    flags.DEFINE_bool("use_feature_units", False,
                      "Whether to include feature units.")
    flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

    flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
    flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
    flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
    flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

    flags.DEFINE_string("agent", "evolution.agent.TestAgent",
                        "Which agent to run, as a python path to an Agent class.")
    flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                      "Agent 1's race.")

    flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
    flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                      "Agent 2's race.")
    flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                      "If agent2 is a built-in Bot, it's strength.")

    flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
    flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
    flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

    flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")
    flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")

    flags.DEFINE_string("load_from", "", "relative file path for loading models")
    flags.DEFINE_integer("gen", "0", "from generation")
    flags.DEFINE_integer("top", "1", "model placement. Starting from the best = 1")

    FLAGS(sys.argv)
    tf.reset_default_graph()

    agent_classes = []
    players = []

    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    agent_classes.append(agent_cls)

    work_dir = os.getcwd()
    work_dir = os.path.join(work_dir, 'experiments')
    load_from = FLAGS.load_from

    load_path = os.path.join(work_dir, load_from)
    model_load_path = os.path.join(load_path, 'models')
    model_load_path = os.path.join(model_load_path, 'gen_{}'.format(FLAGS.gen))
    model = load_model(model_load_path, "model_top_{}.json".format(FLAGS.top))
    print("model {} loaded".format("model_top_{}.json".format(FLAGS.top)))
    parameters = load_dict(load_path, 'worker_parameters.json')
    ga_parameters = load_dict(load_path, 'ga_parameters.json')

    players.append(sc2_env.Agent(sc2_env.Race[parameters['agent_race']]))

    if 'use_biases' not in parameters:
        parameters['use_biases'] = True     # for compatibility with experiments from older versions

    feature_inputs = list()
    flat_feature_names = parameters['features_flat']
    flat_feature_names = flat_feature_names.split(',')
    feature_inputs.append(ModelInput('flat', flat_feature_names, feature_dims.get_flat_feature_dims(flat_feature_names)))
    spacial_size = parameters['screen_size']
    if parameters['use_minimap']:
        feature_inputs.append(ModelInput('minimap', ['feature_minimap'], feature_dims.get_minimap_dims(), spacial_size))
    feature_inputs.append(ModelInput('screen', ['feature_screen'], feature_dims.get_screen_dims(), spacial_size))

    arg_outputs = []
    for arg_type in feature_dims.ACTION_TYPES:
        arg_outputs.append(ModelOutput(arg_type, arg_type.sizes[0], feature_dims.is_spacial_action[arg_type]))

    scope = "test"
    RandomUtil.reinitialize_random_table(size=parameters['random_table_size'],
                                         sigma=parameters['random_table_sigma'],
                                         seed=parameters['random_table_seed'])
    model_config = ModelConfig(feature_inputs, arg_outputs, spacial_size, NUM_FUNCTIONS, DataFormat.NHWC, scope, parameters['use_biases'])

    sc2_env = sc2_env.SC2Env(map_name=parameters['map_name'],
                             players=players,
                             agent_interface_format=sc2_env.parse_agent_interface_format(
                             feature_screen=parameters['screen_size'],
                             feature_minimap=parameters['screen_size'],
                             rgb_screen=parameters['rgb_screen_size'],
                             rgb_minimap=parameters['rgb_screen_size'],
                             action_space=parameters['action_space'],
                             use_feature_units=parameters['use_feature_units']))
    env = EnvWrapper(sc2_env, model_config)
    with tf.Session() as sess:
        agent = agent_cls(sess, model_config, tf.global_variables_initializer)
        run_loop(model, agent, env, max_frames=0, max_episodes=0, max_no_ops=0, debug=True)
