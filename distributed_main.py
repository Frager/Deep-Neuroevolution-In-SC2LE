import sys
from absl import flags
from evolution.model_evolvable import CompressedModel
from evolution.ga import GA
from pysc2.env import sc2_env
from evolution.celery_app import app

FLAGS = flags.FLAGS
flags.DEFINE_integer("screen_size", "32", "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_size", "32", "Resolution for minimap feature layers.")
flags.DEFINE_integer("rgb_screen_size", None, "Resolution for rendered screen.")
flags.DEFINE_integer("rgb_minimap_size", None, "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 1, "Total episodes.")
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

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_float("random_table_sigma", 0.005, "Sigma for random table.")
flags.DEFINE_integer("random_table_seed", 1234, "Random table initialisation seed")
flags.DEFINE_integer("random_table_size", 1000, "Random table size")
# Not necessary when using app.run()
FLAGS(sys.argv)


def main():
    parameters = dict()
    parameters['screen_size'] = FLAGS.screen_size
    parameters['minimap_size'] = FLAGS.minimap_size
    parameters['rgb_screen_size'] = FLAGS.rgb_screen_size
    parameters['rgb_minimap_size'] = FLAGS.rgb_minimap_size
    parameters['action_space'] = FLAGS.action_space
    parameters['use_feature_units'] = FLAGS.use_feature_units
    parameters['disable_fog'] = FLAGS.disable_fog
    parameters['max_agent_steps'] = FLAGS.max_agent_steps
    parameters['game_steps_per_episode'] = FLAGS.game_steps_per_episode
    parameters['max_episodes'] = FLAGS.max_episodes
    parameters['step_mul'] = FLAGS.step_mul
    parameters['agent'] = FLAGS.agent
    parameters['agent_race'] = FLAGS.agent_race
    parameters['agent2'] = FLAGS.agent2
    parameters['agent2_race'] = FLAGS.agent2_race
    parameters['difficulty'] = FLAGS.difficulty
    parameters['map_name'] = FLAGS.map
    parameters['random_table_sigma'] = FLAGS.random_table_sigma
    parameters['random_table_seed'] = FLAGS.random_table_seed
    parameters['random_table_size'] = FLAGS.random_table_size

    # TODO: load Models
    # TODO: load params
    app.control.purge()

    population = 4

    ga = GA(population=population, compressed_models=None, env_params=parameters)

    generation = 0
    while True:
        generation += 1
        median_score, mean_score, max_score = ga.evolve_iteration()
        print('Generation {}: scores: max={} median={} mean={}'.format(generation, max_score, median_score, mean_score))
        # TODO: log scores, generation
        # TODO: save models and params


if __name__ == '__main__':
    main()
