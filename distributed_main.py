import sys
import os
from absl import flags
from common.file_writer import *
from evolution.ga import GA
from pysc2.env import sc2_env
from evolution.celery_app import app
import time

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
flags.DEFINE_integer("random_table_size", 10000, "Random table size")


flags.DEFINE_bool("save", True, "save models?")
flags.DEFINE_string("save_to", "unnamed_experiment", "relative file path when saving models")
flags.DEFINE_string("load_from", "", "relative file path for loading models")
flags.DEFINE_integer("save_interval", 1, "save generations in intervals")

flags.DEFINE_integer("max_generations", 0, "number of generations until the algorithm stops")

flags.DEFINE_integer("population", 9, "population per generation")
flags.DEFINE_integer("truncation", 5, "truncation ")
flags.DEFINE_integer("elites", 1, "number of best models to keep unchanged in next generation")

# TODO: truncation, population, elites
# Not necessary when using app.run()
FLAGS(sys.argv)


def main():
    app.control.purge()     # stop all tasks that might still be in queue

    do_save = FLAGS.save
    save_to = FLAGS.save_to
    load_from = FLAGS.load_from
    save_interval = FLAGS.save_interval

    max_generations = FLAGS.max_generations

    work_dir = os.getcwd()
    work_dir = os.path.join(work_dir, 'experiments')

    if load_from != '':
        load_path = os.path.join(work_dir, load_from)
        models_load_path = os.path.join(load_path, 'models')
        models = load_models(models_load_path)
        parameters = load_dict(load_path, 'worker_parameters.json')
        ga_parameters = load_dict(load_path, 'ga_parameters.json')
    else:
        parameters, ga_parameters = parameters_from_flags()
        models = None

    save_path = os.path.join(work_dir, save_to)
    save_dict(parameters, save_path, 'worker_parameters.json')
    save_dict(ga_parameters, save_path, 'ga_parameters.json')
    models_save_path = os.path.join(save_path, 'models')

    ga = GA(population=ga_parameters['population'], compressed_models=models, env_params=parameters)

    generation = 0
    while max_generations == 0 or generation < max_generations:
        generation += 1
        start = time.time()
        scored_models, max_score, mean_score, median_score = ga.evolve_iteration(elites=ga_parameters['elites'],
                                                                                 truncation=ga_parameters['truncation'])
        print('Generation {}: scores: max={} median={} mean={}'.format(generation, max_score, median_score, mean_score))
        end = time.time()
        generation_timer = end-start
        if do_save:
            write_summary(save_path, generation, generation_timer, max_score, mean_score, median_score)
            if generation % save_interval == 0:
                save_models(generation, scored_models, models_save_path)


def parameters_from_flags():
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

    ga_parameters = dict()
    ga_parameters['population'] = FLAGS.population
    ga_parameters['truncation'] = FLAGS.truncation
    ga_parameters['elites'] = FLAGS.elites
    return parameters, ga_parameters


if __name__ == '__main__':
    main()
