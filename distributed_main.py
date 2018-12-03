import sys
import os
from absl import flags
from evolution.model_evolvable import CompressedModel
from evolution.ga import GA
from pysc2.env import sc2_env
from evolution.celery_app import app
import json

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


flags.DEFINE_bool("save", True, "save models?")
flags.DEFINE_bool("load", False, "load models?")
flags.DEFINE_string("save_to", "unnamed_experiment", "relative file path when saving models")
flags.DEFINE_string("load_from", "", "relative file path for loading models")
flags.DEFINE_integer("save_interval", 1, "save generations in intervals")
# TODO: truncation, population, elites
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

    do_save = FLAGS.save
    do_load = FLAGS.load
    save_to = FLAGS.save_to
    load_from = FLAGS.load_from
    save_interval = FLAGS.save_interval

    # TODO: load Models
    # TODO: load params
    app.control.purge()

    work_dir = os.getcwd()
    file_path = os.path.join(work_dir, save_to)

    population = 31
    ga = GA(population=population, compressed_models=None, env_params=parameters)

    generation = 0
    while True:
        generation += 1
        scored_models, max_score, mean_score, median_score = ga.evolve_iteration()
        # TODO: log scores, generation
        print('Generation {}: scores: max={} median={} mean={}'.format(generation, max_score, median_score, mean_score))
        write_summary(file_path, generation, max_score, mean_score, median_score)
        if do_save and generation % save_interval == 0:
            save_models(scored_models, file_path)


def write_summary(file_dir, generation, max_score, mean_score, median_score):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    summary_dir = os.path.join(file_dir, 'summary.json')
    if os.path.exists(summary_dir):
        with open(summary_dir, 'r') as infile:
            summary_file = json.load(infile)
    else:
        summary_file = dict()
    summary_file['generation_{}'.format(generation)] = {
        'max_score': float(max_score),
        'mean_score': float(mean_score),
        'median_score': float(median_score)
    }
    with open(summary_dir, 'w') as outfile:
        json.dump(summary_file, outfile, indent=4)


def save_models(scored_models, file_dir):
    models_path = os.path.join(file_dir, 'Models')
    i = 0
    for model, score in scored_models:
        i += 1
        save_model(model, models_path, "model_top_{}_score_{}".format(i, score))


def save_model(compressed_model, file_dir, file_name):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, file_name+'.json')
    with open(file_path, 'w') as outfile:
        json.dump(compressed_model.__dict__, outfile, indent=4)


def load_model(file_dir, file_name):
    def json2obj(json_data):
        return CompressedModel(start_seed=json_data['start_seed'],
                               evolve_seeds=json_data['evolve_seeds'],
                               scope=json_data['scope'])
    file_path = os.path.join(file_dir, file_name + '.json')
    with open(file_path, 'r') as infile:
        json_file = json.load(infile)
        model = json2obj(json_file)
    return model


if __name__ == '__main__':
    main()

