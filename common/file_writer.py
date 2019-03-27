import os
import json
import csv
import sys
from evolution.model_evolvable import CompressedModel


# contains methods for writing/loading data to files


def write_summary(file_dir, generation, generation_timer, truncated_stats, all_stats):
    # creates new csv file at file_dir named summary.csv and writes a summary of the populations stats for one
    # generation
    # If summary.csv already exists, it appends the stats, retaining older generations stats

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    summary_dir = os.path.join(file_dir, 'summary.csv')
    fieldnames = ['Generation', 'time_in_s',
                  't_max_score', 't_mean_score', 't_median_score',
                  'all_max_score', 'all_mean_score', 'all_median_score']
    if not os.path.exists(summary_dir):
        with open(summary_dir, 'w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
            writer.writeheader()
    summary = {'Generation': generation,
               'time_in_s': float(generation_timer),
               't_max_score': float(truncated_stats['max_score']),
               't_mean_score': float(truncated_stats['mean_score']),
               't_median_score': float(truncated_stats['median_score']),
               'all_max_score': float(all_stats['max_score']),
               'all_mean_score': float(all_stats['mean_score']),
               'all_median_score': float(all_stats['median_score'])}
    with open(summary_dir, 'a', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
        writer.writerow(summary)


def save_dict(dictionary, file_dir, file_name):
    # saves a dictionary to json file

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'w') as outfile:
        json.dump(dictionary, outfile, indent=4)


def load_dict(file_dir, file_name):
    # loads a dictionary from json file

    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'r') as infile:
        json_data = json.load(infile)
    return json_data


def load_models(load_path, generation):
    # loads all models (GA individuals) from json files stored in "load_path/generation" and
    # returns it as an array of CompressedModel

    gen_dirs = [f for f in os.listdir(load_path) if not os.path.isfile(os.path.join(load_path, f))]
    if "gen_{}".format(generation) in gen_dirs:
        load_path = os.path.join(load_path, "gen_{}".format(generation))
    else:
        sys.exit("Trying to load generation ({}) from {} that dose not exist".format(generation, load_path))
    print("loading models from :"+load_path)
    model_files = [f for f in os.listdir(load_path) if os.path.isfile(os.path.join(load_path, f))]
    models = []
    for i in range(len(model_files)):
        models.append(load_model(load_path, "model_top_{}.json".format(i+1)))
    # for model in model_files:
    #     models.append(load_model(load_path, model))
    return models


def load_model(file_dir, file_name):
    # loads a model (GA individual) from json file "load_path/file_name" and
    # returns it as CompressedModel

    def dict2model(dictionary):
        return CompressedModel(start_seed=dictionary['start_seed'],
                               evolve_seeds=dictionary['evolve_seeds'],
                               scope=dictionary['scope'])
    model = dict2model(load_dict(file_dir, file_name)['model'])
    return model


def save_models(generation, scored_models, file_dir):
    # saves an array of type CompressedModel to folder "file_dir/generation" as multiple json files

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_dir = os.path.join(file_dir, 'gen_{}'.format(str(generation)))
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    i = 0
    for model, score in scored_models:
        save_file = {
            'score': int(score),
            'model': model.__dict__
        }
        i += 1
        save_dict(save_file, file_dir, "model_top_{}.json".format(i))


def save_model(compressed_model, file_dir, file_name):
    # saves a CompressedModel as json file to "file_dir/file_name"

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, file_name+'.json')
    with open(file_path, 'w') as outfile:
        json.dump(compressed_model.__dict__, outfile, indent=4)

