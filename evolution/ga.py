from evolution.model_evaluator import *
from evolution.model_evolvable import CompressedModel
import time
import numpy as np
from copy import deepcopy
from celery.states import FAILURE, SUCCESS


class FinishedTask:
    def __init__(self, result):
        self.result = result


# Code Basis: https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66
class GA:
    # contains code for the Genetic Algorithm
    # Acts as a Manager that queues evaluation tasks to be executed by a Worker network

    def __init__(self, population, compressed_models, env_params):
        # if compressed_models is null: creates random population.
        # else: use compressed_models as population.
        #   if population > len(compressed_models), adds random individuals to fill up difference

        self.population = population        # population count
        self.compressed_models = list()     # list of individuals
        if compressed_models is None:
            self.compressed_models = [CompressedModel() for _ in range(population)]
        else:
            self.compressed_models = compressed_models
            len_models = len(compressed_models)
            if len_models < population:
                for _ in range(self.population - len_models):
                    choice = np.random.choice(len_models)
                    model = deepcopy(compressed_models[choice])
                    model.evolve()
                    self.compressed_models.append(model)
        self.env_params = env_params
        self.max_no_ops = 0

    def evaluate_models(self, models, env_params, max_no_ops, num_evaluations):
        # creates tasks for evaluating models
        # returns sorted list of models with their scores when all tasks returned scores

        tasks = list()
        for model in models:
            # queue model evaluation tasks
            tasks.append(evaluate_model.delay(model, env_params, max_no_ops=max_no_ops,
                                              max_episodes=num_evaluations))
        while True:
            # check for finished tasks every 5 seconds and get results
            for i in range(len(tasks)):
                if not isinstance(tasks[i], FinishedTask):
                    if tasks[i].status == SUCCESS:
                        tasks[i] = FinishedTask(get_result(tasks[i]))
                    elif tasks[i].status == FAILURE:    # retry task when failed
                        print('Task Failed')
                        tasks[i].forget()
                        tasks[i] = evaluate_model.delay(
                            models[i], env_params, max_no_ops=max_no_ops,
                            max_episodes=num_evaluations)
            scores = [convert_result(task) for task in tasks]
            # if all tasks returned scores: break
            if all(score is not None for score in scores):
                break
            time.sleep(5)
        scored_models = list(zip(models, scores))
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models

    def evolve_iteration(self, truncation=10, elites=1, elite_eval=5, max_no_ops=30):
        # Genetic operations for one generational loop.
        # population evaluation -> truncation selection -> elite evaluation -> mutation
        # returns last generations scored population and statistics

        # evaluate population
        self.max_no_ops = max_no_ops
        scored_models = self.evaluate_models(models=self.compressed_models, env_params=self.env_params,
                                             max_no_ops=self.max_no_ops, num_evaluations=1)
        scores = [s if s >= 0 else 0 for _, s in scored_models]   # don't calculate negative scores (from no_op penalty)

        # get stats from population evaluation
        all_median_score = np.median(scores)
        all_mean_score = np.mean(scores)
        all_max_score = scored_models[0][1]
        all_stats = {
            'max_score': all_max_score,
            'mean_score': all_mean_score,
            'median_score': all_median_score
        }
        # truncation selection
        scored_models = scored_models[:truncation]
        truncated_models = [s_m[0] for s_m in scored_models]

        # elite evaluation of elitism (more evaluations on truncated models)
        scored_models = self.evaluate_models(models=truncated_models, env_params=self.env_params,
                                             max_no_ops=self.max_no_ops, num_evaluations=elite_eval)

        # get stats from elite evaluation
        scores = [s if s >= 0 else 0 for _, s in scored_models]
        truncated_median_score = np.median(scores)
        truncated_mean_score = np.mean(scores)
        truncated_max_score = scored_models[0][1]
        truncated_stats = {
            'max_score': truncated_max_score,
            'mean_score': truncated_mean_score,
            'median_score': truncated_median_score
        }

        # add elite to new population
        self.compressed_models = [scored_models[i][0] for i in range(elites)]

        # creates new population by randomly uniformly sampling from best truncated individuals and mutating them
        for _ in range(self.population):
            choice = np.random.choice(len(scored_models))
            if scored_models[choice][1] <= 0:
                model = CompressedModel()   # if model scored 0 try a new randomly initialized model
            else:
                model = deepcopy(scored_models[choice][0])
            model.evolve()
            self.compressed_models.append(model)

        return scored_models, truncated_stats, all_stats


def convert_result(task):
    # convert score_cumulative array to a singe score value
    if isinstance(task, FinishedTask):
        results = task.result
        value = 0
        for result in results:
            value += result[0]
        return value/len(results)
    return None


def result_is_ready(task):
    if task.status == SUCCESS:
        return True
    return False


def get_result(task):
    result = task.get()
    # delete result from backend server
    task.forget()   # Not supported by rabbitrq (works with redis)
    return result
