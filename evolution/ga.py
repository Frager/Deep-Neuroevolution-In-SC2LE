from evolution.tasks import *
from evolution.model_evolvable import CompressedModel
import time
import numpy as np
from copy import deepcopy
from celery.states import FAILURE, SUCCESS, RETRY

class FinishedTask:
    def __init__(self, result):
        self.result = result


# from https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66
class GA:
    def __init__(self, population, compressed_models, env_params):
        self.population = population
        self.compressed_models = list()
        if compressed_models is None:
            self.compressed_models = [CompressedModel() for _ in range(population)]
        else:
            self.compressed_models = compressed_models
            if len(compressed_models) < population:
                self.compressed_models += [CompressedModel() for _ in range(population-len(self.compressed_models))]
        self.env_params = env_params
        self.max_no_ops = 0

    def get_best_models(self, models, env_params, max_no_ops, num_evaluations):
        tasks = list()
        for model in models:
            # queue model evaluations
            tasks.append(evaluate_model.delay(model, env_params, max_no_ops=max_no_ops,
                                              max_episodes=num_evaluations))
        while True:
            # TODO: handle exxeptions (and dropped tasks)
            # check for finished tasks and get results
            for i in range(len(tasks)):
                if not isinstance(tasks[i], FinishedTask):
                    if tasks[i].status == SUCCESS:
                        tasks[i] = FinishedTask(get_result(tasks[i]))
                    elif tasks[i].status == FAILURE:    # retry task
                        print('Task Failed')
                        tasks[i].forget()
                        tasks[i] = evaluate_model.delay(
                            models[i], env_params, max_no_ops=max_no_ops,
                            max_episodes=num_evaluations)
            scores = [convert_result(task) for task in tasks]
            if all(score is not None for score in scores):
                break
            time.sleep(1)
        scored_models = list(zip(models, scores))
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models

    def evolve_iteration(self, truncation=10, elites=1, elite_eval=5, max_no_ops=30):
        self.max_no_ops = max_no_ops
        scored_models = self.get_best_models(models=self.compressed_models, env_params=self.env_params,
                                             max_no_ops=self.max_no_ops, num_evaluations=1)
        scores = [s if s >= 0 else 0 for _, s in scored_models]   # don't calculate no_op penalties
        all_median_score = np.median(scores)
        all_mean_score = np.mean(scores)
        all_max_score = scored_models[0][1]

        all_stats = {
            'max_score': all_max_score,
            'mean_score': all_mean_score,
            'median_score': all_median_score
        }
        scored_models = scored_models[:truncation]
        truncated_models = [s_m[0] for s_m in scored_models]
        # Elitism (do more evaluations on truncated models)
        scored_models = self.get_best_models(models=truncated_models, env_params=self.env_params,
                                             max_no_ops=self.max_no_ops, num_evaluations=elite_eval)
        scores = [s if s >= 0 else 0 for _, s in scored_models]
        truncated_median_score = np.median(scores)
        truncated_mean_score = np.mean(scores)
        truncated_max_score = scored_models[0][1]
        truncated_stats = {
            'max_score': truncated_max_score,
            'mean_score': truncated_mean_score,
            'median_score': truncated_median_score
        }

        self.compressed_models = [scored_models[i][0] for i in range(elites)]
        for _ in range(self.population):
            choice = np.random.choice(len(scored_models))
            if scored_models[choice][1] <= 0:
                model = CompressedModel()   # is model scored 0 try a new model instead
            else:
                model = deepcopy(scored_models[choice][0])
            model.evolve()
            self.compressed_models.append(model)
        return scored_models, truncated_stats, all_stats


# convert score_cumulative array to a singe score value
def convert_result(task):
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
    # TODO: try catch block
    result = task.get()
    # delete result from backend (in this case redis)
    task.forget()   # Not supported by rabbitrq (works with redis)
    return result
