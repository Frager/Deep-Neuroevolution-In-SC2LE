from evolution.tasks import *
from evolution.model_evolvable import CompressedModel
import time
import numpy as np
from copy import deepcopy

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

    def get_best_models(self):
        tasks = list()
        for model in self.compressed_models:
            # queue model evaluations
            tasks.append(evaluate_model.delay(model, self.env_params, self.max_no_ops))
        while True:
            # TODO: handle dropped tasks
            # check for finished tasks and get results
            for i in range(len(tasks)):
                if result_is_ready(tasks[i]) and not isinstance(tasks[i], FinishedTask):
                    tasks[i] = FinishedTask(get_result(tasks[i]))
            scores = [convert_result(task.result) for task in tasks]
            if all(score is not None for score in scores):
                break
            time.sleep(1)
        scored_models = list(zip(self.compressed_models, scores))
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models

    def evolve_iteration(self, truncation=10, max_eval=5000, max_no_ops=30):
        self.max_no_ops = max_no_ops
        scored_models = self.get_best_models()
        scores = [s if s >= 0 else 0 for _, s in scored_models]   # don't calculate no_op penalties
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]
        scored_models = scored_models[:truncation]
        # Elitism
        self.compressed_models = [scored_models[0][0]]
        for _ in range(self.population):
            choice = np.random.choice(len(scored_models))
            if scored_models[choice][1] <= 0:
                model = CompressedModel()   # is model scored 0 try a new model instead
            else:
                model = deepcopy(scored_models[choice][0])
            model.evolve()
            self.compressed_models.append(model)
        return scored_models, max_score, mean_score, median_score


# convert score_cumulative array to a singe score value
def convert_result(result):
    if result is None:
        return None
    return result[0]


def result_is_ready(task):
    if task.result is None:
        return False
    return True


def get_result(task):
    # TODO: try catch block
    result = task.get()
    # delete result from backend (in this case redis)
    # task.forget() # Not supported by rabbitrq (works with redis)
    return result
