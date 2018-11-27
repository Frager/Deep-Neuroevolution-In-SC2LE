from evolution.tasks import *
from evolution.model_evolvable import CompressedModel
import time


class GA:
    def __init__(self, population, compressed_models):
        self.current_results = list()
        self.current_generation = list()
        self.population = population
        self.compressed_models = [CompressedModel() for _ in range(population)] \
            if compressed_models is None else compressed_models

    @property
    def get_best_models(self):
        tasks = list()
        for model in self.compressed_models:
            # queue model evaluations
            tasks.append(evaluate_model.delay(model))
        while True:
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
    task.forget()
    return result


class FinishedTask:
    def __init__(self, result):
        self.result = result


# app.control.purge()
models = list()
models.append(CompressedModel((0.5, 123)))
models.append(CompressedModel((0.5, 243)))
# models.append(CompressedModel((0.5, 743)))
# models.append(CompressedModel((0.5, 14)))
algo = GA(1, models)

scored_models_test = algo.get_best_models
print(scored_models_test)
