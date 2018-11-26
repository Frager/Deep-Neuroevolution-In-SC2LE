from ga.tasks import *
from ga.model_evolvable import CompressedModel

seed = (0.5, 123)
compressed_model = CompressedModel(start_seed=seed)

evaluate_model.delay(compressed_model)
