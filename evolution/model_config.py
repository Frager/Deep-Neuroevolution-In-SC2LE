class ModelConfig:
    # Class that defines model structure

    def __init__(self, feature_inputs, arg_outputs, size, num_functions, model_format, scope, use_biases):
        self.feature_inputs = feature_inputs    # string of feature input names
        self.arg_outputs = arg_outputs
        self.size = size
        self.model_format = model_format
        self.scope = scope
        self.num_functions = num_functions
        self.use_biases = use_biases
