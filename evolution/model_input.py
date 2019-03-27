
class ModelInput:
    # Class that defines model input structure

    def __init__(self, input_name, feature_names_list, channel_dims, spacial_dims=None):
        self.input_name = input_name
        self.feature_names_list = feature_names_list
        self.channel_dims = channel_dims
        self.spacial_dims = spacial_dims
        if spacial_dims is None:
            self.is_spacial = False
        else:
            self.is_spacial = True
        self.block_input = None

    def get_dimensions(self):
        return [(*self.spacial_dims, channel) for channel in self.channel_dims]

    def get_channel_dimensions(self):
        return [*self.channel_dims]

    def get_spacial_dimensions(self):
        return self.spacial_dims

    def get_feature_names_as_scope(self):
        scope = ''
        for feat_name in self.feature_names_list:
            scope += feat_name + '_'
        scope = scope[:-1]
        return scope

    def get_feature_names(self):
        return self.feature_names_list

