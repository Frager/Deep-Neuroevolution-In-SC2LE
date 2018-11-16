
class ModelInput:
    def __init__(self, feature_names_list, channel_dims, spacial_dims=None):
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

    # TODO: is this useful?
    def set_block_input(self, block_input):
        self.block_input = block_input

    # TODO: is this useful?
    def get_block_input(self):
        if self.block_input is None:
            print("no block_input set")
        return self.block_input

    def get_feature_names_as_scope(self):
        scope = ''
        for feat_name in self.feature_names_list:
            scope += feat_name + '_'
        scope = scope[:-1]
        return scope

    def get_feature_names(self):
        return self.feature_names_list

