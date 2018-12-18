from pysc2.lib import actions
from pysc2.lib import features

NUM_FUNCTIONS = len(actions.FUNCTIONS)
ACTION_TYPES = actions.TYPES


CAT = features.FeatureType.CATEGORICAL

# TODO: what do do with observations of unknown length?
flat_feature_spec = dict({
        # "action_result": (0,),  # See error.proto: ActionResult.
        # "alerts": (0,),  # See sc2api.proto: Alert.
        "available_actions": (NUM_FUNCTIONS,),
        # "build_queue": (0, len(getattr(features, "UnitLayer"))),  # pytype: disable=wrong-arg-types
        # "cargo": (0, len(getattr(features, "UnitLayer"))),  # pytype: disable=wrong-arg-types
        "cargo_slots_available": (1,),
        "control_groups": (10, 2),
        "game_loop": (1,),
        "last_actions": (1,),
        # "multi_select": (0, len(getattr(features, "UnitLayer"))),  # pytype: disable=wrong-arg-types
        "player": (len(getattr(features, "Player")),),  # pytype: disable=wrong-arg-types
        "score_cumulative": (len(getattr(features, "ScoreCumulative")),),  # pytype: disable=wrong-arg-types
        "single_select": (1, len(getattr(features, "UnitLayer"))),  # Only (n, 7) for n in (0, 1).
                                                                    #  pytype: disable=wrong-arg-types
})

# from https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/pre_processing.py#L28-L31
is_spacial_action = {}
for name, arg_type in ACTION_TYPES._asdict().items():
    # HACK: we should infer the point type automatically
    is_spacial_action[arg_type] = name in ['minimap', 'screen', 'screen2']


# channel dimensions. Categorical feature types one-hot encoded
def get_screen_dims():
    return [feat.scale ** (feat.type == CAT) for feat in features.SCREEN_FEATURES]


def get_minimap_dims():
    return [feat.scale ** (feat.type == CAT) for feat in features.MINIMAP_FEATURES]


def get_flat_feature_dims(feature_names):
    dims = []
    for feature_name in feature_names:
        dim = get_flat_feature_dim(feature_name)
        dims.append(dim)
    return dims


def get_flat_feature_dim(feature_name):
    dim = flat_feature_spec[feature_name]
    return dim



