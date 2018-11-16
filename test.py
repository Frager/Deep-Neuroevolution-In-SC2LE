from pysc2.lib.actions import FUNCTIONS, FUNCTION_TYPES, TYPES, FunctionCall


def to_sc2_action(action_id, action_args):

    function = FUNCTIONS[action_id]
    print(function)
    type = function.function_type
    print(type)
    function_args = []
    function_types = FUNCTION_TYPES[type]
    print(function_types)
    for type in function_types:
        function_args.append(action_args[TYPES.index(type)])
    return FunctionCall(function, function_args)


args = ['screen', 'minimap', 'screen2', 'queued', 'control_group_act', 'control_group_id', 'select_point_act', 'select_add', 'screen', 'screen', 'screen', 'screen']

print(to_sc2_action(3, args))
