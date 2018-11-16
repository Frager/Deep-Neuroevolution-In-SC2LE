from pysc2.env import sc2_env


# Just create one environment for testing
def make_env(**params):
    env = sc2_env.SC2Env(**params)
    return env
