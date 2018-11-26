
render = True                   # Whether to render with pygame
feature_screen_size = 32        # Resolution for screen feature layers.
feature_minimap_size = 32       # Resolution for minimap feature layers.
rgb_screen_size = None          # Resolution for rendered screen.
rgb_minimap_size = None         # Resolution for rendered minimap.
action_space = None             # Which action space to use. Needed if you take both feature
use_feature_units = False       # Whether to include feature units.
disable_fog = False             # Whether to disable Fog of War.
max_agent_steps = 0             # Total agent steps.
game_steps_per_episode = None   # Game steps per episode.
max_episodes = 0                # Total episodes.
step_mul = 8                    # Game steps per agent step.
agent_class = "ga.agent.TestAgent"    # Which agent to run, as a python path to an Agent class
agent_race = "random"           # Agent 1's race.
agent2 = "Bot"                  # Second agent, either Bot or agent class.
agent2_race = "random"          # Agent 2's race.
difficulty = "very_easy"        # If agent2 is a built-in Bot, it's strength.
profile = False                 # Whether to turn on code profiling.
trace = False                   # Whether to trace the code execution.
parallel = 1                    # How many instances to run in parallel.
save_replay = True              # Whether to save a replay at the end.
map_name = "MoveToBeacon"       # Name of a map to use.

