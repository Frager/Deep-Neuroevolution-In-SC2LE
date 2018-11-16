from common.preprocessor import Preprocessor


class EnvWrapper:
    def __init__(self, env, model_config):
        self.env = env
        self.model_config = model_config
        self.preprocessor = Preprocessor(model_config)

    def reset(self):
        timesteps = self.env.reset()
        return self.preprocess_timesteps(timesteps)

    def step(self, action):
        # TODO: wrap actions
        processed_actions = self.process_action(action)
        print(processed_actions)
        timesteps = self.env.step(processed_actions)
        return self.preprocess_timesteps(timesteps)

    def preprocess_timesteps(self, timesteps):
        obs_raw = [timestep.observation for timestep in timesteps]
        available_actions_raw = [ob.available_actions for ob in obs_raw]
        rewards = [timestep.reward for timestep in timesteps]
        dones = [timestep.last() for timestep in timesteps]
        available_actions = [self.preprocessor.preprocess_available_actions(available_actions_raw)]
        processed_obs = self.preprocessor.preprocess_observations(obs_raw)
        return {'observation': processed_obs,
                'rewards': rewards,
                'dones': dones,
                'available_actions': available_actions}

    def process_action(self, action):
        return self.preprocessor.preprocess_action(action)

    def observation_spec(self):
        return self.env.observation_spec

    def action_spec(self):
        return self.env.action_spec
