from common.preprocessor import Preprocessor


class EnvWrapper:
    # Wrapper class for SC2Env.
    # Used to fit the data coming from SC2Env to the agents model and vice versa

    def __init__(self, env, model_config):
        self.env = env
        self.model_config = model_config
        self.preprocessor = Preprocessor(model_config)

    def reset(self):
        timesteps = self.env.reset()
        return self.preprocess_timesteps(timesteps)

    def step(self, action):
        processed_actions = self.preprocessor.preprocess_action(action)
        timesteps = self.env.step(processed_actions)
        return self.preprocess_timesteps(timesteps)

    def preprocess_timesteps(self, timesteps):
        obs_raw = [timestep.observation for timestep in timesteps]
        available_actions_raw = [ob.available_actions for ob in obs_raw]
        rewards = [timestep.reward for timestep in timesteps]
        score_cumulative = [timestep.observation['score_cumulative'] for timestep in timesteps]
        dones = [timestep.last() for timestep in timesteps]
        # Available actions get one hot encoded
        available_actions = [self.preprocessor.preprocess_available_actions(available_actions_raw)]
        # raw observations are made to better fit the configuration of the agents model
        processed_obs = self.preprocessor.preprocess_observations(obs_raw)
        return {'observation': processed_obs,
                'rewards': rewards,
                'score_cumulative': score_cumulative,
                'dones': dones,
                'available_actions': available_actions}

    def observation_spec(self):
        return self.env.observation_spec

    def action_spec(self):
        return self.env.action_spec

    def close(self):
        self.env.close()
