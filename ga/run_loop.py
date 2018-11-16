# A run loop for agent/environment interactions
import time
import tensorflow as tf


# from deepmind/blizzard sc2le
def run_loop(agent, env, model_config, max_frames=0, max_episodes=0):
    total_frames = 0
    total_episodes = 0
    start_time = time.time()
    sigma = 0.5
    start_seed = 1
    # TODO: handle seeds
    agent.setup_model(model_config, sigma, start_seed)
    try:
        while not max_episodes or total_episodes < max_episodes:
            total_episodes += 1
            timestep = env.reset()
            # for a in agents:
            #     a.reset()
            while True:
                value_estimates = []
                total_frames += 1
                action, value_estimate = agent.step(timestep['observation'], timestep['available_actions'])
                value_estimates.append(value_estimate)
                if max_frames and total_frames >= max_frames:
                    return
                if all(done is True for done in timestep['dones']):
                    break
                timestep = env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))


