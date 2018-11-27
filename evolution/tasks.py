from evolution.celery_app import app
import time
from evolution.worker_environment import WorkerEnvoronment


@app.task(base=WorkerEnvoronment)
def evaluate_model(compressed_model, max_frames=0, max_episodes=1):
    reward = run_loop(compressed_model, evaluate_model.agent, evaluate_model.env,
                      max_frames=max_frames,
                      max_episodes=max_episodes)
    return reward


def run_loop(compressed_model, agent, env, max_frames=0, max_episodes=0):
    total_frames = 0
    total_episodes = 0
    start_time = time.time()
    agent.decompress_model(compressed_model=compressed_model)
    rewards = list()
    timestep = None
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
                    rewards.append(timestep['rewards'][0])
                    break
                timestep = env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))
        return timestep['score_cumulative'][0]