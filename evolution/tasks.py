from evolution.celery_app import app
import time
from evolution.worker_environment import WorkerEnvironment


# , autoretry_for=(Exception,),
#           retry_kwargs={'max_retries': 5, 'countdown': 10}, retry_jitter=True)
@app.task(base=WorkerEnvironment)
def evaluate_model(compressed_model, params, max_no_ops=0, max_episodes=-1, shut_down_env=False):
    # TODO: compare env_params and set up accordingly
    if shut_down_env:
        evaluate_model.shut_down_env()
        return
    evaluate_model.env_params = params
    reward = run_loop(compressed_model, evaluate_model.agent, evaluate_model.env,
                      max_frames=params['step_mul']*params['max_agent_steps'],
                      max_episodes=params['max_episodes'] if max_episodes == -1 else max_episodes,
                      max_no_ops=max_no_ops)
    return reward


no_op_penalty = -10


def run_loop(compressed_model, agent, env, max_frames=0, max_episodes=0, max_no_ops=0):
    total_frames = 0
    total_episodes = 0
    start_time = time.time()
    agent.decompress_model(compressed_model=compressed_model)
    rewards = list()
    # try:
    while not max_episodes or total_episodes < max_episodes:
        timestep = env.reset()
        num_no_ops = 0
        total_episodes += 1
        # for a in agents:
        #     a.reset()
        while True:
            value_estimates = []
            total_frames += 1
            action, value_estimate = agent.step(timestep['observation'], timestep['available_actions'])

            value_estimates.append(value_estimate)
            # if max_frames and total_frames >= max_frames:
            #     return

            num_no_ops = num_no_ops + 1 if action[0] == 0 else 0    # action[0] == 0 checks for no_op
            if max_no_ops and num_no_ops >= max_no_ops:             # if max_no_ops no_ops called in a row
                timestep['score_cumulative'][0] += no_op_penalty    # add penalty and stop model evaluation
                rewards.append(timestep['score_cumulative'][0].tolist())
                break
            if all(done is True for done in timestep['dones']):
                rewards.append(timestep['score_cumulative'][0].tolist())
                break
            timestep = env.step(action)
    # except KeyboardInterrupt:
    #     print('KeyboardInterrupt')
    #    pass
    # finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))
    return rewards
