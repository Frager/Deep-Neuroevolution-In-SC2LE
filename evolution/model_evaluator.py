from celery_app import app
import time
from evolution.worker_environment import WorkerEnvironment


@app.task(base=WorkerEnvironment)
def evaluate_model(compressed_model, params, max_no_ops=0, max_episodes=-1, shut_down_env=False):
    # model evaluation task for a Celery Worker.
    # Parameters are sent by the GA (Manager)
    # Sets up the environment and agents neural network model structure according to params and
    # decompresses compressed_model for evaluation
    # Evaluates compressed_model and returns the total accumulated reward (its fitness score)

    # could be useful to shut down workers and environments. Currently unused.
    if shut_down_env:
        evaluate_model.shut_down_env()
        return
    evaluate_model.env_params = params
    reward = run_loop(compressed_model, evaluate_model.agent, evaluate_model.env,
                      max_frames=params['step_mul']*params['max_agent_steps'],
                      max_episodes=params['max_episodes'] if max_episodes == -1 else max_episodes,
                      max_no_ops=max_no_ops)
    return reward


# When agents perform max_no_ops number of no_ops in a row, the evaluation stops and a no_op_penalty is added to the
# total accumulated reward
no_op_penalty = -10


# from https://github.com/deepmind/pysc2/blob/master/pysc2/env/run_loop.py
def run_loop(compressed_model, agent, env, max_frames=0, max_episodes=0, max_no_ops=0, debug=False):
    # Basic run loop that coordinates interaction between agent and environment
    # returns list of cumulative scores (each entry from one episode)

    total_frames = 0
    total_episodes = 0
    start_time = time.time()
    # initialize model parameters
    agent.decompress_model(compressed_model=compressed_model)
    if debug:
        agent.print_parameter_count()
    rewards = list()
    while not max_episodes or total_episodes < max_episodes:
        # Reset environment and get initial state
        timestep = env.reset()
        num_no_ops = 0
        total_episodes += 1
        while True:
            total_frames += 1
            # Pass environment state to agent an get action. (value_estimate is unused)
            action, value_estimate = agent.step(timestep['observation'], timestep['available_actions'])

            num_no_ops = num_no_ops + 1 if action[0] == 0 else 0    # action[0] == 0 checks for no_op

            if max_no_ops and num_no_ops >= max_no_ops:             # if max_no_ops no_ops called in a row
                timestep['score_cumulative'][0] += no_op_penalty        # add penalty and stop model evaluation
                rewards.append(timestep['score_cumulative'][0].tolist())
                break

            if all(done is True for done in timestep['dones']):             # if environment reached end step
                rewards.append(timestep['score_cumulative'][0].tolist())        # stop model evaluation
                break

            # Pass action to environment and get next state
            timestep = env.step(action)
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))
    return rewards
