import os 
import yaml 
import traceback 

import numpy as np 
import gymnasium as gym
from tqdm import tqdm 

from gymnasium_env.envs import GridEnvironment

from gridAgent import GridRobot 
import torch 

import logging 
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
)


train = True 
experiment = "experiment"
configs = yaml.safe_load(open("./configs/base.yaml"))

device = torch.device(configs["device"] if torch.cuda.is_available() else "cpu")
print(f"config type {type(configs)}")
configs["device"] = device 

# training hyperparameters
learning_rate = configs["training"]["learning_rate"]
n_episodes = configs["training"]["n_episodes"]
start_epsilon = configs["training"]["start_epsilon"] 
epsilon_decay = start_epsilon/(n_episodes) 
final_epsilon = configs["training"]["final_epsilon"] 
max_episode_steps = configs["training"]["max_episode_steps"] # number of step in an episode  

# create a simple env 
# env = gym.make('CartPole-v1', render_mode="human",max_episode_steps=max_episode_steps)
traps = [
    [1, 2],
    [3, 1],
]

env = gym.make(
    'gymnasium_env/GridEnv-v0',
    size=10,
    traps=traps,
    goal=[3, 3],
    max_episode_steps=max_episode_steps,
)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = GridRobot(
    env, 
    learning_rate, 
    start_epsilon,
    epsilon_decay, 
    final_epsilon,  
    discount_factor=0.98
)

model_path = "./checkpoints/best.pt"
# agent_load_success = agent.load_model(path= model_path)

def train_fn(
    agent: GridRobot, 
    env, 
    num_episodes=max_episode_steps):
    try:
        pbar = tqdm(range(n_episodes), desc='training')
        for episode in pbar:
            # start with initial state 
            obs, info = env.reset() 
            done = False 
            episode_step = 0 

            while not done:
                episode_step += 1 
                # agent chooses action (initially random, gradually more intelligent) 
                action = agent.get_action(obs) 

                # take action and observe result 
                next_obs, reward, terminated, truncated, info = env.step(int(action)) 

                # learn from this experience
                agent.update(
                    obs, action, reward, terminated, next_obs
                )
                # move to next state 
                # done =  truncated if (train) else  (terminated or truncated )
                done =  (terminated or truncated )
                obs = next_obs

            if (episode % 1000) == 0:
                # agent.update_bootstrapped_model()
                pbar.set_postfix(
                    episode_step=episode_step,
                    epsilon=agent.epsilon 
                )
                # if previous_step < episode_step:
                    # previous_step = episode_step
                    # agent.save_model()
            # reduce exploration rate (agent become less random over time) 
            agent.decay_epsilon() 

    except KeyboardInterrupt:
        logging.error(f"Keyboard Interruption")

    except Exception as e: 
        logging.error("Error occurred:", e)
        logging.error(traceback.format_exc())


    # analyzing training result 
    from matplotlib import pyplot as plt 

    def get_moving_avgs(arr, window, convolution_mode):
        """compute moving average to smooth noisy data."""
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window), 
            mode=convolution_mode
        )/window 


    # smooth over a 500-episode window 
    rolling_length = 500 
    fig, axes = plt.subplots(ncols=3, figsize=(12, 5)) 

    # episode rewards (win/loss performance) 
    axes[0].set_title("episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue, 
        rolling_length, 
        "valid" 
    )
    axes[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axes[0].set_ylabel("average reward")
    axes[0].set_xlabel("episode")

    # episode lengths (how many actions per hand) 
    axes[1].set_title("episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue, 
        rolling_length, 
        "valid"
    )
    axes[1].plot(range(len(length_moving_average)), length_moving_average)
    axes[1].set_ylabel("average episode length")
    axes[1].set_xlabel("episode")

    # training error (how much we're still learning) 
    axes[2].set_title("training error")
    training_moving_average = get_moving_avgs(
        agent.training_error, 
        rolling_length, 
        "valid"
    )
    axes[2].plot(range(len(training_moving_average)), training_moving_average)
    axes[2].set_ylabel("temporal difference error")
    axes[2].set_xlabel("step")

    plt.tight_layout() 
    plt.show() 

# test the trained agent 
def test_agent(agent: GridRobot, env, num_episodes=5):
    total_rewards = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # pure exploitation

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            env.render()

            while not done:
                action = agent.get_action(obs)
                logging.info(f"action taken: {action}")

                obs, reward, terminated, truncated, info = env.step(int(action))

                env.render()

                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)
            print(f"Episode {episode}: reward={episode_reward}")

    except KeyboardInterrupt:
        logging.error("Keyboard Interrupt")

    except Exception:
        logging.exception("Error during testing")

    finally:
        agent.epsilon = old_epsilon
        env.close()

    if len(total_rewards) == 0:
        print("No completed test episodes.")
        return

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"test results over {len(total_rewards)} episodes:")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Average reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")
    ...

if __name__ == "__main__":

    if train: 
        train_fn(agent, env) 
    # test your agent 
    # Visualization/testing environment
    test_env = gym.make(
        'gymnasium_env/GridEnv-v0',
        size=10,
        traps=traps,
        goal=[3, 3],
        render_mode="human",
        max_episode_steps=max_episode_steps,
    )
    test_agent(agent, test_env, num_episodes=200) 