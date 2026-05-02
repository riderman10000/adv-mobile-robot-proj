from collections import defaultdict 
import numpy as np 


import gymnasium as gym 
import gymnasium
import gymnasium_env

import logging 
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# environment traps 
traps = [
    [1, 2],
    [3, 1],
]

env = gymnasium.make(
    'gymnasium_env/GridEnv-v0', size=10, traps = traps, goal=[3,3]
)

def obs_to_key(obs):
    agent = tuple(
        np.asarray(obs['agent'], dtype=np.int64))
    goal = tuple(
        np.asarray(obs['goal'], dtype=np.int64))

    # return agent + goal  # if the goal shifts use this
    return agent  

class GridRobot: 
    
    def __init__(
        self, 
        env: gym.Env, 
        learning_rate: float, 
        initial_epsilon: float , 
        epsilon_decay: float, 
        final_epsilon: float, 
        discount_factor: float = 0.95, 
    ):
        self.env = env 

        self.q_values = defaultdict(
           lambda: np.zeros(env.action_space.n) 
        )

        # eta 
        self.lr = learning_rate 
        # beta
        self.discount_factor = discount_factor 

        # exploration parameters 
        self.epsilon = initial_epsilon 
        self.epsilon_decay = epsilon_decay 
        self.final_epsilon = final_epsilon 

        # track learning progress 
        self.training_error = [] 

        self.q_values_hist = [] 

    def get_action(
        self, obs: tuple[int, int, bool]
    ):
        """
        choose an action using epsilon-greedy strategy 
        returns: 
            action: 0 (RIGHT), 1 (UP), 2 (LEFT), or 3 (DOWN) 
        """
        obs_key = obs_to_key(obs)
        
        # with probability epsilon: explore 
        if np.random.random() < self.epsilon: 
            return self.env.action_space.sample() 
        # with probability (1- epsilon): exploit (best known action) 
        return int(np.argmax(self.q_values[obs_key]))
    
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int, 
        reward: float, 
        terminated: bool, 
        next_obs: tuple[int, int, bool]
    ): 
        """
        update q value based on experience Heart of q-learning 
        """
        # what's the best we could do from the next state? 
        # episode terminated then no future rewards possible 

        obs_key = obs_to_key(obs)
        next_obs_key = obs_to_key(next_obs)

        # logging.info(f"next obs value: { next_obs_key}")

        future_q_value = (not terminated) * np.max(self.q_values[next_obs_key])

        # what should the Q-value be ? bellman equation 
        target = reward + self.discount_factor * future_q_value 

        # how wrong was our current situation? 
        temporal_difference = target - self.q_values[obs_key][action]
        
        # update our estimate in the direction of the error 
        self.q_values[obs_key][action] = (
            self.q_values[obs_key][action] + self.lr * temporal_difference 
        )

        # track learning progress (useful for debugging) 
        self.training_error.append(temporal_difference)
   
    def decay_epsilon(self):
        """
        reduce exploration rate after each episode.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def hist(self):
        self.q_values_hist.append(self.q_values.copy())
