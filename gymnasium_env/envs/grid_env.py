

# Inspired by Gymnasium custom env tutorial:
# https://gymnasium.farama.org/introduction/create_custom_env/

import time 

from typing import Optional 

from enum import Enum 

import numpy as np 
import matplotlib.pyplot as plt 

import pygame 
import gymnasium as gym 

class Action(Enum): 
    RIGHT = 0 
    UP = 1 
    LEFT = 2 
    DOWN = 3 

class GridEnvironment(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4, 
    }

    def __init__(
        self, size, traps, goal, render_mode = None, max_steps = 100):
        
        self.size = size 
        self.traps = np.array(traps, dtype=np.int64) 
        self.robot_location = np.array([-1, -1], dtype=np.int64)
        self.goal = np.array(goal, dtype=np.int64) 

        self.window_size = 512 # size fo the pygame 

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size -1, shape=(2,), dtype=np.int64),
                "goal": gym.spaces.Box(0, size-1, shape=(2,), dtype=np.int64),
            }
        )

        # available actions (4 directions) 
        self.action_space = gym.spaces.Discrete(4) 

        # map action numbers to actual movment on the grid 
        self.action_to_direction = {
            Action.RIGHT.value: np.array([0, 1]), # move right col + 1
            Action.UP.value: np.array([-1, 0]), # move up row - 1
            Action.LEFT.value: np.array([0, -1]), # move left col - 1
            Action.DOWN.value: np.array([1, 0]), # move down row + 1
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode 
        self.max_steps = max_steps 
        self.current_step = 0 

        self.window = None 
        self.clock = None 
        ...
    
    # constructing observation 

    def _get_obs(self):
        """
        convert internal state to observation format 
        
        returns: 
            dict: observation with agent and target positions
        """
        return {
            "agent": self.robot_location, 
            "goal" : self.goal
        }
        
    def _get_info(self):
        """
        compute auxilary information for debugging 
        returns: 
            dict: info with distance between agent and target 
        """
        return {
            "distance": np.linalg.norm(
                self.robot_location - self.goal, ord=1 
            )
        }
    
    # reset function 
    # starts a new episode 
    def reset(
        self, seed: Optional[int] = None , 
        options: Optional[int] = None ):
        """
        start a new episode 
        Args: 
            seed: Random seed 
            options: additional configuration 
        
        returns:
            tuple: (observation, info) for the initial state 
        """
        super().reset(seed=seed) 
        self.current_step = 0 
        self.is_goal = self._terminated

        while True and (type(options) == type(None)): 
            # randomly place the robot on the grid 
            self.robot_location = self.np_random.integers(
                0, self.size, size=2, dtype=np.int64
            )

            # check if the starting spawn on the goal and traps 
            if not (self.is_goal() or self._is_trap()):
                break 
        
        if (type(options) != type(None)): 
            self.robot_location = options['start']

        observation = self._get_obs()
        info = self._get_info() 

        return observation, info 


    def _terminated(self):
        return np.array_equal(
            self.robot_location, self.goal)

    # reward function 
    def _reward(self):
        # check if agent reached the target 
        if self._terminated():
            return 20 # receive reward +20 when reach the goal

        # penalize when it lands on the trap 
        if self._is_trap() :
            return -10 
        # penalize when they are in other block 
        return -1 

    # step function 
    # takes action updates the environment state, and returns the results 

    def _is_trap(self):
        return any(
            np.array_equal(
                self.robot_location, trap
            ) for trap in self.traps 
        )

    def _sample_stochastic_action(self, action):
        """
        Probability transition model:
        80% intended action,
        10% one perpendicular direction,
        10% other perpendicular direction.
        """

        if action == Action.RIGHT.value or action == Action.LEFT.value:
            possible_actions = [
                action,
                Action.UP.value,
                Action.DOWN.value,
            ]

        elif action == Action.UP.value or action == Action.DOWN.value:
            possible_actions = [
                action,
                Action.LEFT.value,
                Action.RIGHT.value,
            ]

        probabilities = [0.8, 0.1, 0.1]

        return self.np_random.choice(possible_actions, p=probabilities)

    def step(self, action): 
        """
        Execute one timestep within the environment 
        Args: 
            action: the action to take (0-3 for direction) 
        returns: 
            tuple: (observation, reward, terminated, truncated, info)
        """

        self.current_step += 1 

        old_location = self.robot_location.copy() 

        # sample actual action using probability transition model
        actual_action = self._sample_stochastic_action(action)
        # direction to movement 
        direction = self.action_to_direction[actual_action]
        
        # update agent position, ensuring it says within grid bounds 
        self.robot_location = np.clip(
            self.robot_location + direction, 0, self.size - 1 
        )

        terminated = self._terminated() or self._is_trap()

        # for time out indication - not using this for now 
        no_movement = np.array_equal(old_location, self.robot_location) 
        truncated = (self.current_step >= self.max_steps)

        # reward structure 
        reward = self._reward()  

        observation = self._get_obs()
        info = self._get_info()
        info["no_movement"] = no_movement 
        info["step"] = self.current_step 
        info["intended_action"] = action
        info["actual_action"] = actual_action

        return (
            observation, reward, terminated, 
            truncated, info) 


    def render(self):
        # if self.render_mode == "rgb_array":
        return self._render_frame() 
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init() 
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock() 

        canvas = pygame.Surface(
            (self.window_size, self.window_size) 
        )
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size 
        )

        for trap in self.traps:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * trap[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.goal[::-1], 
                (pix_square_size, pix_square_size),
            ),
        )

        # draw the agent 
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.robot_location[::-1] + 0.5) * pix_square_size, 
            pix_square_size / 3, 
        )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x), 
                (self.window_size, pix_square_size * x), 
                width= 3
            )

            pygame.draw.line(
                canvas, 
                0,
                (pix_square_size * x, 0), 
                (pix_square_size * x, self.window_size), 
                width=3, 
            )

        if self.render_mode == "human":
            self.window.blit(
                canvas, canvas.get_rect()
            )
            pygame.event.pump()
            pygame.display.update() 

            self.clock.tick(
                self.metadata["render_fps"]
            )
        else: 
            return np.transpose(
                np.array(
                    pygame.surfarray.pixels3d(canvas)
                ), axes=(1, 0, 2) 
            )
        
    def close(self):
        if self.window is not None : 
            pygame.display.quit() 
            pygame.quit() 


