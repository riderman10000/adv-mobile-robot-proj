from gymnasium.envs.registration import register 

register(
    id="gymnasium_env/GridEnv-v0",
    entry_point="gymnasium_env.envs:GridEnvironment",
)
