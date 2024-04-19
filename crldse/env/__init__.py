from gym.envs.registration import register

register(
    id='MCPDseEnv-v0',
    entry_point='env_gym_wapper:MCPDseEnv',
)