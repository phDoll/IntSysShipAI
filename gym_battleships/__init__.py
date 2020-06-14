from gym.envs.registration import register

register(
    id='Battleships-v0',
    entry_point='gym_battleships.envs:BattleshipsEnv',
    max_episode_steps=2000,
)
