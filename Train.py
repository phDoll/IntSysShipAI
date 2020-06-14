import gym
import gym_battleships
from Config import Config
# Config: First Argument: BoardSize, Second: Ships, Third: Ships placed with Gap or not
from Result import Result

config = Config(10, [2, 2, 2, 2, 3, 4, 5], True)
env = gym.make('Battleships-v0', config=config)
#dokumentation + Readme.md
results = []
for iteration in range(20):
    print('Iteration', iteration)
    observation = env.reset()
    result = Result()
    done = False
    rounds = 0
    while not done:
        rounds += 1
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        result.append_history(rounds, action, observation, reward, done, info)
        #print('observation=', observation, ',reward=', reward, ',done=', done, ',info=', info)
        if done:
            env.render()
            print("End of game: overall_reward=", result.get_overall_reward(), ",rounds", rounds)
            result.set_reward(reward)
            result.set_rounds(rounds)
            results.append(result)
print('Finished')
