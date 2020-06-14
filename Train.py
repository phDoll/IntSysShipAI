import gym
import gym_battleships
from Config import Config
from Result import Result

# Inits config class
config = Config(10, [2, 2, 2, 2, 3, 4, 5], True)
# Inits Battleship gym environment
env = gym.make('Battleships-v0', config=config)
# Inits result Array
results = []
# Iteration: Amount of played Games
for iteration in range(20):
    print('Iteration', iteration)
    # Observed Player board
    observation = env.reset()
    # Init new Result
    result = Result()
    done = False
    # Amount of moves used to finish the game
    rounds = 0
    while not done:
        rounds += 1
        # Get a random action from the action space
        action = env.action_space.sample()
        # Agent performs a step
        observation, reward, done, info = env.step(action)
        # Add step to result Object
        result.append_history(rounds, action, observation, reward, done, info)
        #print('observation=', observation, ',reward=', reward, ',done=', done, ',info=', info)
        # Game is done
        if done:
            # Renders the Game state with radar board
            env.render()
            print("End of game: overall_reward=", result.get_overall_reward(), ",rounds", rounds)
            # Store amount of rounds in result object
            result.set_rounds(rounds)
            # Add current result object to all results
            results.append(result)
print('Finished')
