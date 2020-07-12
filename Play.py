import gym
import gym_battleships
import random

from stable_baselines import ACKTR, DQN
from stable_baselines.common.vec_env import DummyVecEnv
from Config import Config
# Config: First Argument: BoardSize, Second: Ships, Third: Ships placed with Gap or not
from Result import Result

# Inits config class
config = Config(5, [3, 2, 2], True, False, False)

# Inits Battleship gym environment
envTmp = gym.make('Battleships-v0', config=config)

#Wrap environment into a vector environment
env = DummyVecEnv([lambda: envTmp])

# Choose to display board
print("Diplay board: Yes (1), No (0)")
choiceRender = bool(int(input()))

# Choose Model
randomAgent = True
print("Choose Agent: Radom (1), ACKTR (2), DQN (3)")
choice = int(input())
if choice == 2:
    # Load ACKTR Model
    model = ACKTR.load("./ACKTR_Models/ACKTR_5x5_3_2_2_Dynamic.zip", verbose=0, env=env)
    # Disable Random Agent
    randomAgent = False

elif choice == 3:
    # load DQN Model
    model = DQN.load("./DQN_Models/DQN_5x5_3_2_2_Dynamic.zip", verbose=0, env=env)
    # Disable Random Agent
    randomAgent = False

# Inits result Array
results = []
# Iteration: Amount of played Games
for iteration in range(10):
    score = 0
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
        if randomAgent:
            action = random.choice(env.envs[0].env.available_actions)
        # Agent performs a step
        if not randomAgent:
            action, _states = model.predict(observation)
        nextObservation, reward, done, info = env.step(action)
        # Renders the Game state with radar board
        if choiceRender:
            env.render()
        score += reward
        # Add step to result Object
        result.append_history(rounds, action, nextObservation, reward, done, info)
        observation = nextObservation
        # Game is done
        if done:
            print("End of game: Rounds", rounds, "Score", score)
            # Store amount of rounds in result object
            result.set_rounds(rounds)
            # Add current result object to all results
            results.append(result)
print('Finished')
