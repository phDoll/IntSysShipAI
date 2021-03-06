import gym
import gym_battleships

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines import ACKTR
from Config import Config
from Result import Result

# Inits Battleship gym environments and config
config = Config(5, [3, 2, 2], True, False, False)
env2 = gym.make('Battleships-v0', config=config)
env3 = gym.make('Battleships-v0', config=config)
env = DummyVecEnv([lambda: env2])
env4 = DummyVecEnv([lambda: env3])


# Define Callback
#Callback stops training if maximum is reached in mean reward
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=env2.calculate_threshold(), verbose=1)
# Callback safes the currently best model
eval_callback = EvalCallback(env4, callback_on_new_best=callback_on_best, verbose=1, best_model_save_path='./ACKTR_Models/best/')
checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path='./model_checkpoints/')


# Uncomment, to train a new fresh model, otherwise a allready trained model will be trained
# If a frehs model is trained, it should be trained with binary reward (Config) first, to reduce multiple
# shots onto the same field.
#model = ACKTR(MlpPolicy, env, verbose=2, tensorboard_log="./logs/progress_tensorboard/",  n_cpu_tf_sess=4)

# Load current best model
model = ACKTR.load("./ACKTR_Models/best/best_model.zip", verbose=2, env=env, tensorboard_log="./logs/progress_tensorboard/")

# Train model
model.learn(1000000, callback=[checkpoint_callback, eval_callback])

# Delete current model and load the best model
del model
model = ACKTR.load("./ACKTR_Models/best/best_model.zip", verbose=2, env=env, tensorboard_log="./logs/progress_tensorboard/")



# Test trained model
results = []
for iteration in range(100):
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
        # action = env.action_space.sample()
        # Agent performs a step
        # observation, reward, done, info = env.step(action)
        action, _states = model.predict(observation)
        nextObservation, reward, done, info = env.step(action)
        score += reward
        # Add step to result Object
        result.append_history(rounds, action, nextObservation, reward, done, info)
        observation = nextObservation
        # Game is done
        if done:
            print("End of game: overall_reward=", result.get_overall_reward(), ",rounds", rounds, "score", score)
            # Store amount of rounds in result object
            result.set_rounds(rounds)
            # Add current result object to all results
            results.append(result)
print('Finished')

