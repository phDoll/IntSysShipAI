import gym
import gym_battleships
import random
import numpy as np
from keras.layers import Dense, Flatten, LeakyReLU, Activation
from keras.utils.generic_utils import get_custom_objects
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents import SARSAAgent
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.backend import sigmoid
from keras.optimizers import Adam

from Config import Config

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': Activation(swish)})

config = Config(3, [2], False, True)
# Inits Battleship gym environment
env = gym.make('Battleships-v0', config=config)
memory = SequentialMemory(limit=50000, window_length=1)

def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(400, activation='elu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = agent(env.observation_space.shape, env.action_space.n)
policy = EpsGreedyQPolicy()
agent = SARSAAgent(model=model, policy=policy, nb_actions=env.action_space.n)
#agent = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=1000,
#target_model_update=1e-2, policy=policy)
#agent.compile(Adam(lr=1e-3), metrics=['mae'])
agent.compile(optimizer="adam", metrics=['mae'])
agent.fit(env, nb_steps = 100000, visualize = False, verbose = 1)
scores = agent.test(env, nb_episodes = 20, visualize= False)
#print('Average score over 100 test games:{}'.format(np.mean(scores.history['episode_reward'])))
print('Average score over 100 test games:', np.mean(scores.history['episode_reward']), ' , average rounds ',np.mean(scores.history['nb_steps']))
