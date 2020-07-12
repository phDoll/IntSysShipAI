"""
Class representing a result object
"""
class Result:
    """
    Constructor for a result object
    Arguments:
    history = List of Tuples with (rounds, action, observation, reward, done, info).
    reward = Overall reward of a played game
    rounds = Amount of rounds used to finish the game
    """
    def __init__(self, history=None, reward=0, rounds=0):
        if history is None:
            history = []
        self.history = history
        self.overall_reward = reward
        self.rounds = rounds
        self.hit_miss_ratio = 0

    '''
    Method for appending a round history to result.
    round: Number of the current round
    action: Performed action
    observation: Radar Board
    reward: Current reward for the performed action
    done: Boolean if game has finished
    info: Debug info
    '''
    def append_history(self, round, action, observation, reward, done, info):
        self.history.append((round, action, observation, reward, done, info))
        #self.calculate_hit_miss_ratio(info['sunken_count'], info['miss_count'])
        self.calculate_overall_reward(reward)

    '''
    Method for setting a the number of rounds to result.
    rounds: Number of rounds used to finish the game
    '''
    def set_rounds(self, rounds):
        self.rounds = rounds

    '''
    Method for calculating the hit/miss ratio.
    hit: Amount of hits
    miss: Amount of misses
    '''
    def calculate_hit_miss_ratio(self, hit, miss):
        if miss is not 0:
            self.hit_miss_ratio = hit / miss

    '''
    Method for calculating the overall reward.
    reward: current reward of the reward
    '''
    def calculate_overall_reward(self, reward):
        self.overall_reward += reward

    '''
    Getter for the overall_reward of result object.
    '''
    def get_overall_reward(self):
        return self.overall_reward
