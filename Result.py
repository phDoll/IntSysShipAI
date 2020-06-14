class Result:
    def __init__(self, history=[], reward=0, rounds=0):
        self.history = history
        self.overall_reward = reward
        self.rounds = rounds
        self.hit_miss_ratio = 0

    def append_history(self, round, action, observation, reward, done, info):
        self.history.append((round, action, observation, reward, done, info))
        self.calculate_hit_miss_ratio(info['sunken_count'], info['miss_count'])
        self.calculate_overall_reward(reward)

    def set_reward(self, reward):
        self.reward = reward

    def set_rounds(self, rounds):
        self.rounds = rounds

    def calculate_hit_miss_ratio(self, hit, miss):
        if miss is not 0:
            self.hit_miss_ratio = hit / miss

    def calculate_overall_reward(self, reward):
        self.overall_reward += reward

    def get_overall_reward(self):
        return self.overall_reward
