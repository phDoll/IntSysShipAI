"""
Class representing a config object
"""
class Config:
    """
    Constructor for a Config object
    Arguments:
    board_size = Number of fields in x and y direction.
    ships = Array of Ships with length of ships. Example [2,2,3] Sets 2 Ships with length of 2, 1 ship with length 3
    gap = Boolean: True => Ships have a gap of at least 1 water field between each other.
    False => Ships can be placed side by side
    static_placement => A static placement for ships over all iterations is used
    binary_reward => The reward will be +1 for a valid action and -1 for an invalid action
    to train an agent not to choose the same action multiple times.
    """
    def __init__(self, board_size, ships, gap, static_placement, binary_reward):
        self.board_size = board_size
        self.ships = ships
        self.gap = gap
        self.static_placement = static_placement
        self.binary_reward = binary_reward
