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
    """
    def __init__(self, board_size, ships, gap, static_placement):
        self.board_size = board_size
        self.ships = ships
        self.gap = gap
        self.static_placement = static_placement
