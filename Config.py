BOARD_SIZE = 10
WATER = "O"
MISS = "X"
HIT = "*"
SUNKEN = "#"
SHIP = '+'


class Config:
    def __init__(self, board_size, ships, gap):
        self.board_size = board_size
        self.ships = ships
        self.gap = gap
