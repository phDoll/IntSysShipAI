"""
Class representing a ship object
"""
class Ship:
    # A ship has zero hits initially
    hits = 0

    """
    Constructor for a Ship object
    Arguments:
    length = Size of the Ship.
    x = Starting X coordinate
    y = Starting Y coordinate
    is_vertical = Boolean whether ship direction is vertically or horizontally
    """
    def __init__(self, length, x, y, is_vertical):
        self.length = length
        self.x = x
        self.y = y
        self.is_vertical = is_vertical

    '''
    Method for counting hits.
    '''
    def hit(self):
        self.hits += 1

    '''
    Method for checking if the ship is sunken.
    '''
    def sunken(self):
        return self.hits == self.length

    '''
    Getter for length of a ship.
    '''
    def get_length(self):
        return self.length

    '''
    Getter for starting x coordinate of a ship.
    '''
    def get_x(self):
        return self.x

    '''
    Getter for starting y coordinate of a ship.
    '''
    def get_y(self):
        return self.y

    '''
    Getter for is_vertical 
    '''
    def get_is_vertical(self):
        return self.is_vertical

    '''
    Method for checking shoot hitting a part of the ship.
    x_hit: x Coordinate of the shoot
    y_hit: y Coordinate of the shoot
    '''
    def is_hit(self, x_hit, y_hit):
        # Iterate all parts of the ship
        for i in range(self.length):
            # Determine placement direction of the ship
            if self.is_vertical:
                # Check if ship part is hit
                if self.x + i == x_hit and self.y == y_hit:
                    # Update hit counter
                    self.hit()
                    return True
            # Ship is placed horizontally
            else:
                # Check if ship part is hit
                if self.x == x_hit and self.y + i == y_hit:
                    # Update hit counter
                    self.hit()
                    return True
        return False
