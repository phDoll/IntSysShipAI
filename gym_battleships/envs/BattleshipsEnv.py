from copy import deepcopy
from random import randint, getrandbits

import gym
from gym import spaces
import numpy as np
from .Ship import Ship

"""
Class representing the Battleship gym environment
"""
class BattleshipsEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  """
  Constructor for the battleships OpenAi gym environment
  Arguments:
  config = Configuration Object for the battleships game.
  """
  def __init__(self, config):
    super(BattleshipsEnv, self).__init__()

    # Instanciate variables
    # Boolean if ships can touch each other
    self.gap = config.gap
    self.binary_reward = config.binary_reward

    self.static_placement = config.static_placement
    self.placement = None
    self.placement_ships = None

    # The player board "radar" where he registers his shots.
    self.radar = []

    # List containing all valid actions (action get removed after beeing used/shot once)
    self.available_actions = []

    # List containing the ships objects of the enemy
    self.enemyShips = []

    # The enemy board where the ships of the enemy are placed
    self.enemy_board = []

    """
    FieldEncoding to map the state to more human readable content.
    Water:0
    Hit: *
    Sunken: #
    Miss: 0
    """
    self.fieldEncoding = {'W': 0, 'X': 1, '#': 2, '0': -1}

    # Set the available ships to place
    self.ships = config.ships

    # Set the size of the board
    self.board_size = config.board_size

    self.steps = 0

    # Set up the game for a new round
    self.set_up()

    # Allocate possible actions (Each field can be shot => board_size*board_size)
    self.action_space = spaces.Discrete(self.board_size * self.board_size)

    # Allocate oberservations (Each field has a state and can be observed.
    # Possible states are Integer values of fieldEncoding => low=-1, high=2)
    self.observation_space = spaces.Box(low=-1, high=2, shape=(self.board_size, self.board_size),
                                        dtype=np.int)

  '''
  Step function for OpenAi gym.
  Gets called one per round in battleships.
  Arguments:
  action = Integer, the index of the next field to shoot.
  '''
  def step(self, action):
    # Map the action to the board and get x,y coordinates of the next field to shot
    x, y = np.unravel_index(action, (self.board_size, self.board_size))

    # initialize reward with 0
    reward = 0

    #Check if x, y allready have been shot.
    if (x, y) not in self.available_actions:

      double_shot_reward = -1000

      if self.binary_reward:
        double_shot_reward = -1

      return self.radar, double_shot_reward, True, {
        'miss_count': 0,
        'hit_count': 0,
        'empty_count': 0,
        'sunken_count': 0
      }

      # Add negative reward for shooting a forbidden field
      #reward -= 2 * self.board_size

      # Get a random index of available actions
      random_index = np.random.randint(0, len(self.available_actions))

      # Get x,y coordinates from a random valid available action
      #x, y = self.available_actions[random_index]

    # Shoot coordinates
    hit = self.shoot(x, y)

    self.steps += 1

    # Evaluate result of shot
    after_shot_state = self.radar
    miss_after_shot, hit_after_shot, empty_after_shot, sunken_prev_shot = self.count_states(after_shot_state)

    # Add count information to info, for debug and possible calculations of statistics
    info = {
      'miss_count': miss_after_shot,
      'hit_count': hit_after_shot,
      'empty_count': empty_after_shot,
      'sunken_count': sunken_prev_shot
    }

    # Check if game is done
    done = self.check_done()

    # Calculate reward
    reward += self.calculate_reward(hit, done, self.steps)

    if self.binary_reward:
      reward = 1

    return after_shot_state, reward, done, info

  # OpenAI gym reset method. Gets called to set up a new game.
  def reset(self):
    self.set_up()

    # Return the fresh board of the player
    return self.radar

  # OpenAi gym render method
  def render(self, mode='human'):
    for x in range(self.board_size):
      # Print horizontale line
      print("-" * (4 * self.board_size + 2))
      for y in range(self.board_size):
        # Get state of the current field
        current_state_value = self.radar[x, y]
        # Get fieldEncoding for the state of the current field
        current_state = list(self.fieldEncoding.keys())[list(self.fieldEncoding.values()).index(current_state_value)]
        # If state of current Field is Water, print empty space for better readability
        current_state = (current_state if current_state != "W" else " ")
        # Print vertical lines
        print(" | ", end="")
        # Print current state
        print(current_state, end="")
      print(" |")
    print("-" * (4 * self.board_size + 2))

  # OpenAi gym close method
  def close (self):
    print('close')

  # Method to place ships on a given board
  def place_ships(self, board):
    # Initialize variables
    x = 0
    y = 0
    ships = []
    all_ships_placed = False

    # Try to place ships until all ships could be placed
    while not all_ships_placed:
      # Reset placed ships list
      ships = []
      # Reset/Clean the board
      for x in range(self.board_size):
        for y in range(self.board_size):
          board[x, y] = 0
      # No ships have been placed
      ships_placed = 0
      reset = False

      # Iterate over all ships to place
      for ship_length in self.ships:
        # Negative assumption, that the space is occupied
        occupied = True
        trys = 0

        # Try to find a spot to place the ship
        while occupied and not reset:
          # Get new random coordinates on the board
          x, y = self.get_random_coordinates()
          # Get a new random alignment of the ship
          is_vertical = bool(getrandbits(1))
          #Check if the space is allready occupied
          occupied = self.check_occupation(x, y, board, ship_length, is_vertical)
          # Increment the trys for the current ship
          trys += 1
          # If no valid spot can be found, the placement of all ships must be reset
          if trys > 1000:
            reset = True
        # If a valid spot is found:
        if not reset:
          # Create the ship and append it to the ships list
          ships.append(Ship(ship_length, x, y, is_vertical))
          # Increment the number of ships placed
          ships_placed += 1

          # Draw the placed ship onto the board
          if is_vertical:
            for i in range(ship_length):
              board[x + i, y] = 1
          else:
            for i in range(ship_length):
              board[x, y + i] = 1
      # Check if all ships where placed
      if ships_placed == len(self.ships):
        all_ships_placed = True

      if self.static_placement and self.placement is None:
        self.placement = np.copy(board)
        self.placement_ships = deepcopy(ships)

    return ships

  '''
  Method check if the space for a possible ship is occupied on a given boad.
  Arguments:
  x = X Coordinate to place the first part of the ship
  y = Y Coordinate to place the first part of the ship
  board = Board where to check if the ship can be placed on
  ship_length = Length of ship to place
  is_vertical = Boolean if the ship should be placed vertically or horizontally
  '''
  def check_occupation(self, x, y, board, ship_length, is_vertical):
    # Check if ships can touch.
    if self.gap:
      # Check for occupation with a gap
      occupied = self.check_occupation_with_gap(x, y, board, ship_length, is_vertical)
    else:
      # Positive assumption that the space is not occupied
      occupied = False

      # Determine which direction to check
      if is_vertical:
        # Iterate all field of a ship, beginning from the start position (x,y)
        # if the are allready occupied and inside the boundaries of the board
        for i in range(ship_length):
          if x + i >= self.board_size - 1:
            return True
          if board[x + i, y] == 1:
            occupied = True
      else:
        # Same as previous, but horizontally
        for i in range(ship_length):
          if y + i >= self.board_size - 1:
            return True
          if board[x, y + i] == 1:
            occupied = True

    return occupied

  '''
  Method for checking occupation with a gap.
  If a ship is placed vertical, it checks the top row for given coordinate.
  Arguments:
  x = X Coordinate to check on the board
  y = y Coordinate to check on the board
  board = Board to check for occupation
  '''
  def check_occupation_with_gap_top_row_is_vertical(self, x, y, board):
    # If x Coordinate is top row of board, no further checks are needed
    if x == 0:
      return True
    # Check y Coordinate is in bound and left upper field is occupied
    if y != 0 and board[x - 1, y - 1] == 1:
      return True
    # Check center upper field is occupied
    if board[x - 1, y] == 1:
      return True
    # Check y Coordinate is in bound and right upper field is occupied
    if y != self.board_size - 1 and board[x - 1, y + 1] == 1:
      return True

  '''
  Method for checking occupation with a gap.
  If a ship is placed vertical, it checks the left and right column of given coordinate.
  Arguments:
  x = X Coordinate to check on the board
  y = y Coordinate to check on the board
  board = Board to check for occupation
  '''
  def check_occupation_with_gap_left_right_is_vertical(self, x, y, board):
    # Check y Coordinate is in bound and left field is occupied
    if y != 0 and board[x, y - 1] == 1:
      return True
    # Check y Coordinate is in bound and right field is occupied
    if y != self.board_size - 1 and board[x, y + 1] == 1:
      return True

  '''
  Method for checking occupation with a gap.
  If a ship is placed vertical, it checks the bottom row of given coordinate.
  Arguments:
  x = X Coordinate to check on the board
  y = y Coordinate to check on the board
  board = Board to check for occupation
  '''
  def check_occupation_with_gap_bottom_row_is_vertical(self, x, y, board):
    # If x Coordinate is bottom row of board, no further checks are needed
    if x == self.board_size - 1:
      return True
    # Check y Coordinate is in bound and left lower field is occupied
    if y != 0 and board[x + 1, y - 1] == 1:
      return True
    # Check center lower field is occupied
    if board[x + 1, y] == 1:
      return True
    # Check y Coordinate is in bound and right lower field is occupied
    if y != self.board_size - 1 and board[x + 1, y + 1] == 1:
      return True

  '''
  Method for checking occupation with a gap.
  If a ship is placed horizontally, it checks the left column of given coordinate.
  Arguments:
  x = X Coordinate to check on the board
  y = y Coordinate to check on the board
  board = Board to check for occupation
  '''
  def check_occupation_with_gap_left_column(self, x, y, board):
    # If y Coordinate is left column of board, no further checks are needed
    if y == 0:
      return True
    # Check x Coordinate is in bound and left upper field is occupied
    if x != 0 and board[x - 1, y - 1] == 1:
      return True
    # Check center left field is occupied
    if board[x, y - 1] == 1:
      return True
    # Check x Coordinate is in bound and left lower field is occupied
    if x != self.board_size - 1 and board[x + 1, y - 1] == 1:
      return True

  '''
  Method for checking occupation with a gap.
  If a ship is placed horizontally, it checks the top and bottom row of given coordinate.
  Arguments:
  x = X Coordinate to check on the board
  y = y Coordinate to check on the board
  board = Board to check for occupation
  '''
  def check_occupation_with_gap_top_bottom(self, x, y, board):
    # Check x Coordinate is in bound and top field is occupied
    if x != 0 and board[x - 1, y] == 1:
      return True
    # Check x Coordinate is in bound and bottom field is occupied
    if x != self.board_size - 1 and board[x + 1, y] == 1:
      return True

  '''
  Method for checking occupation with a gap.
  If a ship is placed horizontally, it checks the right column of given coordinate.
  Arguments:
  x = X Coordinate to check on the board
  y = y Coordinate to check on the board
  board = Board to check for occupation
  '''
  def check_occupation_with_gap_right_column(self, x, y, board):
    # If y Coordinate is right column of board, no further checks are needed
    if y == self.board_size - 1:
      return True
    # Check x Coordinate is in bound and right upper field is occupied
    if x != 0 and board[x - 1, y + 1] == 1:
      return True
    # Check center right field is occupied
    if board[x, y + 1] == 1:
      return True
    # Check x Coordinate is in bound and right lower field is occupied
    if x != self.board_size - 1 and board[x + 1, y + 1] == 1:
      return True

  '''
  Method for checking occupation with a gap.
  Check whether ship is placed vertically or horizontally. 
  Makes additional checks if coordinates are occupied.
  x = X Coordinate start coordinate to place the ship
  y = y Coordinate start coordinate to place the ship
  board = Board to place the ship on 
  ship_length = Length of the ship to place
  is_vertical = Boolean if the ship should be placed vertically or horizontally
  '''
  def check_occupation_with_gap(self, x, y, board, ship_length, is_vertical):
    # Possitive assumption the field is never occupied
    occupied = False
    # Determine ship placement direction
    if is_vertical:
      # Iterate all field of a ship, beginning from the start position (x,y)
      for i in range(ship_length):
        current_field_x = x + i
        # Check field is inside the boundaries of the board
        if current_field_x >= self.board_size or board[current_field_x, y] == 1:
          occupied = True
          break
        # Check first part of the ship
        if i == 0:
          # Check top row
          if self.check_occupation_with_gap_top_row_is_vertical(current_field_x, y, board):
            occupied = True
            break
          # Check left and right column
          if self.check_occupation_with_gap_left_right_is_vertical(current_field_x, y, board):
            occupied = True
            break
        # Check middle parts of the ship
        elif i < ship_length - 1:
          # Check left and right column
          if self.check_occupation_with_gap_left_right_is_vertical(current_field_x, y, board):
            occupied = True
            break
        # Check last part of the ship
        else:
          # Check bottom row
          if self.check_occupation_with_gap_bottom_row_is_vertical(current_field_x, y, board):
            occupied = True
            break
          # Check left and right column
          if self.check_occupation_with_gap_left_right_is_vertical(current_field_x, y, board):
            occupied = True
            break
    # Ship placement direction is horizontal
    else:
      # Iterate all field of a ship, beginning from the start position (x,y)
      for i in range(ship_length):
        current_field_y = y + i
        # Check field is inside the boundaries of the board
        if current_field_y >= self.board_size or board[x, current_field_y] == 1:
          occupied = True
          break
        # Check first part of the ship
        if i == 0:
          # Check left column
          if self.check_occupation_with_gap_left_column(x, current_field_y, board):
            occupied = True
            break
          # Check top and bottom row
          if self.check_occupation_with_gap_top_bottom(x, current_field_y, board):
            occupied = True
            break
        # Check middle parts of the ship
        elif i < ship_length - 1:
          # Check top and bottom row
          if self.check_occupation_with_gap_top_bottom(x, current_field_y, board):
            occupied = True
            break
        # Check last part of ship
        else:
          # Check right column
          if self.check_occupation_with_gap_right_column(x, current_field_y, board):
            occupied = True
            break
          # Check top and bottom row
          if self.check_occupation_with_gap_top_bottom(x, current_field_y, board):
            occupied = True
            break

    return occupied

  '''
  Method for creating a pair of random coordinates.
  return: Pair of random coordinates
  '''
  def get_random_coordinates(self):
      # random number in range 0 - board_size - 1
      x = randint(0, self.board_size - 1)
      y = randint(0, self.board_size - 1)
      return x, y

  '''
  Method for counting all states currently present on the radar board
  state: Numpy Array of all states present on the radar board
  '''
  def count_states(self, state):
    # Counts how many times a unique element is present on the radar board
    # return the unique element and how many times the unique element is present
    uni_states, counts = np.unique(state.ravel(), return_counts=True)
    hit = counts[uni_states == self.fieldEncoding['X']]
    miss = counts[uni_states == self.fieldEncoding['0']]
    empty = counts[uni_states == self.fieldEncoding['W']]
    sunken = counts[uni_states == self.fieldEncoding['#']]
    # Checks whether hits present and assigns present hits
    if len(hit) == 0:
      hit = 0
    else:
      hit = hit[0]
    # Checks whether misses present and assigns present misses
    if len(miss) == 0:
      miss = 0
    else:
      miss = miss[0]
    # Checks whether empty fields present and assigns present empty fields
    if len(empty) == 0:
      empty = 0
    else:
      empty = empty[0]
    # Checks whether sunken ships present and assigns present sunken ships
    if len(sunken) == 0:
      sunken = 0
    else:
      sunken = sunken[0]

    return miss, hit, empty, sunken

  '''
  Method for shooting on the enemy board.
  Displays the result on radar board
  x: X Coordinate to shoot
  y: Y Coordinate to shoot
  '''
  def shoot(self, x, y):
    # Negative assumption shoot misses a ship
    hit = False
    # Radar board field is set to miss
    self.radar[x, y] = self.fieldEncoding['0']
    # Iterate enemyShips to check for hits
    for ship in self.enemyShips:
      # Check whether shoot is a hit
      if ship.is_hit(x, y):
        # Set radar board field to hit
        self.radar[x, y] = self.fieldEncoding['X']
        hit = True
        # Check whether the ship is sunken
        if ship.sunken():
          # Set radar board ship fields to sunken
          self.draw_sunken(ship, self.radar)

    # Remove shoot Coordinate from list of available actions
    self.available_actions.remove((x, y))
    return hit

  '''
  Method for displaying a sunken ship on radar board.
  ship: Sunken ship
  board: Radar Board 
  '''
  def draw_sunken(self, ship, board):
    # Get start coordinates of the sunken ship
    x = ship.get_x()
    y = ship.get_y()
    # Determine direction of the sunken ship
    if ship.is_vertical:
      # Iterate the sunken ship and update radar board fields to sunken
      for i in range(ship.get_length()):
        board[x + i, y] = self.fieldEncoding['#']
    # sunken ship is placed horizontally
    else:
      # Iterate the sunken ship and update radar board fields to sunken
      for i in range(ship.get_length()):
        board[x, y + i] = self.fieldEncoding['#']

  '''
  Method for checking if the Game is Done.
  All enemy ships must be sunken. 
  '''
  def check_done(self):
    # Possitive assumption the game is allways done
    if self.binary_reward:
      done = False
    else:
      done = True
    #done = False
    # Iterate all enemy ships
    # Check if one of the enemy ships is not yet sunken
    if self.binary_reward:
      if not self.available_actions:
        done = True
    else:
      for ship in self.enemyShips:
        if not ship.sunken():
          done = False
    return done

  '''
  Method for calculating the reward.
  This method must be updated for reward based learning. Currently uses dummy values
  reward: Current Reward of the Agent
  hit: Boolean last shoot was hit or miss 
  done: Boolean game is finished
  '''
  def calculate_reward(self, hit, done, steps):
    # Agent gets a reward for hitting a ship
    reward = 0
    if hit:
      reward += 20
    # Agent gets more reward if he finishes the game
    if done:
     reward += 100 * ((self.board_size*self.board_size) / self.steps)
    # Float for later calculations
    return int(round(reward))

  '''
  Method to setup the game environment.
  '''
  def set_up(self):
    # Inits radar board with Water fields
    self.radar = self.fieldEncoding['W'] * np.ones((self.board_size, self.board_size), dtype='int')
    # Inits enemy board with zeros representing water
    self.enemy_board = 0 * np.ones((self.board_size, self.board_size), dtype='int')
    # resets available_actions
    self.available_actions = []
    # Init available_actions for all fields of the board
    for x in range(self.board_size):
      for y in range(self.board_size):
        self.available_actions.append((x, y))
    # places enemy ships on the enemy board
    if self.placement_ships and self.placement is not None:
      self.enemyShips = deepcopy(self.placement_ships)
      self.enemy_board = np.copy(self.placement)
    else:
      self.enemyShips = self.place_ships(self.enemy_board)

    self.steps = 0

  '''
  Method calculates the maximum mean reward threshold for the callback in training. 
  '''
  def calculate_threshold(self):
    if self.binary_reward:
      return self.board_size * self.board_size
    else:
      ship_fields = 0
      reward = 0
      for ship_length in self.ships:
        ship_fields += ship_length
      reward += ship_fields * 20
      reward += 100 * ((self.board_size*self.board_size) / ship_fields)
      return int(round(reward))
