from abc import ABC, abstractmethod

# Puzzle interface definition
class Puzzle(ABC):

  # Creates a new puzzle state <movements> movements away from the final state
  @abstractmethod
  def new_puzzle(self, movements=0):
    pass

  # Returns a new state as the result of applying <movement> to <state>
  @abstractmethod
  def apply_movement(self, state, movement):
    pass

  # Returns a list of possible movements
  @abstractmethod
  def get_movements(self):
    pass

  # Returns True if <movement> can be applied to <state>
  @abstractmethod
  def is_valid_movement(self, state, movement):
    pass

  # Returns True if <movement> is redundant given the movements in <prev_movements>
  # A movement is redundant if applying it undoes a previous movement
  @abstractmethod
  def is_redundant(self, prev_movements, movement):
    pass

  # Prints the <state> in a human readable format
  @abstractmethod
  def to_string(self, state):
    pass

  # Creates a new state from a string encoded state (not necessarily the same format as to_string())
  @abstractmethod
  def from_string(self, string):
    pass