from abc import ABC, abstractmethod

class Puzzle(ABC):

  @abstractmethod
  def new_puzzle(self, movements=0):
    pass

  @abstractmethod
  def apply_movement(self, state, movement):
    pass

  @abstractmethod
  def get_movements(self):
    pass

  @abstractmethod
  def is_valid_movement(self, state, movement):
    pass

  @abstractmethod
  def is_redundant(self, prev_movements, movement):
    pass

  @abstractmethod
  def to_string(self, state):
    pass

  @abstractmethod
  def from_string(self, string):
    pass