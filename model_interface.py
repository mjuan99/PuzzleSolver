from abc import ABC, abstractmethod

# Model interface definition
class Model(ABC):

  # Trains a model to predict the distance of a given state to the final state
  # <states> is a list of (state, distance) pairs
  @abstractmethod
  def train_model(self, states):
    pass

  # Prints model evaluation metrics
  @abstractmethod
  def evaluate_model(self):
    pass

  # Returns the predicted distance from <state> to the final state
  @abstractmethod
  def predict(self, state):
    pass