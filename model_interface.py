from abc import ABC, abstractmethod

class Model(ABC):

  @abstractmethod
  def train_model(self, states):
    pass

  @abstractmethod
  def evaluate_model(self):
    pass

  @abstractmethod
  def predict(self, state):
    pass