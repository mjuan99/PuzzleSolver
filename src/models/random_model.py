from models.model_interface import Model
import random

class RandomModel(Model):
  def __init__(self, max_distance):
    self.max_distance = max_distance

  def train_model(self, states):
    pass

  def evaluate_model(self):
    pass

  def predict(self, state):
    return random.randint(0, self.max_distance)