from src.models.model_interface import Model
import numpy as np

class RandomNormalModel(Model):
  def __init__(self, max_distance):
    self.generator = RandomNormalModel._normal_generator(0, max_distance, max_distance / 2, max_distance / 5)

  def train_model(self, states):
    pass

  def evaluate_model(self):
    pass

  def predict(self, state):
    return next(self.generator)
  
  @staticmethod
  def _normal_generator(min, max, mu, sigma):
    rng = np.random.default_rng()
    while True:
      val = rng.normal(mu, sigma)
      yield np.round(np.clip(val, min, max)).astype(int)