
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.model_interface import Model
import numpy as np

class RubiksCubeXGBModel(Model):

  def __init__(self):
    self.model = XGBRegressor(n_estimators=100, max_depth=40, learning_rate=0.05, verbosity=0)
    self.color_map = {
      'Y': 0,
      'R': 1,
      'B': 2,
      'W': 3,
      'O': 4,
      'G': 5
    }

  # label encodes a rubkis cube state
  def encode_state(self, state):
    return [self.color_map[color] for color in state]

  # Upsamples to balance the training data, as there are fewer states closer to the final state
  @staticmethod
  def upsample_levels(states):
      # Step 1: Group states by level
      level_to_states = defaultdict(list)
      for state, level in states:
          level_to_states[level].append((state, level))

      # Step 2: Find max level count
      max_count = max(len(samples) for samples in level_to_states.values())

      # Step 3: Upsample each level to match max count
      balanced_states = []
      for level, samples in level_to_states.items():
          if len(samples) < max_count:
              # Sample with replacement to match count
              needed = max_count - len(samples)
              samples_to_add = random.choices(samples, k=needed)
              samples.extend(samples_to_add)

          balanced_states.extend(samples)

      # Step 4: Shuffle to avoid blocks of same-level states
      random.shuffle(balanced_states)

      return balanced_states

  def train_model(self, states):
    states = RubiksCubeXGBModel.upsample_levels(states)
    X = [self.encode_state(state) for state, _ in states]
    y = [level for _, level in states]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5)

    self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
    self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

    self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

  def evaluate_model(self):
    y_preds = self.model.predict(self.X_test)
    mae = mean_absolute_error(self.y_test, y_preds)
    rmse = np.sqrt(mean_squared_error(self.y_test, y_preds))

    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(self.y_test, y_preds, alpha=0.2)
    plt.plot([0, 20], [0, 20], '--', color='gray')
    plt.xlabel("True Distance")
    plt.ylabel("Predicted Distance")
    plt.title("Predicted vs True Distances")
    plt.grid(True)
    plt.show()

  def predict(self, state):
    return float(self.model.predict([self.encode_state(state)])[0])


