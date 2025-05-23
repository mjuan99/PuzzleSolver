from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import numpy as np

from src.models.model_interface import Model

class RubiksCubeMLPModel(Model):

  def __init__(self):
    self.color_map = {
      'Y': 0,
      'R': 1,
      'B': 2,
      'W': 3,
      'O': 4,
      'G': 5
    }

  # one-hot encodes a rubkis cube state
  def encode_state(self, state):
    one_hot = np.zeros((54, 6), dtype=np.float32)
    for i, sticker in enumerate(state):
      one_hot[i, self.color_map[sticker]] = 1.0
    return one_hot.flatten()

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
    states = RubiksCubeMLPModel.upsample_levels(states)
    self.X = []
    self.y = []
    for state, level in states:
      self.X.append(self.encode_state(state))
      self.y.append(level)

    X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, stratify=self.y, test_size=0.3)
    X_val, X_test, y_val, self.y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5)

    # Create datasets
    train_dataset = PuzzleDataset(X_train, y_train)
    val_dataset = PuzzleDataset(X_val, y_val)
    self.test_dataset = PuzzleDataset(X_test, self.y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    self.model = HeuristicMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    num_epochs = 7

    for epoch in range(num_epochs):
        self.model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

  def evaluate_model(self):
    self.model.eval()
    y_preds = []

    with torch.no_grad():
        for X_batch, _ in DataLoader(self.test_dataset, batch_size=256):
            preds = self.model(X_batch).squeeze().cpu().numpy()
            y_preds.extend(preds)

    y_preds = np.array(y_preds)
    y_true = self.y_test  # already a NumPy array

    mae = mean_absolute_error(y_true, y_preds)
    rmse = np.sqrt(mean_squared_error(y_true, y_preds))

    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_preds, alpha=0.2)
    plt.plot([0, 20], [0, 20], '--', color='gray')
    plt.xlabel("True Distance")
    plt.ylabel("Predicted Distance")
    plt.title("Predicted vs True Distances")
    plt.grid(True)
    plt.show()

  def predict(self, state):
    encoded = self.encode_state(state)
    tensor_input = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(torch.device("cpu"))
    with torch.no_grad():
        prediction = self.model(tensor_input).item()
    return prediction


class HeuristicMLP(nn.Module):
    def __init__(self):
        super(HeuristicMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(54 * 6, 512),  # input layer
            nn.ReLU(),
            nn.Linear(512, 1024),     # first hidden layer
            nn.ReLU(),
            nn.Linear(1024, 512),     # first hidden layer
            nn.ReLU(),
            nn.Linear(512, 256),     # first hidden layer
            nn.ReLU(),
            nn.Linear(256, 128),     # second hidden layer
            nn.ReLU(),
            nn.Linear(128, 1)        # output: predicted distance
        )

    def forward(self, x):
        return self.model(x)

class PuzzleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]