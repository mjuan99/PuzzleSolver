import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.models.model_interface import Model  # Assuming Model is defined in model_interface.py
import matplotlib.pyplot as plt
import random
from collections import defaultdict


class RubiksCubeCNNModel(Model):
    def __init__(self, device=None):
        self.color_map = {'W': 0, 'R': 1, 'B': 2, 'O': 3, 'G': 4, 'Y': 5}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CubeCNNNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.X_test, self.y_test = None, None

    #def encode_state(self, state):
    #    flat = np.array([self.color_map[color] for color in state])
    #    return flat.reshape(6, 3, 3)


    
    # one-hot encodes a rubkis cube state
    def encode_state(self, state):
        encoded = [self.color_map[color] for color in state]  # shape: (54,)
        encoded = np.array(encoded).reshape(6, 3, 3)         # shape: (6, 3, 3)
        one_hot = np.eye(6)[encoded]                         # shape: (6, 3, 3, 6)
        one_hot = one_hot.transpose(3, 0, 1, 2)              # shape: (6, 6, 3, 3)
        return one_hot

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
        states = RubiksCubeCNNModel.upsample_levels(states)
        X = np.stack([self.encode_state(state) for state, _ in states])
        X = np.array([self.encode_state(state) for state, _ in states], dtype=np.float32)
        y = np.array([level for _, level in states], dtype=np.float32)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.3)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5)

        self.X_test = torch.tensor(X_test).to(self.device)
        self.y_test = torch.tensor(y_test).to(self.device)

        train_data = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

        self.model.train()
        for epoch in range(10):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_test)
            y_true = self.y_test.cpu().numpy()
            y_preds = preds.cpu().numpy()
            mae = mean_absolute_error(y_true, y_preds)
            rmse = np.sqrt(mean_squared_error(y_true, y_preds))
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
        self.model.eval()
        encoded = self.encode_state(state)
        tensor_input = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor_input)
        return pred.item()


class CubeCNNNet(nn.Module):
    def __init__(self):
        super(CubeCNNNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)