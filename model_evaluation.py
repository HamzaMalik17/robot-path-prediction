# %% [markdown]
# # **Part 3A: Neural Network [40 marks]**

# %% [markdown]
# In this part, you will implement a neural network and test its path prediction performance on the same dataset using PyTorch.

# %% [markdown]
# ### Imports

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ### Processing Dataset
#
# For this task, we consider three different lookback scenarios:
# - Predict the next 3 seconds (90 points) based on the previous 3 seconds (90 points)
# - Predict the next 3 seconds (90 points) based on the previous 6 seconds (180 points)
# - Predict the next 3 seconds (90 points) based on the previous 9 seconds (270 points)

# %%
def process_data(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        x_str, y_str = line.strip().split(',')
        data.append([int(x_str), int(y_str)])
    return np.array(data, dtype=np.float32)


def prepare_sequences(data, lookback, horizon=90):
    """
    Build (X, y) pairs using a sliding window.
    X shape: (lookback * 2,)  — flattened previous (x,y) positions
    y shape: (horizon * 2,)   — flattened next (x,y) positions to predict
    """
    X, y = [], []
    for i in range(lookback, len(data) - horizon):
        X.append(data[i - lookback:i].flatten())
        y.append(data[i:i + horizon].flatten())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


data_train = process_data('./Dataset/Training/training_data.txt')

# %% [markdown]
# ### Model Architecture

# %%
class NeuralNet(nn.Module):
    def __init__(self, input_shape, output_shape, hidden=256):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# %% [markdown]
# ### Training
#
# We use Adam optimizer and RMSE as the evaluation metric.
# Each model takes a lookback window of x,y pairs as input and predicts the next 90 positions (3 seconds).

# %%
HORIZON     = 90    # always predict 3 seconds = 90 frames ahead
EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 1e-3


def train_model(lookback, train_data):
    input_size  = lookback * 2
    output_size = HORIZON  * 2

    X_train, y_train = prepare_sequences(train_data, lookback, HORIZON)
    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train)

    model     = NeuralNet(input_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    rmse_per_epoch = []
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Compute epoch RMSE on full training set
        model.eval()
        with torch.no_grad():
            epoch_rmse = np.sqrt(criterion(model(X_t), y_t).item())
        rmse_per_epoch.append(epoch_rmse)
        print(f'Lookback {lookback} | Epoch {epoch+1}/{EPOCHS} | RMSE: {epoch_rmse:.4f}')

    return model, rmse_per_epoch


model_90,  rmse_90  = train_model(lookback=90,  train_data=data_train)
model_180, rmse_180 = train_model(lookback=180, train_data=data_train)
model_270, rmse_270 = train_model(lookback=270, train_data=data_train)


# %% [markdown]
# ### Evaluation and Analysis
#
# Plot RMSE vs epoch for all three lookback sizes and explain the trend.

# %%
plt.figure(figsize=(10, 6))
plt.plot(rmse_90,  label='Lookback 3s (90 pts)')
plt.plot(rmse_180, label='Lookback 6s (180 pts)')
plt.plot(rmse_270, label='Lookback 9s (270 pts)')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE vs. Epoch for Different Lookback Sizes')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ### Double click to $\color{green}{\text{add explanation/reasoning here}}$
#
# As lookback increases, RMSE decreases more rapidly and converges to a lower value. With a larger lookback (9 seconds vs 3 seconds), the network receives richer temporal context — it can learn longer-range movement patterns such as turns and periodic behaviors. The 270-point lookback model consistently outperforms the 90-point model because predicting 90 future positions is easier when the model has seen more of the robot's recent trajectory.


# %% [markdown]
# ### Visualization of Actual and Predicted Path
#
# Use model_270 (best lookback). Feed 9 seconds of test data as input and predict the following 3 seconds.

# %%
data_test = process_data('./Dataset/Testing/test01.txt')

LOOKBACK = 270
start    = LOOKBACK          # first valid prediction start (frame 270)
end      = start + HORIZON   # frame 360

actual_window = data_test[start:end]   # ground-truth next 90 frames

input_seq = torch.tensor(
    data_test[start - LOOKBACK:start].flatten(),
    dtype=torch.float32
).unsqueeze(0)

model_270.eval()
with torch.no_grad():
    pred_flat  = model_270(input_seq).squeeze().numpy()
pred_coords = pred_flat.reshape(HORIZON, 2)

plt.figure(figsize=(8, 6))
plt.plot(actual_window[:, 0], actual_window[:, 1],
         label='Actual', linewidth=2, color='blue')
plt.plot(pred_coords[:, 0],   pred_coords[:, 1],
         label='Predicted', linewidth=2, linestyle='--', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Actual vs. Predicted Robot Path (lookback=9s, model_270)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ### Discussion
#
# How does changing the number of layers and neurons affect the error?
# We test four architecture configurations below.

# %%
configs = [
    {'hidden': 64,  'layers': 1, 'label': '1 layer,  64 neurons'},
    {'hidden': 128, 'layers': 2, 'label': '2 layers, 128 neurons'},
    {'hidden': 256, 'layers': 2, 'label': '2 layers, 256 neurons'},
    {'hidden': 256, 'layers': 3, 'label': '3 layers, 256 neurons'},
]

LOOKBACK_ARCH = 180   # fix lookback for architecture comparison
X_arch, y_arch = prepare_sequences(data_train, LOOKBACK_ARCH, HORIZON)
X_at = torch.tensor(X_arch)
y_at = torch.tensor(y_arch)


class FlexNet(nn.Module):
    def __init__(self, input_shape, output_shape, hidden, num_layers):
        super().__init__()
        layers = [nn.Linear(input_shape, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, output_shape))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


criterion = nn.MSELoss()
arch_results = {}

for cfg in configs:
    net = FlexNet(LOOKBACK_ARCH * 2, HORIZON * 2, cfg['hidden'], cfg['layers'])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ds  = torch.utils.data.TensorDataset(X_at, y_at)
    dl  = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    rmses = []
    for epoch in range(EPOCHS):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            criterion(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            rmses.append(np.sqrt(criterion(net(X_at), y_at).item()))
    arch_results[cfg['label']] = rmses
    print(f"{cfg['label']} | Final RMSE: {rmses[-1]:.4f}")

plt.figure(figsize=(10, 6))
for label, rmses in arch_results.items():
    plt.plot(rmses, label=label)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Architecture Comparison: Layers & Neurons vs RMSE')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ### Double click to $\color{green}{\text{add explanation/reasoning here}}$
#
# A single layer with 64 neurons is too small to capture the non-linear temporal patterns, resulting in the highest final RMSE (underfitting). Increasing to 2 layers with 128 neurons improves performance significantly. Moving to 256 neurons per layer gives a further drop in RMSE. Adding a third 256-neuron layer provides diminishing returns and slightly risks overfitting, though with only 50 epochs the effect is minor. The 2-layer 256-neuron configuration offers the best balance of capacity and training stability.


# %% [markdown]
# # **Part 3B (Bonus Part): Mapping the Predicted Path onto the Video [5 marks]**
#
# Overlay the actual and predicted paths frame-by-frame onto the test video using OpenCV.
# Blue = actual path, Red = predicted path.

# %%
import cv2
import os

video_path = './Dataset/Testing/test01.mp4'

if not os.path.exists(video_path):
    print("test01.mp4 not found in Dataset/Testing/. Skipping video overlay.")
else:
    FPS          = 30
    START_FRAME  = LOOKBACK           # 270 — matches the prediction window above
    HORIZON_DRAW = HORIZON            # 90 frames = 3 seconds

    cap    = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter('predicted_path_overlay.mp4', fourcc, FPS, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    for frame_idx in range(HORIZON_DRAW):
        ret, frame = cap.read()
        if not ret:
            break

        # Draw path trails up to current frame
        for j in range(frame_idx):
            pt1_a = (int(actual_window[j, 0]),  int(actual_window[j, 1]))
            pt2_a = (int(actual_window[j+1, 0]),int(actual_window[j+1, 1]))
            cv2.line(frame, pt1_a, pt2_a, (255, 0, 0), 2)   # blue — actual

            pt1_p = (int(pred_coords[j, 0]),  int(pred_coords[j, 1]))
            pt2_p = (int(pred_coords[j+1, 0]),int(pred_coords[j+1, 1]))
            cv2.line(frame, pt1_p, pt2_p, (0, 0, 255), 2)   # red — predicted

        # Current position dots
        cv2.circle(frame,
                   (int(actual_window[frame_idx, 0]), int(actual_window[frame_idx, 1])),
                   6, (255, 0, 0), -1)
        cv2.circle(frame,
                   (int(pred_coords[frame_idx, 0]), int(pred_coords[frame_idx, 1])),
                   6, (0, 0, 255), -1)

        # Legend
        cv2.putText(frame, 'Actual',    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, 'Predicted', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print("Saved: predicted_path_overlay.mp4")
