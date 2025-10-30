import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

import os

from pathlib import Path
import re, numpy as np, pandas as pd

def load_dataset_simple(dataset_fp, channel="T7", fs=32, clip_sec=2):
    root = Path(dataset_fp)
    clip_len = fs * clip_sec
    X_list, y_list, g_list = [], [], []
    for sub_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        m = re.search(r"S(\d{2})", sub_dir.name)  # matches S01 or (S01)
        if not m: 
            continue
        sid = int(m.group(1))
        csv_dir = sub_dir / "Preprocessed EEG Data" / ".csv format"
        for g in (1, 2, 3, 4):
            f = csv_dir / f"S{sid:02d}G{g}AllChannels.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
            x = df[channel].to_numpy()
            n = len(x) // clip_len
            if n == 0:
                continue
            clips = x[: n*clip_len].reshape(n, clip_len).astype(np.float32)
            X_list.append(clips)
            y_list.append(np.full(n, g-1, dtype=int))   # labels 0..3
            g_list.append(np.full(n, sid, dtype=int))   # subject id
    X = np.expand_dims(np.vstack(X_list), 1)  # (N,1,T)
    y = np.concatenate(y_list)
    groups = np.concatenate(g_list)
    return X, y, groups


# Constants
SAMPLE_RATE = 32 # (Hz)
GAMES = ["boring", "calm", "horror", "funny"]


dataset_fp = '/home/fabric/.cache/kagglehub/datasets/wajahat1064/emotion-recognition-using-eeg-and-computer-games/versions/2/Dataset - Emotion Recognition data Based on EEG Signals and Computer Games/Database for Emotion Recognition System Based on EEG Signals and Various Computer Games - GAMEEMO/GAMEEMO'

# Read the data
X, y, groups = load_dataset_simple(dataset_fp)

# data = []
# for game_id, game in enumerate(GAMES):
#     for sub in dataset_fp:
#         if sub[:1] == 'S':
#             # Ex: '(S01)/Preprocessed EEG Data/.csv format/S01G1AllChannels.csv'
#             csv_fp = f'{sub}/Preprocessed EEG Data/.csv format/'
#             game_data = pd.read_csv(os.path.join(csv_fp, f"{sub}G{game_id + 1}AllChannels.csv"))
#             game_data["game"] = game
#             data.append(game_data)

# data = pd.concat(data, axis = 0, ignore_index = True)
# data.head()


# TODO: choose one of the frontal (F3 / F4 / F7 / F8 / FC5 / FC6) or temporal (T7 / T8) electrodes and ensure the signal is clean
electrode = "T7"

fig, ax = plt.subplots(1, 1)
for game in GAMES:
    ax.plot(data[data["game"] == game][electrode], label = game)
ax.set_xlabel("Time (seconds)")
ax.set_xticks(range(0, len(data), SAMPLE_RATE * 60 * 10))
ax.set_ylabel("mV")

ax.legend()


data = data[[electrode, "game"]]
data.head()



# TODO: adjust if needed
clip_length = 2 # (seconds)

# Split into clips
clipped_data = []
y = []
for game_id, game in enumerate(GAMES):
    clips = np.array_split(
        data[data['game'] == game][electrode].to_numpy(), 
        len(data[data['game'] == game]) // (clip_length * SAMPLE_RATE))
    clipped_data.extend(clips)
    y.extend([game_id] * len(clips))

# Remove edge effects
min_length = np.min([len(arr) for arr in clipped_data])
X = []
for array in clipped_data:
    X.append(array[:min_length])

X = np.vstack(X, dtype = float)
y = np.array(y, dtype = int)

print(X.shape)
print(y.shape)



np.random.seed(123)

# Add an additional axis required by torch's Conv layers
X = np.expand_dims(X, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Convert to torch tensors
X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

print(X_train.shape)



class LFPDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y.long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# Batch generators

# TODO: adjust if needed
batch_size = 32

train_batch_generator = torch.utils.data.DataLoader(LFPDataset(X_train, y_train), batch_size = batch_size,
                                                    shuffle = True)

test_batch_generator = torch.utils.data.DataLoader(LFPDataset(X_test, y_test), batch_size = batch_size,
                                                    shuffle = False)



# TODO: adjust if needed
model = torch.nn.Sequential(
    torch.nn.Conv1d(1, 1, kernel_size = 4, padding = "same"),
    torch.nn.ReLU(),
    torch.nn.Conv1d(1, 1, kernel_size = 4, padding = "same"),
    # torch.nn.Conv1d(1, 1, kernel_size = 4, padding = "same"),       # Add as one part of analysis for time/accuracy
    torch.nn.Flatten(),
    torch.nn.Linear(64, 4),                                           # Change 64 as other part of analysis (ie, 16,32,128)
    torch.nn.LogSoftmax(dim = 1)
)




def train(n_epoch, model):
    # TODO: adjust learning rate if needed
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    

    for e in range(n_epoch):
        model.train(True)

        train_loss = []
        train_acc = []
        for X_batch, y_batch in train_batch_generator:
            model.zero_grad()
            logits = model(X_batch).squeeze()
            loss = torch.nn.functional.nll_loss(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().numpy())
            
            prediction = torch.softmax(logits, dim = 1).detach().numpy()
            prediction = np.argmax(prediction, axis = 1)
            train_acc.append(accuracy_score(y_batch.detach().numpy(), prediction))

        model.train(False)
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for X_batch, y_batch in test_batch_generator:
                logits = model(X_batch).squeeze()
                loss = torch.nn.functional.nll_loss(logits, y_batch)
                test_loss.append(loss.detach().numpy())

                prediction = torch.softmax(logits, dim = 1).detach().numpy()
                prediction = np.argmax(prediction, axis = 1)
                test_acc.append(accuracy_score(y_batch.detach().numpy(), prediction))

        print(f"Epoch {e} : train_loss={np.mean(train_loss)}, train_acc={np.mean(train_acc)}, test_loss={np.mean(test_loss)}, test_acc={np.mean(test_acc)}")

    return model

# Record Time
import time
start_time = time.time()

train(n_epoch = 100, model = model)

end_time = time.time()
train_time = end_time - start_time

print(f'Training Time {train_time}')


a_clip = X[0]
plt.plot(a_clip.flatten())



prediction = model(torch.tensor(np.expand_dims(a_clip, 1)).float())
prediction = torch.softmax(prediction, dim = 1).detach().numpy()
prediction = int(np.argmax(prediction, axis = 1)[0])

GAMES[prediction]
