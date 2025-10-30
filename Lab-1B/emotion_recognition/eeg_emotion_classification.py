import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os

# Constants
SAMPLE_RATE = 32 # (Hz)
GAMES = ["boring", "calm", "horror", "funny"]





# Read the data
data = []
for game_id, game in enumerate(GAMES):
    game_data = pd.read_csv(os.path.join("data", f"S01G{game_id + 1}AllChannels.csv"))
    game_data["game"] = game
    data.append(game_data)

data = pd.concat(data, axis = 0, ignore_index = True)

data.head()




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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

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
    torch.nn.Flatten(),
    torch.nn.Linear(64, 4),
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



train(n_epoch = 100, model = model)



a_clip = X[0]
plt.plot(a_clip.flatten())



prediction = model(torch.tensor(np.expand_dims(a_clip, 1)).float())
prediction = torch.softmax(prediction, dim = 1).detach().numpy()
prediction = int(np.argmax(prediction, axis = 1)[0])

GAMES[prediction]