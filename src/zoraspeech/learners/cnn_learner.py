
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random


class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transforms=transforms.MelSpectrogram()):
        self.file_paths = file_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        audio_path = self.file_paths[idx]

        waveform, _ = torchaudio.load(audio_path)

        # ensure its mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0).unsqueeze(0)

        # apply transforms
        if self.transforms:
            spec = self.transforms(waveform)
        return spec, self.labels[idx]

class CNNLearner:
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.train_size = 0
        self.validation_size = 0
        self.test_size = 0

    def create_dataset(self):
        # Create datasets
        full_dataset = AudioDataset(files, labels, transforms=transforms.MelSpectrogram())

        self.train_size = int(0.7 * len(full_dataset))
        self.validation_size = int(0.2 * len(full_dataset))
        self.test_size = int(0.1 * len(full_dataset))


    def learn(self):
        print("learning")
        print("done learning")
        print("saving model weights to weights/weights.pth")

'''
print(train_size)
print(validation_size)
print(test_size)

train_dataset, validation_dataset, test_dataset  = t.utils.data.random_split(full_dataset, [train_size, validation_size, test_size])


# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(validation_loader)}")


####

import matplotlib.pyplot as plt

# see a batch
for batch in train_loader:
    inputs, targets = batch
    print(inputs.shape)
    print(inputs[0][0].shape)
    print(targets)
    break


mel_freq_bins = inputs[0][0].shape[0]
time_steps = inputs[0][0].shape[1]

print("mel freq bins: ", mel_freq_bins)
print("time steps: ", time_steps)


####

# train with conv net
import torch.nn as nn

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ConvModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of the flattened features
        self.flat_features = mel_freq_bins * (mel_freq_bins // 8) * (time_steps // 8)
        
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, 1, 128, 366)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.flat_features) # rewrite this line with einops / ARENA
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model
conv_model = ConvModel()
print(conv_model)

# Move model to the appropriate device
conv_model = conv_model.to(device)

loss_fn = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(conv_model.parameters(), lr=0.001)

epochs = 15

loss_history = []

print(f"Training for {epochs} epochs")
for epoch in tqdm(range(epochs)):
    conv_model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = conv_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    loss_history.append(avg_loss)

print("finished training")

# plot the loss
plt.plot(loss_history)
plt.show()

# evaluate our model

conv_model.eval()
correct = 0
total = 0

# plot the accuracy
accuracy_history = []

with t.no_grad():
    for batch in validation_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = conv_model(inputs)
        _, predicted = t.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        accuracy_history.append(100 * correct / total)

plt.plot(accuracy_history)
plt.show()

print(f"Accuracy of the conv model on the test set: {100 * correct / total}%")


#####


# save the model with today's datetime
import datetime
datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#make dir called model_weights
os.makedirs('model_weights', exist_ok=True) 
saved_model_path = f'model_weights/audrey_model_weights_{datetime}.pth'

t.save(conv_model.state_dict(), saved_model_path)
'''