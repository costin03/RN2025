import pickle
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ExtendedMNISTDataset(Dataset):
    def __init__(self, root: str = '/kaggle/input/fii-nn-2025-homework-4', train: bool = True):
        if train:
            file_path = os.path.join(root, "extended_mnist_train.pkl")
        else:
            file_path = os.path.join(root, "extended_mnist_test.pkl")

        with open(file_path, "rb") as fp:
            self.data = pickle.load(fp)

        self.train = train

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        image, label = self.data[i]
        image = np.array(image, dtype=np.float32).flatten() / 255.0

        return torch.tensor(image), torch.tensor(label, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = ExtendedMNISTDataset(train=True)
test_dataset = ExtendedMNISTDataset(train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Antrenare:")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoca {epoch + 1}/{EPOCHS}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({'Loss': running_loss / (total / BATCH_SIZE), 'Acc': 100 * correct / total})

print("Antrenare finalizata.")

model.eval()
predictions = []

with torch.no_grad():
    for images, _ in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

submission = pd.DataFrame({
    'ID': range(len(predictions)),
    'target': predictions
})

submission.to_csv('submission.csv', index=False)
