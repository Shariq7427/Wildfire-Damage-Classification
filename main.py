import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. LOAD DATASET
# =========================
dataset = load_dataset("kevincluo/structure_wildfire_damage_classification")
full_data = dataset["train"]

print("Dataset loaded!")
print(full_data)

# =========================
# 2. TRANSFORMS
# =========================
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================
# 3. CUSTOM DATASET WRAPPER
# =========================
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"]
        label = self.data[idx]["label"]

        if self.transform:
            img = self.transform(img)

        return img, label

# =========================
# 4. SPLIT DATASET
# =========================
train_size = int(0.8 * len(full_data))
val_size = int(0.1 * len(full_data))
test_size = len(full_data) - train_size - val_size

train_data, val_data, test_data = random_split(
    full_data,
    [train_size, val_size, test_size]
)

train_dataset = CustomDataset(train_data, train_transform)
val_dataset = CustomDataset(val_data, test_transform)
test_dataset = CustomDataset(test_data, test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# =========================
# 5. SIMPLE CNN MODEL
# =========================
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# =========================
# 6. DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(full_data.features["label"].names)

model = CNN(num_classes).to(device)

# =========================
# 7. LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 8. TRAINING LOOP
# =========================
def train(epochs=5):
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return train_losses

losses = train(epochs=5)

# =========================
# 9. LOSS PLOT
# =========================
plt.plot(losses)
plt.title("Training Loss")
plt.show()

# =========================
# 10. EVALUATION
# =========================
y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# =========================
# 11. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

# =========================
# 12. REPORT
# =========================
print(classification_report(y_true, y_pred))