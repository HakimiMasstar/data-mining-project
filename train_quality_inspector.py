import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import json
import os

# Config
DATASET_DIR = 'Datasets'
MODEL_PATH = 'quality_inspector.pth'
INDICES_PATH = 'class_indices.json'
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InspectorCNN(nn.Module):
    def __init__(self, num_classes):
        super(InspectorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train():
    print(f"Training Quality Inspector on {device}...")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # We use 'train' folder if it exists, else root
    train_dir = os.path.join(DATASET_DIR, 'train')
    if not os.path.exists(train_dir):
        train_dir = DATASET_DIR
        
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Save Class Map
    with open(INDICES_PATH, 'w') as f:
        json.dump(dataset.class_to_idx, f)
        
    model = InspectorCNN(len(dataset.classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.2f}%")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Inspector trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
