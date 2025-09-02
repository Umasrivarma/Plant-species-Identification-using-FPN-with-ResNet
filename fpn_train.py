import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data_loader import PlantDataset
from fpn_resnet import ResNetFPNClassifier  # <-- updated import

data_dir = 'plant_species/my_plant_dataset'
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
num_classes = len(class_names)
print(f"📂 Found {num_classes} classes")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = PlantDataset(root_dir=data_dir, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = ResNetFPNClassifier(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 5
train_acc_list = []
val_acc_list = []

print(f"\n🔄 Starting training for {epochs} epochs...\n")
for epoch in range(epochs):
    print(f"🔄 Epoch {epoch+1}/{epochs}")
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress = 100. * (batch_idx + 1) / len(train_loader)
        acc = 100. * correct / total
        print(f"\r[Train] Progress: {progress:.2f}%, Loss: {loss.item():.4f}, Acc: {acc:.2f}%", end='')

    epoch_train_acc = 100. * correct / total
    train_acc_list.append(epoch_train_acc)
    print(f"\n✅ Epoch {epoch+1} Training Acc: {epoch_train_acc:.2f}%")

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = logits.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    epoch_val_acc = 100. * val_correct / val_total
    val_acc_list.append(epoch_val_acc)
    print(f"🔎 Validation Acc: {epoch_val_acc:.2f}%\n")

torch.save(model.state_dict(), 'fpn_classifier_model_final.pth')
print("💾 Saved model to fpn_classifier_model_final.pth")

final_train_acc = sum(train_acc_list) / len(train_acc_list)
final_val_acc = sum(val_acc_list) / len(val_acc_list)
print("🎉 Training completed!")
