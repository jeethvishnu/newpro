import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

# Define the root
root_dir = './dataset'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])

# Define race mapping
race_mapping = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others"
}


# Define custom dataset class
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels = self.load_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def load_dataset(self):
        images = []
        labels = []

        for part_dir in sorted(os.listdir(self.root_dir)):
            if not os.path.isdir(os.path.join(self.root_dir, part_dir)):
                continue
            for filename in sorted(os.listdir(os.path.join(self.root_dir, part_dir))):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    images.append(os.path.join(part_dir, filename))
                    try:
                        # Extract race label from filename based on format [age][gender][race]_[date&time].jpg
                        race_label = int(filename.split('_')[2])
                        labels.append(race_label)
                    except (IndexError, ValueError) as e:
                        print(f"Skipping file {filename} due to error: {e}")
                        images.pop()  # Remove image entry if label extraction fails
                        continue

        if len(images) != len(labels):
            print(f"Error: Mismatch between number of images ({len(images)}) and labels ({len(labels)})")

        return images, labels


# Create dataset instance
dataset = UTKFaceDataset(root_dir=root_dir, transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define CNN model using a pretrained
class PretrainedCNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(PretrainedCNNModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


# Initialize
model = PretrainedCNNModel(num_classes=5)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=15):
    model.train()
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    return train_losses, train_accuracies



train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs=15)

# Function to evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


# Evaluate the model on test data
test_accuracy = evaluate_model(model, test_loader)

# Evaluate the model on training data
train_accuracy = evaluate_model(model, train_loader)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Train Accuracy: {train_accuracy:.4f}")


# Function to plot training losses and accuracies
def plot_losses_and_accuracies(train_losses, train_accuracies, test_accuracy, train_accuracy):
    plt.figure(figsize=(16, 5))

    # Plot training loss
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    # Plot training accuracy
    plt.subplot(1, 4, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot test accuracy as horizontal line
    plt.subplot(1, 4, 3)
    plt.axhline(y=test_accuracy, color='r', linestyle='-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot train accuracy as horizontal line
    plt.subplot(1, 4, 4)
    plt.axhline(y=train_accuracy, color='g', linestyle='-', label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Plot training losses and accuracies
plot_losses_and_accuracies(train_losses, train_accuracies, test_accuracy, train_accuracy)


# Function to plot bar graph of predicted vs actual labels
def plot_bar_graph(model, test_loader, race_mapping):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    labels_count = [all_labels.count(i) for i in range(len(race_mapping))]
    predictions_count = [all_predictions.count(i) for i in range(len(race_mapping))]

    x = range(len(race_mapping))
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    opacity = 0.7

    bars1 = plt.bar(x, labels_count, bar_width, alpha=opacity, color='b', label='Actual')
    bars2 = plt.bar([p + bar_width for p in x], predictions_count, bar_width, alpha=opacity, color='r',
                    label='Predicted')

    plt.xlabel('Race')
    plt.ylabel('Count')
    plt.title('Actual vs Predicted Labels')
    plt.xticks([p + bar_width / 2 for p in x], list(race_mapping.values()))
    plt.legend()

    plt.tight_layout()
    plt.show()


# Plot bar graph of predicted vs actual labels
plot_bar_graph(model, test_loader, race_mapping)


# Function to plot confusion matrix
def plot_confusion_matrix(model, test_loader, race_mapping):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=race_mapping.values(),
                yticklabels=race_mapping.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# Plot confusion matrix
plot_confusion_matrix(model, test_loader, race_mapping)