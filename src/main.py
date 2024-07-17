import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data_loading import UTKFaceDataset, transform
from model import PretrainedCNNModel
from training import train_model
from evaluation import evaluate_model, calculate_metrics
from plotting import plot_losses_and_accuracies, plot_bar_graph, plot_confusion_matrix


# Define race mapping
race_mapping = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others"
}


# Define the root directory where your UTKFace dataset is located
root_dir = "../dataset"


# Create dataset instance
dataset = UTKFaceDataset(root_dir=root_dir, transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = PretrainedCNNModel(num_classes=5)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1  # Adjust as needed
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

# Evaluate the model on test data
test_accuracy, test_labels, test_predictions, test_probabilities = evaluate_model(model, test_loader)

# Evaluate the model on training data
train_accuracy, train_labels, train_predictions, train_probabilities = evaluate_model(model, train_loader)

print(f"Test Accuracy: {train_accuracy:.4f}")
print(f"Train Accuracy: {test_accuracy:.4f}")

# Calculate Precision, Recall, F1 Score, and ROC-AUC
precision, recall, f1, roc_auc = calculate_metrics(test_labels, test_predictions, test_probabilities)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Save the model
torch.save(model.state_dict(), os.path.join('..', 'models', 'utkface_model.pth'))


# Plot training losses and accuracies
plot_losses_and_accuracies(train_losses, train_accuracies, test_accuracy, train_accuracy)

# Plot bar graph of predicted vs actual labels
plot_bar_graph(model, test_loader, UTKFaceDataset.race_mapping)

# Plot confusion matrix
plot_confusion_matrix(model, test_loader, UTKFaceDataset.race_mapping)
