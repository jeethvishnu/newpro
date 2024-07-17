import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_losses_and_accuracies(train_losses, train_accuracies, test_accuracy, train_accuracy):
    plt.figure(figsize=(16, 5))

    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    # Plot training accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot test accuracy as horizontal line
    plt.subplot(1, 3, 3)
    plt.axhline(y=test_accuracy, color='r', linestyle='-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

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
    bars2 = plt.bar([p + bar_width for p in x], predictions_count, bar_width, alpha=opacity, color='r', label='Predicted')

    plt.xlabel('Race')
    plt.ylabel('Count')
    plt.title('Actual vs Predicted Labels')
    plt.xticks([p + bar_width / 2 for p in x], list(race_mapping.values()))
    plt.legend()

    plt.tight_layout()
    plt.show()

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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=race_mapping.values(), yticklabels=race_mapping.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
