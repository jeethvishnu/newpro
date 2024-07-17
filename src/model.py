import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class PretrainedCNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(PretrainedCNNModel, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
