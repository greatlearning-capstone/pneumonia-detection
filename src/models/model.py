import torch
import torch.nn as nn
from torchvision import models

class PneumoniaClassifier(nn.Module):
    """
    Simple pneumonia classification model using ResNet-50.
    
    Args:
        num_classes (int): Number of output classes (default: 3)
        pretrained (bool): Use pretrained weights (default: True)
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        super(PneumoniaClassifier, self).__init__()
        
        # Load ResNet-50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        """Forward pass"""
        return self.resnet(x)

if __name__ == "__main__":
    
    model = PneumoniaClassifier(num_classes=3)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
