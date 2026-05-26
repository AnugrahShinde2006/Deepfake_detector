import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DeepfakeVisualModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeVisualModel, self).__init__()
        
        # Load EfficientNet-B0
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.model = efficientnet_b0(weights=weights)
        else:
            self.model = efficientnet_b0(weights=None)
            
        # The default EfficientNet-B0 output is 1000 classes (ImageNet).
        # We replace the classifier head to output a single value (binary classification).
        # EfficientNet-B0's final classifier is a Sequential block. The in_features of the last Linear layer is 1280.
        
        num_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=num_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=1)
        )

    def forward(self, x):
        # Outputs a single raw logit per image
        return self.model(x)

if __name__ == "__main__":
    # Test model initialization and output shape
    model = DeepfakeVisualModel(pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}") # Expected: [1, 1]
