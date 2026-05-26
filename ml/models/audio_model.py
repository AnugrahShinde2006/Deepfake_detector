import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DeepfakeAudioModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeAudioModel, self).__init__()
        
        # Load lightweight ResNet-18
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
        else:
            self.model = resnet18(weights=None)
            
        # We are passing Mel-Spectrograms (which we saved as RGB images) to the model.
        # ResNet18 outputs 1000 classes by default. We change the fully connected layer.
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = DeepfakeAudioModel(pretrained=False)
    dummy_spectrogram = torch.randn(1, 3, 224, 224)
    out = model(dummy_spectrogram)
    print(f"Audio Model output shape: {out.shape}")
