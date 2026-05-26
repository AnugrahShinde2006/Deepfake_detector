import torch
import torch.nn as nn

class DeepfakeFusionModel(nn.Module):
    def __init__(self):
        """
        A simple Multi-Layer Perceptron (MLP) that ingests the scalar confidence 
        outputs from the Visual, Audio, and Temporal models.
        Input size = 3 (visual_score, audio_score, temporal_score)
        """
        super(DeepfakeFusionModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, visual_score, audio_score, temporal_score):
        # Concatenate the three scores into a single vector of size 3
        # Ensure they are all shape [batch_size, 1]
        x = torch.cat((visual_score, audio_score, temporal_score), dim=1)
        return self.network(x)

if __name__ == "__main__":
    model = DeepfakeFusionModel()
    v_score = torch.rand(2, 1) # batch of 2
    a_score = torch.rand(2, 1)
    t_score = torch.rand(2, 1)
    out = model(v_score, a_score, t_score)
    print(f"Fusion Model output shape: {out.shape}")
