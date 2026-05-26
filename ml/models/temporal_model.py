import torch
import torch.nn as nn

class DeepfakeTemporalModel(nn.Module):
    def __init__(self, feature_dim=1280, hidden_dim=256, num_layers=1, dropout=0.3):
        """
        Takes a sequence of feature vectors extracted by the Visual Model 
        (EfficientNet-B0 has a feature dimension of 1280 before the classifier).
        """
        super(DeepfakeTemporalModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, feature_dim)
        # For example: 10 frames of 1280-dim vectors -> (batch, 10, 1280)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # We take the hidden state of the final timestep
        final_hidden_state = hidden[-1]
        
        out = self.classifier(final_hidden_state)
        return out

if __name__ == "__main__":
    # Test with sequence length 10, feature dim 1280
    model = DeepfakeTemporalModel()
    dummy_sequence = torch.randn(1, 10, 1280)
    out = model(dummy_sequence)
    print(f"Temporal Model output shape: {out.shape}")
