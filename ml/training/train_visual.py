import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.visual_model import DeepfakeVisualModel
from data.dataset import DeepfakeDataset

def train_visual_model(data_dir, epochs=10, batch_size=128, learning_rate=1e-4):
    # Set device and enable cuDNN benchmark for faster training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 1. Prepare Dataset
    print("Loading Native PyTorch Dataset...")
    full_dataset = DeepfakeDataset(data_dir=data_dir)
    
    if len(full_dataset) == 0:
        print("Dataset is empty. Run python ml/data/download_dataset.py first!")
        return

    # Train/Val split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Initialize Model, Loss, Optimizer, and Scaler (for Mixed Precision)
    model = DeepfakeVisualModel(pretrained=True).to(device)
    
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Creates a GradScaler once at the beginning of training for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_dir, 'visual_best.pt')

    # 3. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1) # [batch_size, 1]

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision context manager
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scales loss and calls backward() to create scaled gradients
            scaler.scale(loss).backward()
            
            # Unscales gradients and calls optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(correct_train/total_train):.4f}"})

        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        print(f"Epoch [{epoch+1}/{epochs}] - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_weights_path)
            print("Saved new best model.")

if __name__ == "__main__":
    # Based on RTX 3050 stress test showing a max limit of 189,
    # we set a safe but extremely fast batch_size of 128.
    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'processed_data', 'faces')
    train_visual_model(data_dir=data_directory, epochs=10, batch_size=128, learning_rate=1e-4)
