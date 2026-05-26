import torch
import gc
import sys
import os

# Ensure we can import from models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.visual_model import DeepfakeVisualModel

def stress_test_gpu():
    print("=== RTX 3050 VRAM Stress Test ===", flush=True)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure PyTorch with CUDA is installed.", flush=True)
        return

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    
    print("Loading PyTorch model into GPU (this might take a few seconds to download weights)...", flush=True)
    model = DeepfakeVisualModel(pretrained=True).to(device)
    model.train() 
    
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    batch_size = 1
    
    try:
        while True:
            print(f"Testing Batch Size: {batch_size}...", end=" ", flush=True)
            
            dummy_images = torch.randn(batch_size, 3, 224, 224, device=device)
            dummy_labels = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(dummy_images)
                loss = criterion(outputs, dummy_labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            allocated_vram = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_vram = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"Success! | Allocated: {allocated_vram:.2f} GB | Reserved: {reserved_vram:.2f} GB", flush=True)
            
            del dummy_images
            del dummy_labels
            del outputs
            del loss
            torch.cuda.empty_cache()
            
            if batch_size < 32:
                batch_size += 4
            else:
                batch_size += 16
                
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[💥 BOOM! Out Of Memory!] Your RTX 3050 crashed at Batch Size {batch_size}.", flush=True)
            print("Your maximum safe batch size for training is likely around", max(1, batch_size - 4), flush=True)
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print(f"An unexpected error occurred: {e}", flush=True)

if __name__ == "__main__":
    stress_test_gpu()
