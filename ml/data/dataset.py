import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dynamically loads images from the disk to save RAM.
        data_dir should have two subfolders: 'real' and 'fake'.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')
        
        if os.path.exists(real_dir):
            for root, _, files in os.walk(real_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(0) # 0 for real
                        
        if os.path.exists(fake_dir):
            for root, _, files in os.walk(fake_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(1) # 1 for fake

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image with OpenCV
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for corrupt images: create a blank 224x224 image
            import numpy as np
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image for torchvision transforms
        from PIL import Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to Tensor with Resizing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)
