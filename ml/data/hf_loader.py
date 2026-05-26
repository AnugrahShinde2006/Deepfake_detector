import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HuggingFaceDeepfakeDataset(Dataset):
    def __init__(self, hf_dataset, target_size=(224, 224)):
        """
        Wrapper to convert a Hugging Face dataset into a PyTorch Dataset
        that is compatible with our DataLoader and ImageNet normalization.
        """
        self.dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            # Standard ImageNet normalization matching our pre-trained visual model
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Safely extract the image (handles variations in dataset column names)
        image = item.get('image') or item.get('img')
        
        # Ensure the image is pure RGB before passing to PyTorch
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        tensor_img = self.transform(image)
        
        # Dynamically find the label column (Hugging Face datasets vary wildly)
        lbl = None
        for key in ['label', 'labels', 'target', 'class', 'is_fake', 'fake']:
            if key in item:
                lbl = item[key]
                break
                
        if lbl is None:
            # Fallback: Just grab whatever column isn't the image!
            for k, v in item.items():
                if k not in ['image', 'img']:
                    lbl = v
                    break
                    
        # Label is mapped to a PyTorch float tensor for the BCE loss function
        label = float(lbl)
        
        return tensor_img, torch.tensor(label, dtype=torch.float32)
