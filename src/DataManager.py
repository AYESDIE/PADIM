import os
import PIL
import torch
import torchvision

class DatasetManager(torch.utils.data.Dataset):
    def __init__(self, dataset_path, image_transform = None):
        self.dataset_path = dataset_path
        self.image_transform = image_transform
        if self.image_transform is None:
            self.image_transform  = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.image_paths = []
        for image_path in sorted(os.listdir(self.dataset_path)):
            self.image_paths.append(f"{self.dataset_path}/{image_path}")

        self.len = len(self.image_paths)
    
    def __getitem__(self, idx):
        x = self.image_paths[idx]
        x = PIL.Image.open(x).convert("RGB")
        x = self.image_transform(x)
        return x

    def __len__(self):
        return self.len