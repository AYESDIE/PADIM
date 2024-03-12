import os
import PIL
import torch
import torchvision

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, is_train, image_transform = None, mask_transform = None):
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.image_transform = image_transform
        if self.image_transform is None:
            self.image_transform  = torchvision.transforms.Compose([
                torchvision.transform.Resize(256),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.mask_transform = mask_transform
        if self.mask_transform is None:
            self.mask_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor()
            ])
        
        self._x, self._y, self._mask = self.load_dataset()
        self.len = len(self._x)
        
    def load_dataset(self):
        x = []
        y = []
        mask = []

        if not self.is_train:
            _ground_truth_dir = self.dataset_path.replace("test", "ground_truth")
            dirs = os.listdir(self.dataset_path)
            dirs.remove("good")
            for dir in dirs:
                for img, _mask in zip(sorted(os.listdir(f"{self.dataset_path}/{dir}")),
                                    os.listdir(f"{_ground_truth_dir}/{dir}")):
                    x.append(f"{self.dataset_path}/{dir}/{img}")
                    y.append(1)
                    mask.append(f"{_ground_truth_dir}/{dir}/{_mask}")

        for img in sorted(os.listdir(f"{self.dataset_path}/good")):
            x.append(f"{self.dataset_path}/good/{img}")
            y.append(0)
            mask.append(None)

        assert len(x) == len(y)
        return x, y, mask
    
    def __getitem__(self, idx):
        x = self._x[idx]
        y = self._y[idx]
        mask = self._mask[idx]

        x = PIL.image.open(x).convert("RGB")
        x = self.image_transform(x)

        if y == 0:
            mask = torch.zeros([1, 256, 256])
        else:
            mask = PIL.image.open(mask)
            mask = self.mask_transform(mask)
        
        return x, y, mask

    def __len__(self):
        return self.len