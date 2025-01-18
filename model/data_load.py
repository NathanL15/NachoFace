import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
import json

class VGGFace2Dataset:
    def __init__(self, root_dir, batch_size, split_file="data_split.json"):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        full_dataset = datasets.ImageFolder(root=os.path.join(self.root_dir, 'train'), transform=transform)
        
        train_indices, test_indices = self._get_split_indices(len(full_dataset))

        self.train_dataset = Subset(full_dataset, train_indices)
        self.test_dataset = Subset(full_dataset, test_indices)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def _get_split_indices(self, dataset_size):
        if os.path.exists(self.split_file):
            with open(self.split_file, 'r') as f:
                split = json.load(f)
            return split['train'], split['test']
        else:
            # Create a new split
            indices = torch.randperm(dataset_size).tolist()
            train_size = int(dataset_size * 0.8)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            # Save the split for reproducibility
            split = {'train': train_indices, 'test': test_indices}
            with open(self.split_file, 'w') as f:
                json.dump(split, f)
            return train_indices, test_indices