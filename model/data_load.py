import os
import torch
from torchvision import datasets, transforms
import random
from torch.utils.data import random_split, DataLoader, Subset
import json

class VGGFace2Dataset:
    def __init__(self, root_dir, batch_size, split_file="data_split.json"):
        self.root_dir = root_dir
        self.split_file = split_file
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        full_dataset = datasets.ImageFolder(root=os.path.join(self.root_dir, 'train'), transform=transform)
        
        train_indices, test_indices = self._get_split_indices(len(full_dataset))

        self.train_dataset = TripletVGGFace2(full_dataset, train_indices)
        self.test_dataset = TripletVGGFace2(full_dataset, test_indices)

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

class TripletVGGFace2(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.labels = [dataset.targets[i] for i in indices]
        self.label_to_indices = self._get_label_to_indices()

    def _get_label_to_indices(self):
        label_to_indices = {}
        for idx in self.indices:
            label = self.dataset.targets[idx]
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __getitem__(self, idx):
        # Get anchor
        anchor_idx = self.indices[idx]
        anchor_label = self.dataset.targets[anchor_idx]
        anchor_img = self.dataset[anchor_idx][0]

        # Get positive (same class as anchor)
        positive_idx = random.choice([i for i in self.label_to_indices[anchor_label] if i != anchor_idx])
        positive_img = self.dataset[positive_idx][0]

        # Get negative (different class from anchor)
        negative_label = random.choice([l for l in self.label_to_indices.keys() if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img = self.dataset[negative_idx][0]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.indices)