import torch
from torchvision import datasets, transforms

class VGGFace2Dataset:
    def __init__(self, root_dir, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = datasets.ImageFolder(root=root_dir + '/train', transform=transform)
        self.test_dataset = datasets.ImageFolder(root=root_dir + '/test', transform=transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
