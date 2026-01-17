import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, num_workers=2, data_dir="./data", augment=False):
    normalize = transforms.Normalize(
        # valores encontrados na literatura para normalização do dataset
        mean = (0.4914, 0.4822, 0.4465), 
        std = (0.2023, 0.1994, 0.2010)
    )
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # ========================== DATASETS ==========================
    train_dataset = datasets.CIFAR10(
        root = data_dir,
        train = True, 
        download = True,
        transform = train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root = data_dir,
        train = False, 
        download = True,
        transform = test_transform 
    )
    
    # =========================== LOADERS ===========================
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False, 
        num_workers = num_workers
    )
    
    return train_loader, test_loader