import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_cifar10_loaders(batch_size, augment=False):
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),            
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    # ========================== DATASETS ==========================
    train_dataset = datasets.CIFAR10(
        root = "./data",
        train = True, 
        download = True,
        transform = train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root = "./data",
        train = False, 
        download = True,
        transform = test_transform 
    )
    
    # =========================== LOADERS ===========================
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2,
        pin_memory=True
    )   
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False, 
        num_workers = 2,
        pin_memory=True
    )
    
    return train_loader, test_loader