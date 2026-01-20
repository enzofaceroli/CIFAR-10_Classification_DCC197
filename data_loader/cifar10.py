import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# add opcao de redimensionar ou nao
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

def get_cifar10_loaders(batch_size, augment=False, resize=False):
    if resize:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
        crop_size = 224
        base_transforms = [transforms.Resize((224, 224))]
    else:
        mean = CIFAR10_MEAN
        std = CIFAR10_STD
        crop_size = 32
        base_transforms = []

    # data augmentation
    if augment:
        train_transform = transforms.Compose(base_transforms + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose(base_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    test_transform = transforms.Compose(base_transforms + [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True, 
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False, 
        download=True,
        transform=test_transform 
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )   
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader