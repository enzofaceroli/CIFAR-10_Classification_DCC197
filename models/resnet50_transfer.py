import torch.nn as nn
from torchvision import models 

def build_resnet50(num_classes = 10, pretrained = True, freeze_backbone = True):
    model = models.resnet50(pretrained = pretrained)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model