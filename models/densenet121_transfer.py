import torch.nn as nn
from torchvision import models

def build_densenet121(num_classes = 10, pretrained = True, freeze_backbone = True):
    model = models.densenet121(pretrained = pretrained)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model