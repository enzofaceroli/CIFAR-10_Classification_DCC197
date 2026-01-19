VGG16_MANUAL_TESTS = [
    {
        "name": "vgg16_manual_bs64_lr1e-3",
        "model": "vgg16_manual",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 20,
        "freeze_backbone": False,
        "augmentation": True
    },
    {
        "name": "vgg16_manual_bs128_lr1e-3",
        "model": "vgg16_manual",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 20,
        "freeze_backbone": False,
        "augmentation": True
    },
    {
        "name": "vgg16_manual_lr1e-4",
        "model": "vgg16_manual",
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 20,
        "freeze_backbone": False,
        "augmentation": True
    }
]
