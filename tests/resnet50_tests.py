RESNET50_TESTS = [
    # Batch size (lr = 1e-3)
    {
        "name": "resnet50_bs64_lr1e-3",
        "model": "resnet50",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": False,
        "augmentation": True
    },
    {
        "name": "resnet50_bs128_lr1e-3",
        "model": "resnet50",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": False,
        "augmentation": True
    },

    # Learning rate (batch escolhido = 64)
    {
        "name": "resnet50_lr1e-3_ep10",
        "model": "resnet50",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": False,
        "augmentation": True
    },
    {
        "name": "resnet50_lr1e-4_ep15",
        "model": "resnet50",
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 15,
        "freeze_backbone": False,
        "augmentation": True
    },
    {
        "name": "resnet50_lr1e-5_ep20",
        "model": "resnet50",
        "batch_size": 64,
        "learning_rate": 1e-5,
        "epochs": 20,
        "freeze_backbone": False,
        "augmentation": True
    },

    # Melhor configuração com backbone congelado
    {
        "name": "resnet50_best_frozen",
        "model": "resnet50",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": True,
        "augmentation": True
    },
]