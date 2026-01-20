DENSENET121_TESTS = [
    # Batch size
    {
        "name": "densenet121_bs64_lr1e-3",
        "model": "densenet121",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": False,
        "augmentation": True
    },
    {
        "name": "densenet121_bs128_lr1e-3",
        "model": "densenet121",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": False,
        "augmentation": True
    },

    # Learning rate
    {
        "name": "densenet121_lr1e-3_ep10",
        "model": "densenet121",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": False,
        "augmentation": True
    },
    {
        "name": "densenet121_lr1e-4_ep15",
        "model": "densenet121",
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 15,
        "freeze_backbone": False,
        "augmentation": True
    },
    {
        "name": "densenet121_lr1e-5_ep20",
        "model": "densenet121",
        "batch_size": 64,
        "learning_rate": 1e-5,
        "epochs": 20,
        "freeze_backbone": False,
        "augmentation": True
    },

    # Melhor configuração congelada
    {
        "name": "densenet121_frozen",
        "model": "densenet121",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": True,
        "augmentation": True
    },

]