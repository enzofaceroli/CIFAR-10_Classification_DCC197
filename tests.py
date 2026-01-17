TESTS = [

    # =====================================================
    # ResNet50 — impacto do congelamento
    # =====================================================
    {
        "name": "resnet50_bs128_lr1e-3_frozen_aug",
        "model": "resnet50",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": True,
        "augmentation": True
    },
    {
        "name": "resnet50_bs128_lr1e-4_unfrozen_aug",
        "model": "resnet50",
        "batch_size": 128,
        "learning_rate": 1e-4,
        "epochs": 15,
        "freeze_backbone": False,
        "augmentation": True
    },

    # =====================================================
    # ResNet50 — impacto do batch size
    # =====================================================
    {
        "name": "resnet50_bs64_lr1e-3_frozen_aug",
        "model": "resnet50",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": True,
        "augmentation": True
    },
    {
        "name": "resnet50_bs256_lr1e-3_frozen_aug",
        "model": "resnet50",
        "batch_size": 256,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": True,
        "augmentation": True
    },

    # =====================================================
    # VGG16 — impacto do data augmentation
    # =====================================================
    {
        "name": "vgg16_bs128_lr1e-3_frozen_noaug",
        "model": "vgg16",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 15,
        "freeze_features": True,
        "augmentation": False
    },
    {
        "name": "vgg16_bs128_lr1e-3_frozen_aug",
        "model": "vgg16",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 15,
        "freeze_features": True,
        "augmentation": True
    },

    # =====================================================
    # VGG16 — impacto do learning rate
    # =====================================================
    {
        "name": "vgg16_bs128_lr1e-4_frozen_aug",
        "model": "vgg16",
        "batch_size": 128,
        "learning_rate": 1e-4,
        "epochs": 20,
        "freeze_features": True,
        "augmentation": True
    },

    # =====================================================
    # DenseNet121 — comparação direta com ResNet
    # =====================================================
    {
        "name": "densenet121_bs128_lr1e-3_frozen_aug",
        "model": "densenet121",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_features": True,
        "augmentation": True
    },
    {
        "name": "densenet121_bs128_lr1e-4_unfrozen_aug",
        "model": "densenet121",
        "batch_size": 128,
        "learning_rate": 1e-4,
        "epochs": 15,
        "freeze_features": False,
        "augmentation": True
    },

    # =====================================================
    # Comparação justa entre arquiteturas (mesma config)
    # =====================================================
    {
        "name": "resnet50_baseline",
        "model": "resnet50",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_backbone": True,
        "augmentation": True
    },
    {
        "name": "vgg16_baseline",
        "model": "vgg16",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_features": True,
        "augmentation": True
    },
    {
        "name": "densenet121_baseline",
        "model": "densenet121",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epochs": 10,
        "freeze_features": True,
        "augmentation": True
    }
]
