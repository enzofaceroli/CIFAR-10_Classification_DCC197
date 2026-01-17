# main.py

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader.cifar10 import get_cifar10_loaders
from test import TESTS

from models.resnet50_transfer import build_resnet50
from models.vgg16_transfer import build_vgg16
from models.densenet121_transfer import build_densenet121
from models.vgg16_manual import VGG16Custom

from utils.train import train_model
from utils.evaluate import evaluate_model
from utils.plots import plot_confusion_matrix


def get_model(exp_config, num_classes):
    model_name = exp_config["model"]

    if model_name == "resnet50":
        return build_resnet50(
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=exp_config["freeze_backbone"]
        )

    if model_name == "vgg16":
        return build_vgg16(
            num_classes=num_classes,
            pretrained=True,
            freeze_features=exp_config["freeze_features"]
        )

    if model_name == "densenet121":
        return build_densenet121(
            num_classes=num_classes,
            pretrained=True,
            freeze_features=exp_config["freeze_features"]
        )

    if model_name == "vgg16_manual":
        return VGG16Custom(num_classes=num_classes)

    raise ValueError("Modelo não suportado.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    for exp in EXPERIMENTS:
        print("=" * 70)
        print(f"Experimento: {exp['name']}")

        train_loader, test_loader = get_cifar10_loaders(
            batch_size=exp["batch_size"],
            augment=exp["augmentation"]
        )

        model = get_model(exp, num_classes=10)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=exp["learning_rate"]
        )

        train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=exp["epochs"]
        )

        acc, cm = evaluate_model(model, test_loader, device)
        print(f"Acurácia final: {acc * 100:.2f}%")

        plot_confusion_matrix(
            cm,
            class_names,
            save_path=f"confusion_matrix_{exp['name']}.pdf"
        )

    print("Todos os experimentos finalizados.")


if __name__ == "__main__":
    main()
