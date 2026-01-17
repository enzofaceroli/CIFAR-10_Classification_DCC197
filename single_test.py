import torch
import torch.nn as nn
import torch.optim as optim

from data_loader.cifar10 import get_cifar10_loaders

from models.resnet50_transfer import build_resnet50
from models.vgg16_transfer import build_vgg16
from models.densenet121_transfer import build_densenet121
from models.vgg16_manual import VGG16Custom

from utils.train import train_model
from utils.evaluate import evaluate_model
from utils.plots import plot_confusion_matrix


def choose_model():
    print("Escolha o modelo:")
    print("1 - ResNet50 (transfer)")
    print("2 - VGG16 (transfer)")
    print("3 - DenseNet121 (transfer)")
    print("4 - VGG16 (manual)")

    option = input("Opção: ")

    if option == "1":
        freeze = input("Congelar backbone? (s/n): ").lower() == "s"
        return "resnet50", {"freeze_backbone": freeze}

    if option == "2":
        freeze = input("Congelar features? (s/n): ").lower() == "s"
        return "vgg16", {"freeze_features": freeze}

    if option == "3":
        freeze = input("Congelar features? (s/n): ").lower() == "s"
        return "densenet121", {"freeze_features": freeze}

    if option == "4":
        return "vgg16_manual", {}

    raise ValueError("Opção inválida.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model_name, extra_params = choose_model()

    batch_size = int(input("Batch size: "))
    learning_rate = float(input("Learning rate: "))
    epochs = int(input("Épocas: "))
    augmentation = input("Usar data augmentation? (s/n): ").lower() == "s"

    #dataset
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        augment=augmentation
    )

    #modelo
    if model_name == "resnet50":
        model = build_resnet50(num_classes=10, pretrained=True, **extra_params)

    elif model_name == "vgg16":
        model = build_vgg16(num_classes=10, pretrained=True, **extra_params)

    elif model_name == "densenet121":
        model = build_densenet121(num_classes=10, pretrained=True, **extra_params)

    elif model_name == "vgg16_manual":
        model = VGG16Custom(num_classes=10)

    else:
        raise ValueError("Modelo não suportado.")

    model.to(device)

    #treinamento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )

    #avaliação
    acc, cm = evaluate_model(model, test_loader, device)
    print(f"Acurácia no teste: {acc * 100:.2f}%")

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()
