import torch

def test_model(model, test_loader, criterion, device):
    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    avg_loss = running_loss / len(test_loader)

    print(
        f"Test Loss: {avg_loss:.4f} "
        f"- Test Acc: {acc:.2f}%"
    )

    return avg_loss, acc
