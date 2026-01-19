import torch
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        print(f"INICIANDO EPOCA {epoch}")
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True
        )
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%"
            )

        epoch_loss = running_loss / len(train_loader)
        acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {acc:.2f}%")
    