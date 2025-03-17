import torch
import os

def train_model(train_dataloader, model, criterion, optimizer, model_path, num_epochs=10, save_interval=1, batch_interval=10):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (features, labels) in enumerate(train_dataloader, start=1):
            features, labels = features.to(device), labels.to(device)
            features = features.permute(1, 0, 2)  # (seq_len, batch, feature_dim)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % batch_interval == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            if batch_idx % save_interval == 0:
                torch.save(model, model_path)
                print(f"Model saved to {model_path}")

    torch.save(model, '../trained_model/latest/model.pth')
    print(f"Model saved to {model_path}")