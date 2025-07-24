import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BraTSSliceDataset
from src.model import SimpleCNN

# Dataset & loader
dataset = BraTSSliceDataset(data_dir='data', modality_index=3)

# Check input shape
sample, _ = dataset[0]
print("Sample input shape:", sample.shape)  

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(in_channels=4).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Early stopping params
patience = 3
best_loss = float('inf')
epochs_without_improvement = 0

# Training loop
for epoch in range(5):  
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Show batch loss in tqdm
        progress_bar.set_postfix(loss=loss.item())

        # Print outputs only for first 10 batches of first epoch
        if epoch == 0 and batch_idx < 10:
            print(f"[Batch {batch_idx}] Output min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Early stopping logic
    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), f"simplecnn_4ch_epoch{epoch+1}.pth")
        print("Model improved and saved.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break
