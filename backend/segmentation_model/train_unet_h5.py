import os
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from unet import UNet


DATA_DIR = "./data"
EPOCHS = 5
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "unet_brats_h5_best.pth"
VAL_SPLIT = 0.1

class H5SliceDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        with h5py.File(self.files[idx], 'r') as f:
            img = f['image'][:]  # (240, 240, 4)
            mask = f['mask'][:]  # (240, 240) or (240, 240, 3)
        img = np.transpose(img, (2, 0, 1))
        img = (img - img.min(axis=(1,2), keepdims=True)) / (img.max(axis=(1,2), keepdims=True) - img.min(axis=(1,2), keepdims=True) + 1e-8)
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)


ds = H5SliceDataset(DATA_DIR)
val_size = int(len(ds) * VAL_SPLIT)
train_size = len(ds) - val_size
train_ds, val_ds = random_split(ds, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


model = UNet(in_channels=4, out_channels=1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

best_val_loss = float('inf')

for epoch in range(EPOCHS):
   
    model.train()
    train_loss = 0
    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=train_loss / (pbar.n + 1))
    train_loss /= len(train_dl)

    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
    val_loss /= len(val_dl)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

   
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model improved and saved.")