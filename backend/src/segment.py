import torch
import numpy as np
from src.unet import UNet

def load_unet_model(weights_path: str):
    model = UNet(in_channels=4, out_channels=1)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def predict_mask(model, slice_np: np.ndarray):
    # slice_np: (4, 240, 240)
    x = torch.tensor(slice_np, dtype=torch.float32).unsqueeze(0)  # (1, 4, 240, 240)
    with torch.no_grad():
        mask = torch.sigmoid(model(x)).squeeze().cpu().numpy()  # (240, 240)
    return (mask > 0.5).astype(np.uint8)  # Binary mask