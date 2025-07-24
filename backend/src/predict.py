import torch
import numpy as np
from src.model import SimpleCNN

def load_model(weights_path: str):
    model = SimpleCNN(in_channels=4)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def predict_all_slices(model, volume_np: np.ndarray):

    
    scores = []
    labels = []

    with torch.no_grad():

        for slice_np in volume_np:
            print("[DEBUG] slice_np.shape before transpose:", slice_np.shape)

            if slice_np.ndim == 3:
                if slice_np.shape[0] == 4:
                    pass
                elif slice_np.shape[-1] == 4:
                    # Transpose from HWC → CHW
                    slice_np = np.transpose(slice_np, (2, 0, 1))
                else:
                    raise ValueError(f"Unrecognized shape: {slice_np.shape}")
            else:
                raise ValueError(f"Expected 3D slice, got shape: {slice_np.shape}")

            # Convert to tensor + predict
            x = torch.tensor(slice_np, dtype=torch.float32).unsqueeze(0)  # (1, 4, 240, 240)
            print("[DEBUG] Input to model:", x.shape)
            score = torch.sigmoid(model(x)).item()
            scores.append(score)
            labels.append(1 if score > 0.5 else 0)        

    return np.array(scores), np.array(labels)