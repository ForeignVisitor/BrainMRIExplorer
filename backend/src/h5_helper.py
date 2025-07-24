import h5py
import numpy as np
from PIL import Image


def load_h5_file(path):
    with h5py.File(path, "r") as f:
        if "image" not in f:
            raise KeyError(f"'image' key not found. Keys in file: {list(f.keys())}")

        volume = f["image"][:]
        print(f"[DEBUG] Loaded raw volume shape: {volume.shape}")

        if volume.ndim == 4:
            if volume.shape[1] == 4:
                return volume  # Already shaped (N, 4, 240, 240)
            elif volume.shape[-1] == 4:
                volume = np.transpose(volume, (0, 3, 1, 2))
                print(f"[FIXED] Transposed volume shape: {volume.shape}")
                return volume
            else:
                raise ValueError(f"❌ Unrecognized 4D shape: {volume.shape}")
        
        elif volume.ndim == 3 and volume.shape[-1] == 4:
            print("[INFO] Detected single 2D slice with shape (240, 240, 4). Wrapping as volume.")
            volume = np.transpose(volume, (2, 0, 1))         # (4, 240, 240)
            volume = np.expand_dims(volume, axis=0)          # (1, 4, 240, 240)
            print(f"[FIXED] Wrapped volume shape: {volume.shape}")
            return volume

        else:
            raise ValueError(f"❌ Unsupported shape: {volume.shape} (expected 3D or 4D with channels)")

def get_slice_image(volume, idx):
    
    
    slice_data = volume[idx]  # (4, 240, 240)

    rgb = np.zeros((240, 240, 3), dtype=np.uint8)
    for i, ch in enumerate([0, 2, 1]):
        ch_img = slice_data[ch]
        ch_img = ((ch_img - ch_img.min()) / (ch_img.max() - ch_img.min() + 1e-8)) * 255
        rgb[:, :, i] = ch_img.astype(np.uint8)

    return Image.fromarray(rgb)