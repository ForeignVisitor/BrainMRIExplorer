from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles 
import uuid
from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fastapi.responses import StreamingResponse
from datetime import datetime

import io
import numpy as np

from src.h5_helper import load_h5_file, get_slice_image
from src.predict import load_model, predict_all_slices
from src.segment import load_unet_model, predict_mask

UPLOAD_DIR = Path("uploads")
SLICE_DIR = Path("slices")
MODEL_PATH = Path("models/simplecnn_4ch_epoch5.pth")
SEG_MODEL_PATH = Path("models/unet_brats_h5_best.pth")

UPLOAD_DIR.mkdir(exist_ok=True)
SLICE_DIR.mkdir(exist_ok=True)

app = FastAPI()


app.mount("/precomputed", StaticFiles(directory="precomputed"), name="precomputed")

# Allow React frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load models once
model = load_model(MODEL_PATH)
seg_model = load_unet_model(SEG_MODEL_PATH)

def get_region_for_slice(idx):
    if 0 <= idx <= 24:
        return "Cerebellum/Brainstem"
    elif 25 <= idx <= 49:
        return "Occipital Lobe"
    elif 50 <= idx <= 79:
        return "Temporal Lobe"
    elif 80 <= idx <= 114:
        return "Parietal Lobe"
    elif 115 <= idx <= 154:
        return "Frontal Lobe"
    else:
        return None

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".h5"):
        raise HTTPException(status_code=400, detail="Invalid file format")

    file_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{file_id}.h5"

    with open(path, "wb") as f:
        f.write(await file.read())

    volume = load_h5_file(path)
    #ENFORCE 155 SLICES
    if volume.shape[0] != 155:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file must contain 155 slices (found {volume.shape[0]})."
        )

    scores, labels = predict_all_slices(model, volume)

    #Find anomalous regions using segmentation model
    region_anomaly = {
        "Cerebellum/Brainstem": False,
        "Occipital Lobe": False,
        "Temporal Lobe": False,
        "Parietal Lobe": False,
        "Frontal Lobe": False
    }
    for idx in range(volume.shape[0]):
        slice_np = volume[idx]  # (4, 240, 240)
        mask = predict_mask(seg_model, slice_np)  # (240, 240)
        if np.count_nonzero(mask) > 10:  # If mask has >10 nonzero pixels
            region = get_region_for_slice(idx)
            if region:
                region_anomaly[region] = True

    return {
        "file_id": file_id,
        "num_slices": volume.shape[0],
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "region_anomaly": region_anomaly,
    }

@app.get("/slice/{file_id}/{index}")
def get_slice_image_api(file_id: str, index: int):
    h5_path = UPLOAD_DIR / f"{file_id}.h5"
    out_path = SLICE_DIR / f"{file_id}_{index}.png"

    if not h5_path.exists():
        raise HTTPException(status_code=404, detail="MRI file not found")

    if not out_path.exists():
        volume = load_h5_file(h5_path)
        img = get_slice_image(volume, index)
        img.save(out_path)

    return FileResponse(out_path, media_type="image/png")

@app.get("/mask/{file_id}/{index}")
def get_mask_image_api(file_id: str, index: int):
    h5_path = UPLOAD_DIR / f"{file_id}.h5"
    if not h5_path.exists():
        raise HTTPException(status_code=404, detail="MRI file not found")
    volume = load_h5_file(h5_path)
    slice_np = volume[index]  # (4, 240, 240)
    mask = predict_mask(seg_model, slice_np)  # (240, 240)

    #RED OVERLAY
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = 255  # Red channel
    rgba[..., 3] = (mask * 180).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


@app.get("/report/{file_id}")
def download_report(file_id: str):
    
    h5_path = UPLOAD_DIR / f"{file_id}.h5"
    if not h5_path.exists():
        h5_path = Path("testing") / f"{file_id}.h5"
        if not h5_path.exists():
            raise HTTPException(status_code=404, detail="MRI file not found")

    volume = load_h5_file(h5_path)
    num_slices = volume.shape[0]

    
    region_anomaly = {
        "Cerebellum/Brainstem": False,
        "Occipital Lobe": False,
        "Temporal Lobe": False,
        "Parietal Lobe": False,
        "Frontal Lobe": False
    }
    for idx in range(volume.shape[0]):
        slice_np = volume[idx]
        mask = predict_mask(seg_model, slice_np)
        if np.count_nonzero(mask) > 10:
            region = get_region_for_slice(idx)
            if region:
                region_anomaly[region] = True

    # Generate PDF in memory
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 770, "Brain MRI Analysis Report")
    c.setFont("Helvetica", 10)
    c.drawString(400, 770, f"Date: {now}")
    c.drawString(400, 755, "City: Passau")
    c.setFont("Helvetica", 12)
    c.drawString(72, 730, f"File ID: {file_id}")
    c.drawString(72, 710, f"Number of Slices: {num_slices}")
    c.drawString(72, 690, "Region Findings:")
    y = 670
    for region, is_anom in region_anomaly.items():
        status = "Anomaly" if is_anom else "Normal"
        color = (1, 0, 0) if is_anom else (0, 0.5, 0)
        c.setFillColorRGB(*color)
        c.drawString(100, y, f"{region}: {status}")
        y -= 20
    c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=report_{file_id}.pdf"
    })