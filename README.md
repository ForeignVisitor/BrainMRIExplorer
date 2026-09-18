# 🧠 BrainMRIExplorer

**BrainMRIExplorer** is a visual computing application for interactive exploration and AI-based anomaly detection in brain MRI scans.

It features slice navigation, multi-modality viewing, deep learning-based anomaly segmentation, region-based findings, and PDF report generation.

To have a look at our used Dataset, check out the link: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/data

---

## 🚀 Features

- Upload and explore full brain MRI volumes (`.h5`)
- Supports 4 MRI modalities: **T1**, **T2**, **FLAIR**, **T1ce**
- AI-based anomaly detection and segmentation (U-Net, CNN)
- Interactive slice navigation & contrast adjustment.
- Region-based findings (frontal, parietal, temporal, occipital, cerebellum/brainstem)
- Overlay of segmentation mask on MRI
- Precomputed demo volumes for instant demo
- Downloadable PDF report with findings

---

## 📁 Folder Structure

```
BrainMRIExplorer/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── src/
│   │   ├── h5_helper.py
│   │   ├── predict.py
│   │   ├── segment.py
│   │   └── unet.py
│   ├── models/
│   │   ├── simplecnn_4ch_epoch5.pth
│   │   └── unet_brats_h5_best.pth
│   ├── uploads/           # (empty, for uploads)
│   ├── slices/            # (empty, for generated images)
│   ├── precomputed/       # (contains precomputed demo PNGs)
│   └── testing/           # (contains demo .h5 volumes)
├── frontend/
│   ├── package.json
│   ├── public/
│   └── src/
│       ├── App.jsx
│       ├── Viewer.jsx
│       ├── App.module.css
│       ├── Viewer.module.css
│       └── assets/
│           ├── brain.svg
│           └── profile.jpg
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Backend (FastAPI)

- **Python 3.8+ required**

#### Install dependencies:

```bash
cd backend
python -m venv venv
# Activate venv:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

#### Start the backend:

```bash
uvicorn main:app --reload
```

The backend will run at: **http://localhost:8000**

---

### 2. Frontend (React)

- **Node.js 16+ recommended**

#### Install dependencies:

```bash
cd frontend
npm install
```

#### Start the frontend:

```bash
npm run dev
```

The frontend will run at: **http://localhost:5173**

---

### 3. Extra folders

Follow the link https://drive.google.com/file/d/1R7nny3jZ73KiY4kJPrDG-O6sZaLv04j7/view?usp=sharing and download the folder named "extract_me_in_backend.rar". Then extract it and make sure its contents are in the /backend/ directory as the name states.

## 🧪 Usage

1. Upload a full brain MRI volume (`.h5`), you can check backend/data/combined_volumes for examples of data
2. Or select a demo volume from the dropdown for instant demo
3. Navigate through slices and adjust contrast as desired
4. View AI-detected anomalies and region-based findings
5. Download a PDF report of the analysis

---

## 💡 Notes

- **Demo volumes:** Place `.h5` demo volumes in `backend/testing/`
- **Precomputed PNGs:** Save in `backend/precomputed/` with the format:
  - `volumeID_sliceIndex.png`
  - `volumeID_sliceIndex_mask.png`
- **Models:** Store your trained models in `backend/models/`
- No large datasets are included — only demo/test files are provided for demonstration.

---

## 🛠️ Troubleshooting

- **MRI file not found?** Ensure the `.h5` file exists in `backend/testing/`.
- **Slow image loading?** Use precomputed PNGs for demo volumes.
- **Python errors?** Ensure your virtual environment is activated and dependencies are correctly installed.

---

## 📬 Contact

Follow this link to get to our project Repo: https://github.com/ForeignVisitor/BrainMRIExplorer

For any questions, please contact **[Souhail Karam]**

📧 [souhailkaram.studies@gmail.com]
