import { useState } from 'react';
import styles from './App.module.css';
import profileImg from './assets/profile.jpg';
import Viewer from './Viewer.jsx';
import brainLogo from './assets/brain.svg';




export default function App() {
  const DEMO_FILE_IDS = [
    "volume_106_full", "volume_114_full", "volume_145_full", "volume_151_full"
  ];


  const DEMO_REGION_ANOMALY = {
    "volume_106_full": {
      "Cerebellum/Brainstem": false,
      "Occipital Lobe": true,
      "Temporal Lobe": true,
      "Parietal Lobe": true,
      "Frontal Lobe": true
    },
    "volume_114_full": {
      "Cerebellum/Brainstem": true,
      "Occipital Lobe": true,
      "Temporal Lobe": true,
      "Parietal Lobe": true,
      "Frontal Lobe": true
    },
    "volume_145_full": {
      "Cerebellum/Brainstem": true,
      "Occipital Lobe": true,
      "Temporal Lobe": true,
      "Parietal Lobe": true,
      "Frontal Lobe": true
    },
    "volume_151_full": {
      "Cerebellum/Brainstem": true,
      "Occipital Lobe": true,
      "Temporal Lobe": true,
      "Parietal Lobe": true,
      "Frontal Lobe": true
    },

    
  };
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadText, setUploadText] = useState('');
  const [selectedLang, setSelectedLang] = useState("English");
  const [showMenu, setShowMenu] = useState(false);
  const [showViewer, setShowViewer] = useState(false);

  // Store backend data
  const [fileId, setFileId] = useState(null);
  const [numSlices, setNumSlices] = useState(0);
  const [labels, setLabels] = useState([]);
  const [scores, setScores] = useState([]);
  const [regionAnomaly, setRegionAnomaly] = useState({});

  // Handle demo selection
  const handleDemoSelect = (e) => {
    const demoId = e.target.value;
    if (demoId) {
      setFileId(demoId);
      setNumSlices(155);
      setLabels([]);
      setScores([]);
      setRegionAnomaly(DEMO_REGION_ANOMALY[demoId] || {});
      setShowViewer(true);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setUploadText("Uploading...");

    try {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "http://localhost:8000/upload", true);

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded * 100) / event.total);
          setUploadProgress(progress);
        }
      };

      xhr.onload = () => {
        if (xhr.status === 200) {
          setUploadText("File Uploaded ✅");
          // Parse backend response
          const data = JSON.parse(xhr.responseText);
          setFileId(data.file_id);
          setNumSlices(data.num_slices);
          setLabels(data.labels);
          setScores(data.scores);
          setRegionAnomaly(data.region_anomaly);

          setTimeout(() => {
            setShowViewer(true);
          }, 1000);
        } else {
          // Show backend error message
          let msg = "Upload failed.";
          try {
            const data = JSON.parse(xhr.responseText);
            if (data.detail) msg = data.detail;
          } catch {}
          alert(msg);
          setUploadText(""); // Reset upload text on error
          setUploadProgress(0);
        }
      };

      xhr.send(formData);
    } catch (error) {
      alert("Upload failed.");
      setUploadText("");
      setUploadProgress(0);
      console.error(error);
    }
  };

  return (
    <div className={styles.wrapper}>
      {/* Top Navbar */}
      <header className={styles.navbar}>
        <button className={styles.logo} onClick={() => setShowViewer(false)}>
          <img src={brainLogo} alt="Brain Logo" style={{ height: "28px", verticalAlign: "middle", marginRight: "0.5rem" }} />
          BrainGPT
        </button>
        <div className={styles.navRight}>
          <select
            className={styles.languageSwitcher}
            value={selectedLang}
            onChange={(e) => setSelectedLang(e.target.value)}
          >
            <option>English</option>
            <option>Deutsch</option>
            <option>Français</option>
          </select>
          <div
            className={styles.profileWrapper}
            onClick={() => setShowMenu((prev) => !prev)}
          >
            <img
              className={styles.profileImage}
              src={profileImg}
              alt="Profile"
            />
            {showMenu && (
              <div className={styles.profileDropdown}>
                <div>Profile</div>
                <div>Settings</div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Page Body */}
      {showViewer ? (
        <Viewer
          onBack={() => setShowViewer(false)}
          fileId={fileId}
          numSlices={numSlices}
          labels={labels}
          scores={scores}
          regionAnomaly={regionAnomaly}
        />
      ) : (
        <main className={styles.pageLayout}>
          <h1 className={styles.uploadTitle}>Explore Brain MRI Slices</h1>
          {/* Demo selector */}
          <div style={{ marginBottom: "1.5rem" }}>
            <select className={styles.demoDropdown} onChange={handleDemoSelect} defaultValue="">
              <option value="">Select Demo Volume</option>
              {DEMO_FILE_IDS.map(id => (
                <option key={id} value={id}>{id}</option>
              ))}
            </select>
          </div>
          <label className={styles.uploadButton}>
            Upload Full Brain Volume (.h5)
            <input
              type="file"
              accept=".h5"
              onChange={handleUpload}
              style={{ display: "none" }}
            />
          </label>
          <div className={styles.uploadSection}>
            {!!uploadText && (
              <div className={styles.progressStatus}>{uploadText}</div>
            )}
            <div className={styles.progressWrapper}>
              <div className={styles.progressBar}>
                <div
                  className={styles.progressValue}
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p style={{ marginTop: "0.5rem" }}>{uploadProgress}%</p>
            </div>
            <p className={styles.supported}>
              Supported format: <code>.h5</code> • <b>Must contain 155 slices</b> • Max size: 100MB<br/>
              <b>Upload a single file containing the full brain scan.</b>
            </p>
          </div>
        </main>
      )}
    </div>
  );
}