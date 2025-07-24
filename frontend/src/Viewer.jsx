import styles from './Viewer.module.css';
import { useState, useEffect } from 'react';

// List of demo fileIds
const DEMO_FILE_IDS = [
  "volume_106_full", "volume_114_full", "volume_145_full", "volume_151_full"

];

export default function Viewer({ onBack, fileId, numSlices, labels, scores, regionAnomaly = {} }) {
  const regionSliceMap = {
    "Cerebellum/Brainstem": [0, 24],
    "Occipital Lobe": [25, 49],
    "Temporal Lobe": [50, 79],
    "Parietal Lobe": [80, 114],
    "Frontal Lobe": [115, 154]
  };

  const brainRegions = Object.keys(regionSliceMap);

  const [contrast, setContrast] = useState(100);
  const [selectedRegion, setSelectedRegion] = useState('');
  const [sliceIdx, setSliceIdx] = useState(0);
  const [regionRange, setRegionRange] = useState([0, numSlices - 1]);
  const [maskUrl, setMaskUrl] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  
  const handleDownloadReport = () => {
    if (!fileId) return;
    fetch(`http://localhost:8000/report/${fileId}`)
      .then(response => response.blob())
      .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${fileId}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      });
  };

  function handleRegionSelect(region) {
    setSelectedRegion(region);
    if (regionSliceMap[region]) {
      const [start, end] = regionSliceMap[region];
      setRegionRange([start, end]);
      setSliceIdx(Math.floor((start + end) / 2));
    } else {
      setRegionRange([0, numSlices - 1]);
      setSliceIdx(0);
    }
  }

  // Detect if this is a demo volume
  const isDemo = fileId && DEMO_FILE_IDS.includes(fileId);



  console.log("fileId:", fileId, "isDemo:", isDemo);

  useEffect(() => {
    if (!fileId) return;
    if (isDemo) {
      setImageUrl(`http://localhost:8000/precomputed/${fileId}_${sliceIdx}.png`);
      setMaskUrl(`http://localhost:8000/precomputed/${fileId}_${sliceIdx}_mask.png`);
    } else {
      setImageUrl(`http://localhost:8000/slice/${fileId}/${sliceIdx}`);
      setMaskUrl(`http://localhost:8000/mask/${fileId}/${sliceIdx}`);
    }
  }, [fileId, sliceIdx, isDemo]);

  return (
    <div className={styles.viewerWrapper}>
      <div className={styles.content}>
        <div className={styles.leftPanel}>
          <div className={styles.centeredTitle}>Analysis Results</div>

          {/* Overlay MRI and mask */}
          <div className={styles.imageOverlayWrapper}>
            {imageUrl &&
              <img
                src={imageUrl}
                className={styles.image}
                alt="MRI Slice"
                style={{ filter: `contrast(${contrast}%)` }}
              />
            }
            {maskUrl &&
              <img
                src={maskUrl}
                className={styles.maskImage}
                alt="Segmentation Mask"
              />
            }
          </div>

          <div className={styles.sliderGroup}>
            <div className={styles.sliderSection}>
              <label className={styles.sliderMainLabel}>
                🧭 Slice Navigation
                <span className={styles.sliderSubLabel}>
                  {selectedRegion && regionSliceMap[selectedRegion]
                    ? `Navigate within ${selectedRegion} (Slice ${sliceIdx + 1} / ${numSlices})`
                    : `Move through the MRI scan (Slice ${sliceIdx + 1} / ${numSlices})`}
                </span>
              </label>
              <input
                type="range"
                min={regionRange[0]}
                max={regionRange[1]}
                value={sliceIdx}
                onChange={e => setSliceIdx(Number(e.target.value))}
                className={`${styles.slider} ${styles.sliceSlider}`}
              />
            </div>
            <div className={styles.sliderSection}>
              <label className={styles.sliderMainLabel}>
                🌈 Adjust Contrast
                <span className={styles.sliderSubLabel}>
                  Enhance image visibility ({contrast}%)
                </span>
              </label>
              <input
                type="range"
                min="50"
                max="200"
                value={contrast}
                onChange={e => setContrast(e.target.value)}
                className={`${styles.slider} ${styles.contrastSlider}`}
              />
            </div>
          </div>

          <div className={styles.dropdownContainer}>
            <select
              className={styles.dropdown}
              value={selectedRegion}
              onChange={e => handleRegionSelect(e.target.value)}
            >
              <option value="">Select Brain Area</option>
              {brainRegions.map(region => (
                <option key={region}>{region}</option>
              ))}
            </select>
          </div>
        </div>

        <div className={styles.rightPanel}>
          <h3 className={styles.findingsTitle}>Key Findings</h3>
          <ul className={styles.findingsList}>
            {brainRegions.map((region, i) => (
              <li key={i}>
                <span className={styles.findingRegion}>{region}</span>
                <span className={regionAnomaly[region] ? styles.bad : styles.good}>
                  {regionAnomaly[region] ? `🚨 Anomaly` : '✅ Normal'}
                </span>
              </li>
            ))}
          </ul>
          <button className={styles.downloadBtn} onClick={handleDownloadReport}>
            Download Report
          </button>
        </div>
      </div>
    </div>
  );
}