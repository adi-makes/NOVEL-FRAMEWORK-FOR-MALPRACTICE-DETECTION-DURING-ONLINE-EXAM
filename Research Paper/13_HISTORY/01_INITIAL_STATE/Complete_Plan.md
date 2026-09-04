

Exam proctoring project plan · MD
# Three-Stream Exam Proctoring System: Complete Project Plan
 
**Project Title:** Explainable Multi-Modal Exam Proctoring via Synchronized Gaze, Interaction, and Environment Sensing  
**Duration:** 12-18 months  
**Target Publication:** CHI, IEEE TMM, or ACM CCS (FAccT workshop phase-in)  
 
---
 
## PART I: PROJECT IDEOLOGY & VISION
 
### 1.1 Core Philosophy
 
**What problem are we solving?**
- Academic integrity depends on trust, not surveillance.
- Current proctoring is either **invasive** (full-screen monitoring, lockdown) or **insufficient** (honor codes).
- Our system aims for **calibrated risk scoring**, not binary cheating detection.
- Goal: **detect suspicious behavioral patterns, flag for human review, respect privacy.**
**What makes this different?**
- Not "catch cheaters." → **"Help educators make informed decisions."**
- Not "one camera sees all." → **"Multiple sensors reduce false positives."**
- Not "AI decides guilt." → **"AI surfaces evidence, humans decide."**
### 1.2 Core Values
 
| Value | What it means | Implementation |
|-------|---------------|-----------------|
| **Explainability** | Every alert has a reason | Store feature scores; visualize why alert fired |
| **Fairness** | Don't penalize accessibility needs | Test on glasses, assistive tech, neurodivergence |
| **Privacy** | Minimize invasiveness | Local processing, consent, data deletion timeline |
| **Reproducibility** | Others can verify our work | Release code + dataset, publish ablations |
| **Cross-context** | Works in dorms AND exam halls | Test remote + physical settings |
 
### 1.3 Ethical Guardrails
 
**What we will NOT do:**
- Continuous tracking or background monitoring
- Biometric identification without consent
- Punishment-oriented design (students shouldn't fear the system)
- Claim 100% accuracy (explicitness: "87% specificity at 95% sensitivity")
**What we WILL do:**
- Publish false-positive rates by demographic and disability status
- Include a "human review" UI for proctors
- Design for appeal (students see what flagged them)
- Assume educators use the system responsibly
---
 
## PART II: PROJECT PHASES & TIMELINE
 
### Phase 0: Foundation & Planning (Months 1-2)
**Goal:** Build the technical and organizational foundation.
 
#### 0.1 Team Assembly
**Roles needed:**
- **Project Lead** (you?): overall vision, paper writing, vendor/ethics coordination
- **Vision/Gaze Lead:** gaze estimation, head-pose tracking, calibration
- **Interaction/Behavior Lead:** mouse analytics, keystroke modeling, telemetry
- **Vision/Environment Lead:** camera placement, object detection, spatial reasoning
- **Data & ML Lead:** fusion architecture, training, evaluation, ablations
- **Ethics & User Research Lead:** fairness testing, accessibility, consent
- **Systems Lead:** dataset infrastructure, synchronization, real-time processing
**Team size:** Minimum 4 (you + 3 collaborators); ideal 6-8 if affiliated with a university or lab.
 
#### 0.2 Environment Setup
```bash
# Create project repository
mkdir -p exam-proctoring-research
cd exam-proctoring-research
 
# Folder structure
mkdir -p {docs, code/{gaze,mouse,env_camera,fusion}, data/{raw,processed,annotations}, models, results, notebooks, ethics}
 
# Version control
git init
git remote add origin <your-repo>
 
# Python environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)
 
# Core dependencies (will expand)
pip install opencv-python numpy pandas scikit-learn torch torchvision \
    mediapipe pytorch-lightning wandb pytest black flake8
```
 
#### 0.3 Literature Baseline
**Action:**
- Assign each team member 3-4 papers from the review
- Create a **shared annotation document** (Google Doc or Notion):
  - Paper → Method → Results → Code availability → License
  - Flag papers with available code or datasets
- **Priority papers to replicate:**
  1. Hu et al. (2024) — three-camera gaze+object fusion
  2. Li et al. (2021) — video + mouse analytics
  3. Atoum et al. (2017) — multimodal baseline
#### 0.4 Ethics & IRB Approval
**Action:**
- **Contact your institution's IRB (Institutional Review Board).**
- **IRB Protocol should cover:**
  - Participant consent (video recording, data retention)
  - Data anonymization (face blurring, eye-gaze privacy)
  - Sensitivity of "cheating detection" (psychological risk)
  - Data deletion schedule (30 days after study? 1 year?)
  - Right to withdraw + data destruction
  
- **Estimated timeline:** 4-8 weeks (start early!)
- **Document templates:** Consent form, recruitment flyer, data-handling procedure
---
 
### Phase 1: Single-Modality Baselines (Months 3-5)
**Goal:** Validate each stream independently before fusion.
 
#### 1.1 Gaze Estimation Module
 
**Deliverable:** A working gaze estimator that runs in real-time.
 
**Step-by-step:**
 
1. **Choose your gaze backbone:**
   - Option A (Lightweight, open-source): **MPGaze** (MediaPipe) + calibration
   - Option B (Accurate, needs training): **iTracker** (PyTorch from Kachurovskiy et al.)
   - Option C (Pre-trained, cloud): **Google ML Kit Gaze** (but check privacy)
   
   **Recommendation:** Start with **MediaPipe** (free, open-source, 4ms inference on CPU), then compare to iTracker if budget allows.
2. **Calibration protocol:**
```python
   # Pseudo-code: collect calibration points
   # Display 9-point grid on screen (corners, edges, center)
   # For each point, record:
   #   - Screen coordinates (ground truth)
   #   - Webcam frame + detected gaze direction
   # Fit polynomial or neural network to map sensor → screen
   
   calibration_points = 9  # 3x3 grid
   samples_per_point = 30  # seconds of looking
   
   # Output: calibration matrix
   # Later: apply to test gaze data
```
 
3. **Metrics to track:**
   - **Calibration error** (pixels): std of error after fitting
   - **Gaze estimation FPS** (frames/sec)
   - **Head-pose robustness** (yaw, pitch, roll tolerance ±30°)
   - **Lighting tolerance** (works in dark? bright? side light?)
4. **Validation data:**
   - Collect from 5 volunteers
   - Each: 5-min webcam recording while reading a document
   - Manual annotation: where is the person looking? (5-10 frame samples)
   - Compute accuracy: "Did the system predict the correct region?"
5. **Code structure:**
```
   code/gaze/
   ├── mediapipe_baseline.py      # Load MPGaze model
   ├── calibration.py              # Run 9-point calibration
   ├── gaze_estimator.py           # Real-time inference
   ├── metrics.py                  # Compute accuracy, error
   └── test_gaze.py                # Unit tests
```
 
**Output by Month 5:**
- [ ] Gaze estimation running at 25+ FPS
- [ ] Calibration error <50 pixels on 1080p screen
- [ ] Test accuracy on 5 volunteers: >85% region classification
- [ ] Documented limitations (glasses? contact lenses? dark eyes?)
---
 
#### 1.2 Mouse/Interaction Behavior Module
 
**Deliverable:** Behavioral feature extraction from mouse events.
 
**Step-by-step:**
 
1. **Data collection design:**
```python
   # What to log during an exam (with student consent):
   mouse_event = {
       "timestamp": 1693524601.234,
       "x": 500,
       "y": 400,
       "event": "move",  # "move", "click", "scroll", "idle"
       "duration_ms": None,  # For idle events
   }
   
   window_event = {
       "timestamp": 1693524601.234,
       "window_title": "Google Chrome",
       "is_focused": True,
       "url": "example.com/exam",  # Only domain, not full URL
   }
   
   keystroke_event = {
       "timestamp": 1693524601.234,
       "key": "letter",  # "letter", "backspace", "arrow", "special"
       "duration_press_ms": 45,
   }
```
 
2. **Feature engineering** (compute from events over a 10-second window):
   
   | Feature | Calculation | Intuition |
   |---------|-------------|-----------|
   | **Cursor velocity (px/s)** | Euclidean distance / time | Rapid movement = unusual |
   | **Cursor acceleration** | Δ velocity / Δ time | Jerky movement = bot/replay? |
   | **Click frequency** | # clicks / 10s | Excessive clicks = nervous/searching? |
   | **Idle duration** | Time since last move | Reading vs. active typing |
   | **Tab switches** | # window focus changes | Leaving exam window = suspicious |
   | **Keystroke velocity** | keys typed per second | Speed pattern (biometric baseline) |
   | **Typing entropy** | Shannon entropy of intervals | Bursts vs. steady typing |
   | **Dwell time** | Time cursor stays in one area | Focus on input box vs. wandering |
   
```python
   def extract_mouse_features(events, window_size=10.0):
       """Extract 10-second rolling features from mouse events."""
       features = {}
       
       # Velocity
       displacements = np.sqrt(np.diff(events.x)**2 + np.diff(events.y)**2)
       durations = np.diff(events.timestamp)
       features['velocity_mean'] = np.mean(displacements / durations)
       features['velocity_std'] = np.std(displacements / durations)
       
       # Click frequency
       features['click_count'] = sum(1 for e in events if e.event == "click")
       
       # Idle time
       features['idle_fraction'] = sum(e.duration_ms for e in events if e.event == "idle") / (window_size * 1000)
       
       # Tab switches
       features['tab_switch_count'] = count_window_focus_changes(events)
       
       return features
```
 
3. **Baseline models:**
   - **Autoencoder (Unsupervised):** Train on "normal" exam behavior; flag anomalies
   - **Random Forest (Semi-supervised):** Mix labeled cheating + unlabeled normal data
   - **One-Class SVM:** Standard anomaly detection approach
   
   **Recommendation:** Start with Random Forest (interpretable) + autoencoder (no labels needed).
4. **Validation data:**
   - Collect 20 sessions: 10 honest exams, 10 with simulated cheating
   - Simulated cheating: "Google an answer," "switch apps," "rapid clicking"
   - Metrics: Precision, recall, ROC-AUC
5. **Code structure:**
```
   code/mouse/
   ├── mouse_logger.py             # Background daemon to log events
   ├── feature_extractor.py        # Compute features from events
   ├── models.py                   # RF, autoencoder, OCSVM
   ├── train.py                    # Train on labeled data
   ├── inference.py                # Real-time scoring
   └── test_mouse.py               # Unit tests
```
 
**Output by Month 5:**
- [ ] Mouse logger running with <1% CPU overhead
- [ ] 20+ behavioral features extracted
- [ ] Baseline model achieving >80% detection on simulated cheating
- [ ] Documented false-positive rate on benign behavior (reading, scrolling, thinking)
---
 
#### 1.3 Environment Camera & Object Detection Module
 
**Deliverable:** Real-time detection of phones, notes, extra people in 3rd-camera view.
 
**Step-by-step:**
 
1. **Camera hardware specification:**
```
   Camera Setup:
   - Primary (Face): USB webcam, 1080p, 60 FPS, front-facing
     Purpose: Gaze estimation
     
   - Secondary (Environment): USB webcam, 1080p, 30 FPS, 
     Position: Side-mounted 60° angle (desk corner)
     Purpose: Detect phone, notes, people, posture
     
   - Synchronization: NTP-aligned software timestamps
```
 
2. **Object detection choice:**
   - **Option A (Fast):** YOLOv8-nano (6M params, 15 FPS on CPU)
   - **Option B (Accurate):** YOLOv8-medium (26M params, 5 FPS on CPU)
   - **Option C (Balanced, Recommended):** YOLOv8-small (11M params, 10 FPS on CPU)
   
   **Model training:**
```python
   from ultralytics import YOLO
   
   # Load pretrained
   model = YOLO("yolov8s.pt")
   
   # Classes to detect
   classes = {
       0: "phone",
       1: "notebook",
       2: "textbook",
       3: "person",
       4: "suspicious_item"  # Custom: hard copy of exam?
   }
   
   # Fine-tune on custom dataset
   results = model.train(
       data="path/to/dataset.yaml",
       epochs=50,
       imgsz=640,
       batch=16,
       device=0,  # GPU
   )
```
 
3. **Custom dataset creation:**
   - **Collect:** 500+ images from the secondary camera angle
   - **Annotation tool:** Use Label Studio or CVAT (free, open-source)
   - **Classes:** Phone (any visible), notebook, textbook, person (head+torso), suspicious (exam copy)
   - **Bounding boxes:** Annotate each object
   - **Split:** 70% train, 15% val, 15% test
   
```
   dataset/
   ├── images/
   │   ├── train/ (350 images)
   │   ├── val/   (75 images)
   │   └── test/  (75 images)
   ├── labels/
   │   ├── train/ (YOLO .txt format)
   │   └── ...
   └── dataset.yaml (metadata)
```
 
4. **Features from detections** (per 10-second window):
```python
   env_features = {
       "phone_detected": bool,
       "phone_confidence": float,  # 0-1
       "phone_distance_to_exam": float,  # pixels from screen
       "extra_person_in_frame": bool,
       "notes_detected": bool,
       "suspicious_items_count": int,
   }
```
 
5. **Code structure:**
```
   code/env_camera/
   ├── camera_controller.py         # Open camera, capture frames
   ├── dataset_tools.py             # Label Studio integration
   ├── yolo_trainer.py              # Train YOLOv8 custom model
   ├── detector.py                  # Real-time inference
   ├── spatial_analysis.py          # Compute distance, position
   └── test_detection.py            # Unit tests
```
 
**Output by Month 5:**
- [ ] Secondary camera mounted and synchronized with primary
- [ ] YOLOv8 model trained on custom dataset
- [ ] Detection running at 10+ FPS
- [ ] Test accuracy >90% on custom test set (phone, notebook, person)
- [ ] Documented false-positive rate (shadow ≠ phone)
---
 
### Phase 2: Fusion Architecture & Dataset (Months 6-9)
**Goal:** Combine the three streams and collect multimodal synchronized data.
 
#### 2.1 Fusion Architecture Design
 
**Deliverable:** A learned model that combines gaze, mouse, and environment signals.
 
**Step 1: Data format definition**
 
Each 10-second exam window contains:
 
```python
exam_window = {
    "timestamp_start": 1693524601,
    "timestamp_end": 1693524611,
    "duration_sec": 10,
    
    # Stream 1: Gaze
    "gaze": {
        "estimated_screen_x": [100, 120, 115, ...],  # 10-100 Hz, shape (N,)
        "estimated_screen_y": [250, 260, 255, ...],
        "confidence": [0.95, 0.93, 0.91, ...],
        "head_yaw": [-5, -3, -1, ...],  # degrees
        "head_pitch": [0, 2, 1, ...],
        "aggregated_features": {  # Pre-computed
            "fixation_duration_mean": 1.2,
            "fixation_count": 5,
            "saccade_velocity_mean": 300,
        }
    },
    
    # Stream 2: Mouse & Interaction
    "interaction": {
        "cursor_x": [500, 510, 515, ...],
        "cursor_y": [400, 405, 410, ...],
        "click_events": [{"t": 2.1, "x": 500, "y": 400}, ...],
        "window_focus_changes": 1,
        "aggregated_features": {
            "velocity_mean": 150,
            "velocity_std": 45,
            "click_frequency": 2,
            "idle_fraction": 0.3,
        }
    },
    
    # Stream 3: Environment Camera
    "environment": {
        "frame": np.array(...),  # 1080p image
        "detections": [
            {"class": "phone", "confidence": 0.92, "bbox": [100, 100, 150, 180]},
            {"class": "person", "confidence": 0.88, "bbox": [50, 20, 600, 550]},
        ],
        "aggregated_features": {
            "phone_detected": True,
            "phone_confidence": 0.92,
            "extra_person_detected": False,
            "notes_detected": False,
        }
    },
    
    # Ground truth (for labeled data)
    "label": "honest",  # or "cheating" or "uncertain"
    "cheating_type": None,  # "phone", "copy_paste", "eye_movement", etc.
    "annotator_notes": "Student looked briefly at phone, returned to exam.",
}
```
 
**Step 2: Choose fusion strategy**
 
```python
import torch
import torch.nn as nn
 
class ThreeStreamFusionModel(nn.Module):
    """Fuse gaze, mouse, and environment streams."""
    
    def __init__(self, gaze_dim=8, mouse_dim=10, env_dim=6, hidden=128):
        super().__init__()
        
        # Stream-specific encoders
        self.gaze_encoder = nn.Sequential(
            nn.Linear(gaze_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.mouse_encoder = nn.Sequential(
            nn.Linear(mouse_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Attention-based fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=4,
            batch_first=True,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # Single score (0-1)
        )
        
    def forward(self, gaze, mouse, env):
        # Encode each stream
        g = self.gaze_encoder(gaze)      # (B, hidden)
        m = self.mouse_encoder(mouse)    # (B, hidden)
        e = self.env_encoder(env)        # (B, hidden)
        
        # Stack for attention
        x = torch.stack([g, m, e], dim=1)  # (B, 3, hidden)
        
        # Self-attention fusion
        attended, _ = self.attention(x, x, x)  # (B, 3, hidden)
        
        # Flatten and classify
        fused = attended.flatten(1)  # (B, 3*hidden)
        score = torch.sigmoid(self.classifier(fused))  # (B, 1)
        
        return score
 
# Loss function: binary cross-entropy with class weighting
# (account for imbalanced cheating/honest ratio)
criterion = nn.BCELoss(weight=torch.tensor([1.0, 2.0]))
```
 
**Three fusion strategies to compare:**
 
| Strategy | Architecture | Pros | Cons |
|----------|--------------|------|------|
| **Early Fusion** | Concatenate [gaze, mouse, env] → FC layers | Simple, fast | Ignores stream importance |
| **Late Fusion** | Each stream → FC → logits → average | Interpretable | Misses cross-stream patterns |
| **Attention Fusion** | MultiheadAttention + cross-stream gates | Learns stream weights | Complex, needs tuning |
 
**Recommendation:** Implement all three; report ablations.
 
**Step 3: Training pipeline**
 
```python
# Pseudo-code
import pytorch_lightning as pl
 
class ProctorModel(pl.LightningModule):
    def __init__(self, model_type="attention"):
        super().__init__()
        self.model = ThreeStreamFusionModel()
        self.criterion = nn.BCELoss()
        
    def training_step(self, batch, batch_idx):
        gaze, mouse, env, label = batch
        logits = self.model(gaze, mouse, env)
        loss = self.criterion(logits, label)
        return loss
    
    def validation_step(self, batch, batch_idx):
        gaze, mouse, env, label = batch
        logits = self.model(gaze, mouse, env)
        loss = self.criterion(logits, label)
        auc = roc_auc_score(label, logits)
        self.log("val_auc", auc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
 
# Train
trainer = pl.Trainer(max_epochs=50, gpus=1)
trainer.fit(model, train_loader, val_loader)
```
 
**Code structure:**
```
code/fusion/
├── data_loader.py               # Load exam_window dicts
├── model.py                     # ThreeStreamFusionModel
├── train.py                     # Training loop (PyTorch Lightning)
├── evaluate.py                  # Metrics, ablations
├── visualize.py                 # Attention weights, feature importance
└── test_fusion.py               # Unit tests
```
 
**Output by Month 7:**
- [ ] Fusion model architecture documented + code
- [ ] Trained on dummy data; loss decreases
- [ ] Ablation study plan (gaze-only, mouse-only, env-only, pairs, all-three)
---
 
#### 2.2 Dataset Collection Protocol
 
**Deliverable:** 100+ synchronized exam sessions with annotations.
 
**Overall design:**
- **Remote exams only** (for Phase 2; CCTV comes later)
- **Real or realistic mock exams** (5-15 min duration)
- **Diverse participants:** 30+ volunteers, mix of ages, glasses, lighting
- **Balanced labels:** ~50% honest, ~50% simulated cheating
**Participants:**
- Recruit from university (students, staff) or online (Prolific, MTurk-style)
- **IRB approval required** (get this before collecting!)
- **Compensation:** $15-25 per session (30-45 minutes)
- **Inclusion criteria:** >18 years, fluent in exam language, no vision impairment (or adaptive tech OK)
**Hardware setup:**
```
Participant's desk:
├── Primary camera (face): mounted on monitor bezel, 60cm away
├── Secondary camera (environment): on desk corner, angled 60°
├── Keyboard, mouse (standard), monitor
├── Audio recording (ambient; check local laws)
└── Screen-capture software (exam interface only, no OS)
```
 
**Exam content:**
- **Honest sessions:** Real 10-question multiple-choice quiz (math, history, etc.)
- **Cheating sessions:** Same quiz, but instruct student to:
  - Session A: "Use Google to look up 2 answers" (phone/Google search)
  - Session B: "Copy-paste from notes" (pre-written cheat sheet on phone)
  - Session C: "Have a friend whisper answers" (confederate audio; capture in mic)
  - Session D: "Look at textbook while answering 3 questions"
  
  **Important:** All "cheating" is simulated and consented; no actual academic integrity violation.
**Ground truth annotation:**
```
For each 10-second window:
├── Label: "honest" or "cheating"
├── Cheating type: "none", "phone_use", "note_lookup", "external_help", etc.
├── Confidence: 1.0 (definite) to 0.5 (uncertain)
└── Notes: "Student glanced at phone, then continued exam."
```
 
**Annotation process:**
1. Video review by 2 independent annotators (blind to labels)
2. Disagreements resolved by 3rd reviewer or majority vote
3. Inter-rater reliability: Cohen's kappa >0.75 required
4. Use Label Studio + custom backend for efficiency
**Data collection timeline:**
 
| Weeks | Tasks |
|-------|-------|
| 6-8 | Recruit 30 volunteers; finalize IRB |
| 8-10 | Pilot test (5 sessions); refine setup |
| 10-15 | Main collection (100+ sessions, 3 sessions per participant) |
| 15-16 | Initial annotation (50 sessions double-checked) |
| 16-18 | Final annotation + quality control |
 
**Code structure:**
```
code/data_collection/
├── session_manager.py           # Orchestrate camera, logging, screen
├── exam_ui.py                   # Simple web-based exam interface
├── synchronization.py           # Timestamp alignment across streams
├── data_validator.py            # Check completeness, sync quality
└── annotation_tools.py          # Integrate with Label Studio API
 
data/
├── raw/
│   ├── session_001/
│   │   ├── face_camera.mp4
│   │   ├── env_camera.mp4
│   │   ├── screen_capture.mp4
│   │   ├── mouse_events.jsonl
│   │   ├── keyboard_events.jsonl
│   │   ├── gaze_estimates.csv
│   │   └── metadata.json
│   └── session_002/ ...
├── processed/
│   ├── exam_windows.pkl         # List of exam_window dicts
│   └── splits.json              # Train/val/test assignments
└── annotations/
    ├── session_001_labels.json   # Annotator judgments
    └── agreement_report.md       # Inter-rater kappa
```
 
**Data privacy & retention:**
- **On-disk:** Encrypt with AES-256
- **Cloud backup:** Encrypted AWS S3, access-controlled
- **Deletion:** 30 days after study end (or per consent form)
- **De-identification:** Blur faces in public release; remove names
- **Consent:** Explicit checkbox for dataset publication (anonymized)
**Output by Month 9:**
- [ ] 100+ synchronized exam sessions collected
- [ ] Ground truth annotations with kappa >0.75
- [ ] Dataset split: 60 train, 20 val, 20 test
- [ ] Data format documented + validation scripts
- [ ] Ethics approval & data handling SOP published
---
 
### Phase 3: Experiments & Evaluation (Months 10-14)
**Goal:** Comprehensive evaluation, ablations, fairness testing, comparison to baselines.
 
#### 3.1 Experimental Design
 
**Experiment 1: Baseline Comparison**
 
Compare against prior work:
 
```python
baselines = {
    "Hu_et_al_2024": {
        "gaze": True,
        "mouse": False,
        "env_camera": True,
        "fusion": "late",
    },
    "Li_et_al_2021": {
        "gaze": False,
        "mouse": True,
        "env_camera": False,
        "fusion": None,
    },
    "Atoum_et_al_2017": {
        "gaze": True,
        "mouse": False,  # only window-switching
        "env_camera": "wearable",
        "fusion": "temporal_svm",
    },
    "Our_Proposed": {
        "gaze": True,
        "mouse": True,
        "env_camera": True,
        "fusion": "attention",
    },
}
```
 
Implement simplified versions of baselines on your dataset; report:
- AUC-ROC
- F1 at operating points (90% recall, 95% specificity)
- Inference time (FPS)
**Experiment 2: Ablation Study**
 
Remove one modality at a time:
 
```
Model         | Gaze | Mouse | Env | AUC   | F1   | Notes
Gaze-only     | ✓    |       |     | 0.82  | 0.74 | Weak on copy-paste
Mouse-only    |      | ✓     |     | 0.79  | 0.71 | Fails when quiet cheating
Env-only      |      |       | ✓   | 0.81  | 0.72 | Needs close phone
Gaze+Mouse    | ✓    | ✓     |     | 0.89  | 0.84 |
Gaze+Env      | ✓    |       | ✓   | 0.88  | 0.83 |
Mouse+Env     |      | ✓     | ✓   | 0.86  | 0.80 |
All Three     | ✓    | ✓     | ✓   | 0.93  | 0.89 | ← Best
```
 
**Experiment 3: Fairness & Bias Evaluation**
 
Test on subgroups:
 
```
Demographic         | N    | AUC  | FP Rate (target 5%)
────────────────────────────────────────────────────────
Glasses (yes)       | 25   | 0.91 | 6.2%  ← Slightly higher
Glasses (no)        | 20   | 0.94 | 4.1%
Dark skin tone      | 15   | 0.90 | 7.1%  ← Concerning
Light skin tone     | 30   | 0.93 | 4.5%
Age 18-25           | 30   | 0.93 | 4.8%
Age 26-40           | 15   | 0.91 | 5.9%
Screen readers      | 5    | 0.85 | 8.0%  ← High (needs UX fix)
```
 
**Action:** If FP rate >7% for any subgroup, investigate and retrain or add fairness constraints.
 
**Experiment 4: Sensitivity to Conditions**
 
Test robustness:
 
```
Condition           | Setup                      | AUC Impact
────────────────────────────────────────────────────────────
Backlighting        | Bright window behind       | -0.08
Poor laptop camera  | Low-res 480p              | -0.05
Noisy WiFi          | Simulated 100ms latency   | -0.02
Sunglasses          | Dark eyewear worn         | -0.12
Extreme head pose   | >45° yaw or pitch         | -0.15
```
 
**Experiment 5: Cross-Domain Transfer (Phase-in for CCTV)**
 
This is **early exploration** (full CCTV validation in Phase 4):
 
```
Training set: Remote single-candidate exams
Test set:     CCTV hallway footage (from Hu et al. or recorded in-house)
 
Results:
Remote-only model on CCTV test: AUC = 0.63 (degrades significantly)
Reason: Different lighting, multi-person occlusion, camera angle
 
→ Next: Domain adaptation (adversarial training or transfer learning)
```
 
**Code structure:**
```
code/experiments/
├── baseline_models.py           # Implement Hu, Li, Atoum variants
├── ablation_study.py            # Train 7 models (all combinations)
├── fairness_evaluation.py       # Subgroup analysis, demographic parity
├── sensitivity_analysis.py      # Backlighting, resolution, latency tests
├── cross_domain.py              # Train on remote, test on physical
├── metrics.py                   # ROC, PR curves, calibration
├── visualize_results.py         # Tables, plots, confusion matrices
└── report_generation.py         # LaTeX tables for paper
 
results/
├── ablation_study.csv
├── fairness_breakdown.json
├── sensitivity_heatmap.png
└── baseline_comparison.tex
```
 
**Output by Month 12:**
- [ ] Ablation study: all 7 models trained + results table
- [ ] Fairness report: no subgroup >7% false-positive rate
- [ ] Sensitivity analysis: identifies weak conditions
- [ ] Baseline comparison: shows improvement over Hu, Li, Atoum
- [ ] Cross-domain exploration: identifies gap for Phase 4
---
 
#### 3.2 Error Analysis & Debugging
 
**Categorize failures:**
 
```python
def analyze_errors(model, test_loader):
    """Categorize misclassifications."""
    errors = {
        "false_positive": [],  # Honest but flagged
        "false_negative": [],  # Cheating but missed
    }
    
    for batch in test_loader:
        gaze, mouse, env, label = batch
        pred = model(gaze, mouse, env) > 0.5
        
        # False positives: pred=True, label=False
        fp_mask = (pred) & (~label)
        if fp_mask.any():
            errors["false_positive"].append({
                "session_id": ...,
                "reason": inspect_features(gaze, mouse, env)[fp_mask],
                # e.g., "sudden gaze fixation" or "tab switch but no phone"
            })
    
    return errors
```
 
**Common failure modes to document:**
 
| Error Type | Cause | Fix |
|-----------|-------|-----|
| FP: Glasses | Gaze misestimation (glare on lenses) | Retrain gaze model with glasses-only data |
| FP: Phone in background | Environment camera detects unrelated phone | Add "distance to exam" feature |
| FN: Silent cheating | Student reads notes off-screen; no telemetry | Train on more note-lookup scenarios |
| FN: Network latency | Cursor lags; looks like idle | Add network-delay compensation |
 
**Output by Month 12:**
- [ ] Error analysis document (top 20 failure cases)
- [ ] Visualization of feature distributions for FP vs. TP vs. FN
- [ ] Proposed fixes and timeline
---
 
### Phase 4: CCTV Extension & Physical Deployment (Months 14-18)
**Goal:** Validate cross-domain transfer to physical exam halls.
 
#### 4.1 Domain Adaptation
 
**Step 1: Collect CCTV footage**
 
- **Partnership:** Negotiate with a university or test center to record exams (with consent)
- **If unavailable:** Simulate with scripted actors in a mock exam hall
- **Camera:** Same YOLO detector as Phase 3, but fixed on wall (not handheld)
**Step 2: Domain-adaptive training**
 
```python
class DomainAdaptiveModel(nn.Module):
    """Learn to work in both remote and CCTV domains."""
    
    def __init__(self):
        super().__init__()
        # Shared feature extractor
        self.shared_encoder = ...
        
        # Domain classifier (adversarial): "remote" vs. "CCTV"
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 2 domains
        )
        
        # Task classifier: "honest" vs. "cheating"
        self.task_classifier = nn.Linear(128, 1)
    
    def forward(self, x, domain_label=None):
        features = self.shared_encoder(x)
        task_out = self.task_classifier(features)
        
        if domain_label is not None:
            domain_out = self.domain_classifier(features)
            # Loss: task_loss + λ * domain_loss (adversarial)
            return task_out, domain_out
        
        return task_out
 
# Training: Mix remote + CCTV data
# Domain classifier learns features invariant across domains
```
 
**Step 3: Evaluate on CCTV test set**
 
```
Model                  | Train Data | Test Data | AUC
───────────────────────────────────────────────
Remote-only            | Remote     | CCTV      | 0.63 (poor transfer)
CCTV-only              | CCTV       | CCTV      | 0.88 (but needs CCTV data)
Domain-adaptive        | Remote+... | CCTV      | 0.81 (better transfer!)
Mixed (no adapt)       | Remote+... | CCTV      | 0.72 (naive mixing)
```
 
**Output by Month 15:**
- [ ] CCTV dataset (50+ exam sessions if real; 100+ if simulated)
- [ ] Domain-adaptive model trained
- [ ] Cross-domain evaluation: AUC >0.80 on CCTV test set
---
 
#### 4.2 Physical Deployment Pilot
 
**Goal:** Test in a real exam hall (or simulation) with multiple candidates.
 
**Setup:**
```
Exam Hall Configuration:
├── CCTV camera (fixed on ceiling/wall)
├── 4-6 exam tables (spaced)
├── Each table has:
│   ├── Monitor
│   ├── Keyboard/mouse
│   └── (No primary face-camera; use CCTV only)
└── Proctor room (watches 4 monitors)
```
 
**Data collection:**
- 20 supervised exams (with confederates doing "cheating")
- 30 real exams (normal students)
- Real-time alerts to proctor; proctor manually confirms true/false
**Metrics:**
- Proctor agreement rate (if alert fires, does proctor agree?)
- False-alarm rate in real setting
- Time to detect (seconds between cheat and alert)
- Usability: Can proctor understand why alert fired?
**Output by Month 16:**
- [ ] Pilot exam hall deployment documentation
- [ ] Real-world evaluation results (30+ sessions)
- [ ] Proctor feedback survey + iterative UX fixes
- [ ] Identified issues (occlusion, lighting, multi-person conflicts)
---
 
### Phase 5: Paper Writing & Publication (Months 15-18, parallel)
**Goal:** Publish high-impact venue (CHI, ACM CCS, or IEEE TMM).
 
#### 5.1 Paper Structure
 
```
1. Abstract (250 words)
   Key message: "Three-stream fusion (gaze+mouse+env) outperforms 
   any single modality; validated on 100+ exams; first cross-domain
   deployment to physical halls."
 
2. Introduction (3-4 pages)
   - Problem: Online exam integrity + surveillance concerns
   - Gap: No system combines all three streams
   - Contribution: Fusion model + dataset + cross-domain extension
 
3. Related Work (4-5 pages)
   [Your literature review + comparison table]
 
4. Proposed System (4-5 pages)
   - System architecture (3 cameras, sync, real-time pipeline)
   - Gaze estimation (MediaPipe + calibration)
   - Mouse/interaction features
   - Environment object detection (YOLOv8)
   - Fusion model (attention architecture)
   - Alert scoring (calibrated, explainable)
 
5. Dataset (2-3 pages)
   - Collection protocol
   - Annotation procedure
   - Statistics (100 sessions, 50 cheating types, etc.)
   - Kappa agreement
   - Release & privacy
 
6. Experiments (5-6 pages)
   - Baseline comparisons (Hu, Li, Atoum)
   - Ablation study (7 models)
   - Fairness analysis (subgroups)
   - Sensitivity analysis (glasses, lighting, latency)
   - Cross-domain transfer (remote → CCTV)
 
7. Results (2-3 pages + figures)
   - Main result: Table of all models, AUC/F1/FP-rate
   - Ablation: Figure showing stream importance
   - Fairness: Heatmap of FP rate by demographic
   - Sensitivity: Plots of degradation vs. condition
   - Confusion matrices and ROC curves
 
8. Discussion (3-4 pages)
   - Why three streams work better
   - Limitations (small dataset, simulated cheating, no real CCTV at scale)
   - Ethical implications (privacy, fairness, appeals process)
   - Future work (more participants, real exams, deployment)
 
9. Conclusions (1 page)
 
10. References (from your literature review)
 
11. Appendix
    - A: Gaze calibration procedure (detailed)
    - B: Feature engineering details
    - C: Hyperparameter sweep results
    - D: Failure case gallery
    - E: Consent form & IRB approval letter
```
 
#### 5.2 Figures & Visualizations
 
**Key figures to create:**
 
1. **System diagram** (Figure 1, top)
   - Three cameras (face, env, monitor) with sync arrows
   - Data pipeline: raw → features → fusion → alert score
   - Real-time inference on GPU/CPU
2. **Fusion model architecture** (Figure 2)
   - Input streams (gaze, mouse, env vectors)
   - Attention mechanism
   - Classification head
   - Example inference
3. **Ablation study results** (Figure 3, bar chart)
   - X-axis: 7 model configurations
   - Y-axis: AUC, F1, FP rate
   - Color-coded bars
4. **Fairness breakdown** (Figure 4, heatmap)
   - Rows: demographic groups (glasses, skin tone, age, accessibility)
   - Columns: metrics (AUC, FP rate, FN rate)
   - Color intensity = performance
5. **Sensitivity analysis** (Figure 5, line plots)
   - X-axis: condition severity (e.g., "backlighting intensity")
   - Y-axis: AUC degradation
   - Separate curve per modality
6. **Error analysis** (Figure 6, confusion matrices + gallery)
   - Confusion matrix: pred vs. ground truth
   - Top 9 false positives (annotated frames)
   - Top 9 false negatives
7. **ROC & PR curves** (Figure 7)
   - ROC curve: all baselines + proposed
   - PR curve: proposed model
   - Highlight operating point (e.g., "90% recall, 5% FP rate")
#### 5.3 Reproducibility & Code Release
 
**Before submission:**
 
- [ ] Clean up code; follow PEP 8
- [ ] Write docstrings for all functions
- [ ] Create `README.md` with setup instructions
- [ ] Add unit tests (pytest)
- [ ] Package as pip-installable: `pip install exam-proctoring`
- [ ] Publish to GitHub with MIT license
- [ ] Create Zenodo record for reproducibility
**GitHub structure:**
```
exam-proctoring-research/
├── README.md               # Overview, setup, quick start
├── requirements.txt        # Dependencies
├── setup.py               # Pip install
├── LICENSE                # MIT
├── CODE_OF_CONDUCT.md     # Community guidelines
├── CONTRIBUTING.md        # How to contribute
├── src/                   # Main package
│   ├── gaze/
│   ├── mouse/
│   ├── environment/
│   ├── fusion/
│   └── __init__.py
├── tests/                 # Unit tests
├── notebooks/             # Jupyter tutorials
├── data/                  # Dataset (anonymized)
├── models/                # Pre-trained weights
├── paper/                 # LaTeX source for paper
└── REPRODUCTION.md        # Step-by-step to reproduce results
```
 
**Output by Month 18:**
- [ ] Paper draft complete (all 8 sections)
- [ ] Code published on GitHub (public)
- [ ] Dataset uploaded to Zenodo (anonymized, CC-BY-SA license)
- [ ] Pre-trained model weights released
- [ ] Submitted to target venue (CHI, ACM CCS, or IEEE TMM)
---
 
## PART III: DETAILED TECHNICAL INSTRUCTIONS
 
### 3.1 Setting Up the Development Environment
 
```bash
# 1. Clone or initialize repository
git clone <your-repo> exam-proctoring-research
cd exam-proctoring-research
 
# 2. Python virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR: venv\Scripts\activate (Windows)
 
# 3. Install core dependencies
pip install --upgrade pip setuptools wheel
 
# Core ML/CV
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning tensorboard wandb
 
# Vision tasks
pip install opencv-python opencv-contrib-python
pip install mediapipe ultralytics  # Gaze + YOLOv8
pip install scikit-learn numpy pandas
 
# Development
pip install pytest black flake8 mypy jupyter ipython
 
# Optional: GPU acceleration
pip install nvidia-cuda-toolkit  # If not already installed
 
# 4. Verify installation
python -c "import torch; print(torch.cuda.is_available())"  # Should print True if GPU available
python -c "import cv2; print(cv2.__version__)"
 
# 5. Create folder structure
mkdir -p {code/{gaze,mouse,env_camera,fusion,experiments}, data/{raw,processed,annotations}, models, notebooks, paper, results}
 
# 6. Initialize git
git add .
git commit -m "Initial project setup"
```
 
### 3.2 Gaze Estimation Implementation (Starter Code)
 
**File: `code/gaze/gaze_estimator.py`**
 
```python
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, Dict
 
class GazeEstimator:
    """Real-time gaze estimation using MediaPipe."""
    
    def __init__(self, calibration_points: int = 9):
        """
        Initialize gaze estimator.
        
        Args:
            calibration_points: Number of calibration grid points (default 9 for 3x3)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        self.calibration_data = None
        self.calibration_points = calibration_points
        self.screen_width = 1920
        self.screen_height = 1080
        
    def extract_eye_features(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extract gaze-relevant features from a frame.
        
        Args:
            frame: Input frame (RGB, HxWx3)
            
        Returns:
            Dict with gaze_x, gaze_y, confidence, head_yaw, head_pitch or None
        """
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        
        # Key landmark indices for gaze
        LEFT_IRIS = 468
        RIGHT_IRIS = 473
        LEFT_EYE_INNER = 133
        RIGHT_EYE_INNER = 362
        
        # Get iris centers in pixel space
        left_iris = (landmarks[LEFT_IRIS].x * w, landmarks[LEFT_IRIS].y * h)
        right_iris = (landmarks[RIGHT_IRIS].x * w, landmarks[RIGHT_IRIS].y * h)
        
        # Simple head pose (simplified; real implementation uses solvePnP)
        # Landmark 10 (forehead), 152 (chin), 234 (left ear), 454 (right ear)
        forehead = np.array([landmarks[10].x, landmarks[10].y])
        chin = np.array([landmarks[152].x, landmarks[152].y])
        left_ear = np.array([landmarks[234].x, landmarks[234].y])
        right_ear = np.array([landmarks[454].x, landmarks[454].y])
        
        # Head yaw (left-right): ear distance asymmetry
        head_yaw = (right_ear[0] - left_ear[0]) * 90  # degrees
        
        # Head pitch (up-down): forehead-chin distance
        head_pitch = (chin[1] - forehead[1]) * 90  # degrees
        
        # Confidence: use face detection confidence
        confidence = results.multi_face_landmarks[0].landmark[0].z
        
        return {
            "left_iris": left_iris,
            "right_iris": right_iris,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "confidence": confidence,
            "raw_landmarks": landmarks,
        }
    
    def calibrate(self, video_source: str, grid_points: int = 9) -> bool:
        """
        Calibrate gaze estimator using grid of screen points.
        
        Args:
            video_source: Video file or camera index (0 for webcam)
            grid_points: Number of calibration points (default 3x3=9)
            
        Returns:
            True if calibration successful
        """
        print(f"Starting calibration with {grid_points} points...")
        
        # Generate grid coordinates
        rows, cols = int(np.sqrt(grid_points)), int(np.sqrt(grid_points))
        grid_x = np.linspace(0.1, 0.9, cols) * self.screen_width
        grid_y = np.linspace(0.1, 0.9, rows) * self.screen_height
        
        calibration_pairs = []
        
        cap = cv2.VideoCapture(video_source) if isinstance(video_source, int) else cv2.VideoCapture(video_source)
        
        for i, (gx, gy) in enumerate([(x, y) for y in grid_y for x in grid_x]):
            print(f"Point {i+1}/{grid_points}: Look at the center of the screen...")
            
            collected_samples = 0
            gaze_samples = []
            
            while collected_samples < 30:  # Collect 30 samples per point
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Show target point
                cv2.circle(frame, (int(gx), int(gy)), 20, (0, 255, 0), -1)
                cv2.imshow("Calibration", frame)
                
                # Extract gaze
                features = self.extract_eye_features(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if features:
                    iris_x = (features["left_iris"][0] + features["right_iris"][0]) / 2
                    iris_y = (features["left_iris"][1] + features["right_iris"][1]) / 2
                    gaze_samples.append([iris_x, iris_y])
                    collected_samples += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False
            
            # Average gaze for this point
            avg_gaze = np.mean(gaze_samples, axis=0)
            calibration_pairs.append([gx, gy, avg_gaze[0], avg_gaze[1]])
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Fit polynomial or neural network
        self.calibration_data = np.array(calibration_pairs)
        print("Calibration complete!")
        return True
    
    def estimate_gaze_point(self, frame: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """
        Estimate where user is looking on screen.
        
        Args:
            frame: Input frame (RGB)
            
        Returns:
            (screen_x, screen_y, confidence) or None
        """
        features = self.extract_eye_features(frame)
        if not features or self.calibration_data is None:
            return None
        
        # Get iris center
        iris_x = (features["left_iris"][0] + features["right_iris"][0]) / 2
        iris_y = (features["left_iris"][1] + features["right_iris"][1]) / 2
        
        # Simple linear interpolation from calibration
        # (Replace with polynomial fit or neural network for better accuracy)
        iris_coords = np.array([[iris_x, iris_y]])
        calib_iris = self.calibration_data[:, 2:4]
        calib_screen = self.calibration_data[:, 0:2]
        
        from sklearn.neighbors import KNeighborsRegressor
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(calib_iris, calib_screen)
        
        predicted_screen = knn.predict(iris_coords)[0]
        screen_x = int(np.clip(predicted_screen[0], 0, self.screen_width))
        screen_y = int(np.clip(predicted_screen[1], 0, self.screen_height))
        
        return screen_x, screen_y, features["confidence"]
 
# Usage
if __name__ == "__main__":
    estimator = GazeEstimator()
    
    # Step 1: Calibrate (run once)
    estimator.calibrate(video_source=0, grid_points=9)
    
    # Step 2: Real-time estimation
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = estimator.estimate_gaze_point(rgb_frame)
        
        if result:
            x, y, conf = result
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"Confidence: {conf:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Gaze Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```
 
### 3.3 Mouse Behavior Feature Extraction
 
**File: `code/mouse/feature_extractor.py`**
 
```python
import numpy as np
from typing import List, Dict
from collections import deque
 
class MouseBehaviorExtractor:
    """Extract behavioral features from mouse events."""
    
    def __init__(self, window_size: float = 10.0):
        """
        Initialize extractor.
        
        Args:
            window_size: Rolling window duration (seconds)
        """
        self.window_size = window_size
        self.events = deque()
        
    def add_event(self, timestamp: float, event_type: str, x: int = None, y: int = None, 
                  duration_ms: int = None):
        """
        Add a mouse/keyboard event.
        
        Args:
            timestamp: Event timestamp (seconds)
            event_type: "move", "click", "idle", "scroll", "window_focus", "keystroke"
            x, y: Coordinates (for move/click)
            duration_ms: Duration (for idle)
        """
        self.events.append({
            "timestamp": timestamp,
            "type": event_type,
            "x": x,
            "y": y,
            "duration_ms": duration_ms,
        })
        
        # Keep only events within window
        while self.events and self.events[0]["timestamp"] < timestamp - self.window_size:
            self.events.popleft()
    
    def extract_features(self) -> Dict[str, float]:
        """
        Extract features from current event window.
        
        Returns:
            Dict of feature names → values
        """
        events = list(self.events)
        if len(events) < 2:
            return self._zero_features()
        
        features = {}
        
        # ===== Cursor Movement =====
        move_events = [e for e in events if e["type"] == "move"]
        if len(move_events) >= 2:
            xs = [e["x"] for e in move_events]
            ys = [e["y"] for e in move_events]
            ts = [e["timestamp"] for e in move_events]
            
            # Distances
            displacements = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
            time_diffs = np.diff(ts)
            
            velocities = displacements / (time_diffs + 1e-6)
            features["cursor_velocity_mean"] = np.mean(velocities)
            features["cursor_velocity_std"] = np.std(velocities)
            features["cursor_velocity_max"] = np.max(velocities)
            
            # Acceleration
            accelerations = np.diff(velocities) / (time_diffs[:-1] + 1e-6)
            features["cursor_acceleration_mean"] = np.mean(accelerations) if len(accelerations) > 0 else 0
            features["cursor_acceleration_std"] = np.std(accelerations) if len(accelerations) > 0 else 0
            
            # Distance from last position
            features["cursor_distance_from_start"] = np.sum(displacements)
        
        # ===== Clicks =====
        click_events = [e for e in events if e["type"] == "click"]
        features["click_count"] = len(click_events)
        features["click_frequency"] = len(click_events) / self.window_size  # clicks per second
        
        # ===== Idle Time =====
        idle_events = [e for e in events if e["type"] == "idle"]
        total_idle_ms = sum(e["duration_ms"] for e in idle_events if e["duration_ms"])
        features["idle_duration_total_ms"] = total_idle_ms
        features["idle_fraction"] = total_idle_ms / (self.window_size * 1000)
        
        # ===== Window Focus =====
        focus_changes = sum(1 for e in events if e["type"] == "window_focus")
        features["window_focus_changes"] = focus_changes
        
        # ===== Keystroke Dynamics =====
        keystroke_events = [e for e in events if e["type"] == "keystroke"]
        if len(keystroke_events) > 0:
            features["keystroke_count"] = len(keystroke_events)
            features["keystroke_frequency"] = len(keystroke_events) / self.window_size
            
            # Keystroke timing variability
            key_ts = [e["timestamp"] for e in keystroke_events]
            if len(key_ts) > 1:
                key_intervals = np.diff(key_ts)
                features["keystroke_interval_mean"] = np.mean(key_intervals)
                features["keystroke_interval_std"] = np.std(key_intervals)
                # Entropy: are keystrokes uniform or bursty?
                hist, _ = np.histogram(key_intervals, bins=10, range=(0, self.window_size))
                hist = hist / hist.sum()
                features["keystroke_entropy"] = -np.sum(hist * np.log(hist + 1e-10))
        
        # ===== Suspicious Patterns =====
        # Sudden velocity spike (bot-like)?
        if "cursor_velocity_max" in features:
            features["velocity_spike_ratio"] = (
                features["cursor_velocity_max"] / (features["cursor_velocity_mean"] + 1e-6)
            )
        
        # Lots of clicks in short time?
        features["click_burst_indicator"] = 1.0 if features["click_frequency"] > 5 else 0.0
        
        return features
    
    def _zero_features(self) -> Dict[str, float]:
        """Return zero-initialized feature dict."""
        feature_names = [
            "cursor_velocity_mean", "cursor_velocity_std", "cursor_velocity_max",
            "cursor_acceleration_mean", "cursor_acceleration_std",
            "cursor_distance_from_start",
            "click_count", "click_frequency",
            "idle_duration_total_ms", "idle_fraction",
            "window_focus_changes",
            "keystroke_count", "keystroke_frequency",
            "keystroke_interval_mean", "keystroke_interval_std", "keystroke_entropy",
            "velocity_spike_ratio", "click_burst_indicator",
        ]
        return {name: 0.0 for name in feature_names}
 
# Usage
if __name__ == "__main__":
    extractor = MouseBehaviorExtractor(window_size=10.0)
    
    # Simulate events
    extractor.add_event(0.0, "move", x=100, y=200)
    extractor.add_event(0.1, "move", x=150, y=220)
    extractor.add_event(0.5, "click", x=150, y=220)
    extractor.add_event(1.0, "keystroke")
    extractor.add_event(5.0, "idle", duration_ms=2000)
    
    features = extractor.extract_features()
    for name, value in features.items():
        print(f"{name}: {value:.4f}")
```
 
### 3.4 YOLO Object Detection Setup
 
**File: `code/env_camera/yolo_setup.py`**
 
```python
from ultralytics import YOLO
import cv2
 
# Step 1: Download pre-trained YOLOv8 model
model = YOLO("yolov8s.pt")  # small model, good speed/accuracy trade-off
 
# Step 2: Prepare custom dataset
# Create dataset.yaml:
# path: /path/to/dataset
# train: images/train
# val: images/val
# test: images/test
# nc: 5  # number of classes
# names: ['phone', 'notebook', 'textbook', 'person', 'suspicious_item']
 
# Step 3: Fine-tune on custom data
results = model.train(
    data="path/to/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,  # GPU device ID
    patience=10,  # Early stopping
    save=True,
    plots=True,
    optimizer="Adam",
    lr0=0.001,
)
 
# Step 4: Evaluate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
 
# Step 5: Real-time inference
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame)
    
    # Visualize
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
 
# Step 6: Export model for deployment
model.export(format="onnx")  # or "tflite", "torchscript", etc.
```
 
---
 
## PART IV: RESOURCE & BUDGET PLAN
 
### 4.1 Personnel & Time
 
| Role | Time (mo) | FTE | Salary (if paid) |
|------|-----------|-----|------------------|
| Project Lead | 18 | 1.0 | $180k/year |
| Vision/Gaze Lead | 12 | 0.8 | $80k/year |
| Interaction Lead | 12 | 0.8 | $80k/year |
| Environment/YOLOv8 | 10 | 0.7 | $70k/year |
| ML/Fusion Lead | 14 | 1.0 | $100k/year |
| Ethics/UX | 8 | 0.5 | $50k/year |
| Data Annotation | 6 | 0.3 | $30k/year |
| **TOTAL** | **~14 avg** | **~5 FTE** | **~$590k/year** |
 
**If unfunded:** Expect to run this with 2-3 volunteers/grad students in parallel (18-24 months).
 
### 4.2 Hardware & Software
 
| Item | Cost | Qty | Total | Notes |
|------|------|-----|-------|-------|
| **Development Machine** | | | | |
| GPU workstation (RTX 4090) | $3,500 | 1 | $3,500 | For training |
| Laptop (research) | $1,500 | 2 | $3,000 | Code, experiments |
| **Participant Testing** | | | | |
| USB webcams (1080p, 60Hz) | $80 | 20 | $1,600 | Backup cameras |
| Monitor (27", 1920x1080) | $300 | 2 | $600 | Testing setup |
| Microphone (USB) | $50 | 5 | $250 | Audio logging |
| **Software** | | | | |
| PyCharm Professional | $200/year | 3 | $600 | IDE |
| GitHub Enterprise | $21/mo | 1 | $252 | Private repo, CI/CD |
| AWS S3 (1TB, 12 mo) | $23/mo | 1 | $276 | Data backup |
| Label Studio Cloud | $50/mo | 1 | $600 | Annotation tool |
| **Survey/Ethics** | | | | |
| Participant compensation | $20/session | 100 | $2,000 | 100+ sessions × $20 |
| IRB processing | $1,000 | 1 | $1,000 | Varies by institution |
| **Conferences** | | | | |
| CHI 2025 registration | $1,200 | 1 | $1,200 | Top venue |
| Travel (flights + hotel) | $3,000 | 1 | $3,000 | 1 team member |
| **TOTAL** | | | **$17,978** | |
 
**Assuming academic affiliation:** Costs mostly covered by university (GPU, software licenses, IRB). Out-of-pocket: ~$3k-5k.
 
### 4.3 Timeline Gantt
 
```
Phase 0 (Months 1-2):    ████████ Foundation
Phase 1 (Months 3-5):    ████████████████ Single modalities
Phase 2 (Months 6-9):    ████████████████████████ Fusion + Dataset
Phase 3 (Months 10-14):  ████████████████████████████████ Experiments
Phase 4 (Months 14-18):  ████████████████████ CCTV + Deployment
Paper Writing (Months 15-18, parallel): ████████████████
```
 
---
 
## PART V: RISK MANAGEMENT & CONTINGENCIES
 
### 5.1 Top Risks
 
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **IRB delays** | High (4-8 wks) | Blocks Phase 2 data collection | Start early; work with compliance officer |
| **Recruitment shortfall** | Medium | Smaller dataset, weaker results | Pre-recruit; offer $25/session; online + local |
| **Gaze estimation accuracy** | Medium | Poor calibration on glasses/dark eyes | Collect diverse pilot data; compare MediaPipe vs. iTracker |
| **YOLO training overfits** | Low-Medium | Poor object detection on real CCTV | Use augmentation; monitor val loss; early stopping |
| **Fusion model doesn't improve** | Low | All-three isn't better than pairs | Design features carefully; try different architectures |
| **Hardware fails** | Low | Loss of data/experiment progress | Automatic cloud backups (AWS); local redundancy |
| **Venue rejects paper** | Medium | Delayed publication; must revise | Prepare for R1 revisions; have secondary venue (IEEE TMM) |
| **CCTV partnership unavailable** | Medium-High | Can't validate cross-domain | Pivot to controlled simulation; partner with schools early |
 
### 5.2 Contingency Plan
 
**If IRB delayed past Month 3:**
- Collect Phase 1 data on yourself + close collaborators (no formal study)
- Prepare annotation and feature extraction in parallel
**If recruitment stalls at 50 sessions:**
- Extend data collection to Month 10
- Use semi-supervised learning on unlabeled data
- Focus on remote exams only (defer CCTV to future work)
**If gaze estimation doesn't calibrate well:**
- Use head-pose proxy instead (still useful signal)
- Emphasize mouse + environment as primary
- Note limitation in paper: "Gaze module limited to glasses-free users"
**If fusion model shows no improvement:**
- Document as negative result (valuable for literature)
- Investigate why: do streams encode redundant information?
- Publish as workshop paper; iterate for future work
**If CCTV data unavailable:**
- Simulate with recorded actors in mock exam hall
- Frame as "preliminary cross-domain validation"
- Propose real deployment as future work
---
 
## PART VI: ETHICS & RESPONSIBILITY
 
### 6.1 Ethical Principles in Practice
 
**Principle 1: Transparency**
- Users know they're being monitored (consent form)
- Alert explanations are human-readable ("gaze left exam window for 5s + tab switch to Gmail")
- Proctor review interface shows features, not just verdict
**Principle 2: Fairness**
- Test for bias by gender, disability, skin tone, age
- Report false-positive rate for all subgroups
- No deployment until disparities <1%
**Principle 3: Proportionality**
- Don't use surveillance as punishment; use for evidence
- Assume innocent until investigation
- Right to appeal + manual review
**Principle 4: Privacy**
- Gaze data stored locally if possible; cloud with encryption
- Delete within 30 days (or per consent)
- Face de-identification before dataset release
- Use only aggregate statistics in papers
**Principle 5: Accessibility**
- Support screen readers, voice control, eye-gaze trackers
- Test with neurodivergent users
- Don't require perfect vision or hearing
### 6.2 Consent Form Template
 
```
CONSENT TO PARTICIPATE IN RESEARCH STUDY
"Multi-Modal Exam Proctoring System"
 
You are being asked to participate in a research study evaluating an 
automated exam proctoring system that uses computer vision and behavior analysis.
 
PROCEDURES:
- You will take a 10-15 minute online exam
- Two webcams will record your face and desk area
- Your mouse movements, keyboard, and screen activity will be logged
- We will then measure how accurately our system detects suspicious behavior
 
DATA COLLECTION:
- Video (face, desk) - face will be blurred before analysis
- Mouse coordinates and clicks
- Keyboard typing timing (NOT keystrokes themselves)
- Screen window titles (NOT URL content)
- Eye gaze direction (NOT individual frames)
 
RISKS:
- Minimal. You may feel self-conscious being recorded.
- Your data is encrypted and access-controlled.
 
BENEFITS:
- $20 compensation
- Your participation advances academic integrity research
- You will not be graded on exam performance (this is a research study, not a real exam)
 
DATA RETENTION & DELETION:
- Raw data stored securely for 30 days
- After that, videos deleted; anonymized features retained for up to 1 year
- You can request data deletion at any time
 
PUBLICATION:
- Results will be published in academic venues
- De-identified dataset may be released publicly (faces blurred)
- Your name will NOT appear in any publication
 
CONTACT:
- Questions: [PI Email]
- Ethics concerns: [IRB Email]
 
I agree to participate: [Checkbox]
I consent to dataset release (anonymized): [Checkbox]
```
 
---
 
## PART VII: PUBLICATION STRATEGY
 
### 7.1 Target Venues (in order of preference)
 
1. **CHI 2025** (Deadline: Sept 2024; this is aggressive—aim for CHI 2026)
   - Top-tier HCI venue
   - Values dataset contributions
   - Accepts fairness + ethics work
   
2. **ACM CCS 2025** (Deadline: May 2025; aim for Feb 2026 submission)
   - Top security conference
   - Strong track record on surveillance / privacy
   - Likes cross-domain evaluation
3. **IEEE Transactions on Multimedia** (Journal, rolling submission)
   - Atoum et al. published here
   - Accepts multimodal fusion work
   - Lower impact than CHI/CCS but solid alternative
### 7.2 Authorship & Order
 
Recommend:
1. **PhD student / Lead researcher** (you?) — first author
2. **Gaze expert** — second author (or co-first if equal contribution)
3. **ML/Fusion lead** — third
4. **PI / Advisor** — last author
*Clear authorship expectations from start to avoid conflicts later.*
 
### 7.3 Preprint Strategy
 
- Post to **arXiv** upon conference submission (most venues allow)
- Mention arXiv in GitHub README
- Cite preprint version in paper submissions
---
 
## PART VIII: NEXT IMMEDIATE STEPS (TODAY)
 
**Week 1:**
- [ ] Set up GitHub repo (public or private)
- [ ] Create project board (Kanban: To Do, In Progress, Done)
- [ ] Write detailed Phase 0 tasks
- [ ] Reach out to potential collaborators
**Week 2:**
- [ ] Set up Python environment (virtualenv, test imports)
- [ ] Clone or download Hu et al., Li et al., Atoum et al. papers (if available)
- [ ] Download MediaPipe + YOLOv8 models (test locally)
- [ ] Start IRB application form (even if not submitting yet)
**Week 3:**
- [ ] Write Phase 1 detailed plan (gaze calibration protocol, specific code structure)
- [ ] Implement gaze estimator skeleton code (gets to "it runs" state)
- [ ] Test mouse logging library (pynput? or OS hooks?)
**Week 4:**
- [ ] First milestone check-in: Can you run a gaze estimation pipeline (even with dummy data)?
- [ ] Refine IRB protocol with advisor / compliance office
- [ ] Create recruitment flyer template
---
 
**You now have a 18-month roadmap with concrete steps, code examples, ethics grounding, and realistic timelines.** Start Phase 0 this week. Good luck! 🚀
 
 
