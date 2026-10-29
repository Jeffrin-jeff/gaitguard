# GaitGuard

**Gait matching with machine learning: do two walking videos show the same walking pattern?**

GaitGuard is a Python desktop app that compares walking patterns using pose estimation and machine learning. It was built for a dementia patient exit-detection system: the patient's walking video is the reference, and a live camera at the ward door is checked against it. When the walking pattern matches the patient, an alarm is raised.

## Problem Statement
Dementia patients can wander out of care areas unnoticed. Face-based systems often fail with masks, distance or poor lighting. Gait (the way a person walks) is a pattern that can be recognised from a distance, so GaitGuard uses it to identify the patient.

## Key Features
- **Video A vs Video B comparison:** choose a reference video and a test video, and a statistical check runs at once
- **Pose estimation with MediaPipe:** extracts body landmarks (shoulders, hips, knees, ankles, toes) from each video
- **Gait features:** walking speed, cadence, step length, step-time variability, left/right asymmetry, knee bend range, thigh swing range, foot lift height and more
- **Sliding windows:** 4-second analysis windows, with a new window every 0.5 seconds
- **AI verifier (Random Forest):** train on your own video library, tested on pairs of videos it has never seen. It is used for the final answer only if it does at least as well as the statistical check
- **Live camera mode:** compares the camera with Video A once per second; **3 of the last 5 checks** must match to raise the alarm, which avoids false alarms
- **Microcontroller buzzer (optional):** an Arduino or ESP32 over USB serial turns a buzzer ON or OFF
- **Mobile notifications (optional):** phone alerts through ntfy.sh
- **Delete all saved data:** one button removes the cache, AI library, trained model and reports (your original videos are never touched)

## How It Works
1. Choose **Video A** (the patient's walking video, the reference).
2. Choose **Video B** or switch on the live camera.
3. The app extracts pose landmarks and computes gait features for each window.
4. It compares the windows to Video A, using the statistical check or the trained AI model.
5. If the walking matches, the status turns red, the alarm sounds, and the buzzer signal is sent.

## Tech Stack
- Python
- OpenCV and MediaPipe (video and pose estimation)
- NumPy and SciPy (signal processing, step detection)
- scikit-learn (Random Forest classifier)
- Tkinter and Pillow (desktop interface)
- Matplotlib (feature importance chart)
- pyserial (microcontroller link)

## Getting Started
```bash
git clone https://github.com/Jeffrin-jeff/gaitguard.git
cd gaitguard
pip install -r requirements.txt
python gaitguard_v7_mcu1.py
```

## Optional Setup (Windows)
**Buzzer on Arduino or ESP32:**
```
set GAITGUARD_SERIAL_PORT=COM3
```
The app sends `1` (alarm on) or `0` (alarm off) at 115200 baud.

**Phone notifications:**
```
set GAITGUARD_NTFY_TOPIC=your-private-topic-name
```

## Notes
- The door lock and buzzer hardware are part of the full system design. This program covers the detection part and the signal to the microcontroller.
- Walking videos and trained models are not stored in this repository.

## Project Info
Mini Project-III (24BM451), Department of Biomedical Engineering, Rohini College of Engineering and Technology

## Author
Jeffrin