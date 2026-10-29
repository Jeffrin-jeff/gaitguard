"""
GaitGuard v4 - Gait Match with machine learning
Do two walking videos show the SAME walking pattern?

Quick use
  1. Choose Video A (reference, e.g. the patient) and Video B (test). The statistical check runs at once.
  2. For more accuracy: "Add videos to AI library" (give each walking style/person a group name),
     then "Train AI". The AI learns which gait differences matter and is TESTED on pairs of videos
     it has never seen. It is used for the final answer only if it does at least as well as the statistical check.
  3. "Delete ALL saved data" wipes the cache, the AI library, the trained model and the reports.

Install:  pip install opencv-python mediapipe numpy scipy pillow scikit-learn matplotlib
Run:      python gaitguard_v4.py
"""
import os
import re
import time
import queue
import shutil
import pickle
import hashlib
import threading
import traceback
import urllib.request
from collections import deque

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------------------- settings
CACHE_DIR, LIB_DIR, MODEL_DIR = "cache", "library", "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gait_verifier.pkl")
REPORT_PATH, IMP_PNG = "ai_report.txt", "feature_importance.png"
for _d in (CACHE_DIR, LIB_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)

WIN_S, HOP_S = 4.0, 0.5          # one analysis window = 4 s, a new window every 0.5 s
MAX_WIDTH = 960
VOTES_N, VOTES_NEED = 5, 3       # live alarm: 3 of the last 5 checks must match
FEAT_NEEDED, WIN_NEEDED = 0.70, 0.60
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
CACHE_VER = "v4"
NTFY_TOPIC = os.environ.get("GAITGUARD_NTFY_TOPIC", "").strip()
NTFY_COOLDOWN_S = 300
# Microcontroller (Arduino / ESP32) buzzer link over USB serial.
# Set the port in an environment variable, e.g.  set GAITGUARD_SERIAL_PORT=COM3   (Linux: /dev/ttyUSB0)
SERIAL_PORT = os.environ.get("GAITGUARD_SERIAL_PORT", "").strip()
SERIAL_BAUD = 115200


class MCULink:
    """Sends '1' (alarm ON) or '0' (alarm OFF) to the microcontroller once per second."""
    def __init__(self, port, baud):
        self.ser = None
        if not port:
            return
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0.1, write_timeout=0.5)
            time.sleep(2.0)          # Arduino resets when the port opens
            print(f"Microcontroller connected on {port}")
        except Exception as exc:
            print(f"Microcontroller NOT connected ({exc})")
            self.ser = None

    def send(self, alarm_on):
        if self.ser is None:
            return
        if alarm_on != getattr(self, "last_state", None):
            self.last_state = alarm_on
            print(f"Microcontroller: buzzer {'ON' if alarm_on else 'OFF'}")
        try:
            self.ser.write(b"1\n" if alarm_on else b"0\n")
        except Exception as exc:
            print(f"Microcontroller write failed: {exc}")
            self.ser = None

    def pulse(self, seconds=4.0):
        """Buzzer ON for a few seconds without blocking the GUI (used for Video A vs B match)."""
        def job():
            end = time.time() + seconds
            while time.time() < end:
                self.send(True)
                time.sleep(0.5)
            self.send(False)
        threading.Thread(target=job, daemon=True).start()

    def close(self):
        self.send(False)
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None


def send_mobile_notification(message):
    if not NTFY_TOPIC:
        return
    topic = re.sub(r"[^A-Za-z0-9_-]", "", NTFY_TOPIC)
    if not topic:
        return
    request = urllib.request.Request(
        f"https://ntfy.sh/{topic}", data=message.encode("utf-8"),
        headers={"Title": "GaitGuard: NO MATCH", "Priority": "high", "Tags": "warning"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=8):
            pass
    except Exception as exc:
        print(f"Mobile notification failed: {exc}")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

LM_IDS = [0, 11, 12, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
NOSE, SH_L, SH_R, WR_L, WR_R, HIP_L, HIP_R, KN_L, KN_R, AN_L, AN_R, TOE_L, TOE_R = range(13)

FEATURES = [
    ("speed", "Walking speed (torso/s)", "How fast the person moves, in body lengths per second"),
    ("cadence", "Cadence (steps/s)", "How many steps are taken every second"),
    ("step_len", "Step length (torso)", "How far apart the feet are at each step"),
    ("step_cv", "Step-time variability", "How uneven the timing between steps is"),
    ("step_asym", "Left/right asymmetry", "How different the left and right steps are"),
    ("knee_range", "Knee bend range (deg)", "How much the knees bend while walking"),
    ("hip_range", "Thigh swing range (deg)", "How far the thighs swing forwards and back"),
    ("foot_lift", "Foot lift height (torso)", "How high the feet are lifted (shuffling = low)"),
    ("lean_mean", "Trunk lean (deg)", "Forward (+) or backward (-) lean of the upper body"),
    ("lean_std", "Trunk sway (deg)", "How much the upper body wobbles while walking"),
    ("arm_swing", "Arm swing (torso)", "How much the arms swing"),
    ("bounce", "Vertical bounce (torso)", "Up-and-down movement of the hips"),
    ("head_fwd", "Head position (torso)", "How far the head sits ahead of the shoulders"),
]
NF = len(FEATURES)
FKEYS = [f[0] for f in FEATURES]
FNAMES = [f[1] for f in FEATURES]
MIN_SCALE = np.array([0.15, 0.15, 0.05, 0.05, 0.10, 4.0, 4.0, 0.03, 2.0, 1.0, 0.05, 0.01, 0.03])


# ----------------------------------------------------------------------------- pose + features
def run_pose(frame_bgr, pose):
    h, w = frame_bgr.shape[:2]
    res = pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None, res
    lm = res.pose_landmarks.landmark
    return np.array([[lm[i].x * w, lm[i].y * h, lm[i].visibility] for i in LM_IDS]), res


def smooth(a, k=5):
    if len(a) < k:
        return a
    pad = k // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(ap[:, j], ker, mode="valid") for j in range(a.shape[1])], axis=1)


def joint_angle(a, b, c):
    ba, bc = a - b, c - b
    cosang = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def reject(why, key):
    if why is not None:
        why[key] = why.get(key, 0) + 1
    return None


def compute_features(ts, L, why=None):
    """ts (n,) seconds, L (n, 13, 3) pixel landmarks -> 13 gait features, or None if unusable."""
    if len(ts) < 10 or ts[-1] - ts[0] <= 0:
        return reject(why, "too few frames")
    vis = L[:, :, 2].mean(axis=0)
    if vis[[SH_L, SH_R, HIP_L, HIP_R, KN_L, KN_R, AN_L, AN_R]].min() < 0.3:
        return reject(why, "full body not visible (head-to-feet must be in frame)")

    X, Y = smooth(L[:, :, 0]), smooth(L[:, :, 1])
    sh_x, sh_y = (X[:, SH_L] + X[:, SH_R]) / 2, (Y[:, SH_L] + Y[:, SH_R]) / 2
    hp_x, hp_y = (X[:, HIP_L] + X[:, HIP_R]) / 2, (Y[:, HIP_L] + Y[:, HIP_R]) / 2

    torso = float(np.median(np.hypot(sh_x - hp_x, sh_y - hp_y)))
    if torso < 20:
        return reject(why, "person too small in frame (stand closer)")

    slope = np.polyfit(ts, hp_x, 1)[0]
    mid = len(ts) // 2
    s1 = np.polyfit(ts[:mid], hp_x[:mid], 1)[0]
    s2 = np.polyfit(ts[mid:], hp_x[mid:], 1)[0]
    if s1 * s2 < 0 and min(abs(s1), abs(s2)) > 0.15 * torso:
        return reject(why, "turning (direction changed inside window)")
    speed = abs(slope) / torso
    if speed < 0.1:
        return reject(why, "not moving sideways (standing, turning, or walking toward camera)")
    d = 1.0 if slope >= 0 else -1.0

    sep = np.abs(X[:, AN_L] - X[:, AN_R]) / torso
    fps = (len(ts) - 1) / (ts[-1] - ts[0])
    peaks, _ = find_peaks(sep, distance=max(1, int(0.25 * fps)), prominence=0.05)
    if len(peaks) < 3:
        return reject(why, "could not detect steps")
    pt, amp = ts[peaks], sep[peaks]
    iv = np.diff(pt)
    cadence = (len(peaks) - 1) / (pt[-1] - pt[0])
    step_len = float(np.median(amp))
    step_cv = float(np.std(iv) / np.mean(iv))
    step_asym = float(abs(amp[::2].mean() - amp[1::2].mean()) / (amp.mean() + 1e-9))

    def P(j):
        return np.stack([X[:, j], Y[:, j]], axis=1)

    def rng(a):
        return float(np.percentile(a, 95) - np.percentile(a, 5))

    def detrend(y):
        return y - np.polyval(np.polyfit(ts, y, 1), ts)

    def thigh(h, k):
        return np.degrees(np.arctan2((X[:, k] - X[:, h]) * d, (Y[:, k] - Y[:, h])))

    knee_range = (rng(joint_angle(P(HIP_L), P(KN_L), P(AN_L))) +
                  rng(joint_angle(P(HIP_R), P(KN_R), P(AN_R)))) / 2
    hip_range = (rng(thigh(HIP_L, KN_L)) + rng(thigh(HIP_R, KN_R))) / 2
    foot_lift = (rng(detrend(Y[:, TOE_L])) + rng(detrend(Y[:, TOE_R]))) / (2 * torso)
    lean = np.degrees(np.arctan2((sh_x - hp_x) * d, (hp_y - sh_y)))
    arm_swing = (rng((X[:, WR_L] - X[:, SH_L]) * d / torso) +
                 rng((X[:, WR_R] - X[:, SH_R]) * d / torso)) / 2
    bounce = float(np.std(detrend(hp_y)) / torso)
    head_fwd = float(np.mean((X[:, NOSE] - sh_x) * d) / torso)
    lean_std = float(np.std(lean))
    if lean_std > 12 or knee_range > 95 or hip_range > 75:
        return reject(why, "tracking glitch (implausible joint motion)")
    return np.array([speed, cadence, step_len, step_cv, step_asym, knee_range, hip_range, foot_lift,
                     float(np.mean(lean)), lean_std, arm_swing, bounce, head_fwd], dtype=float)


def windows_from_sequence(ts, L, why=None):
    feats, starts = [], []
    if len(ts) < 20:
        reject(why, "person detected in fewer than 20 frames")
        return feats, starts
    t = ts[0]
    while t + WIN_S <= ts[-1] + 1e-6:
        m = (ts >= t) & (ts < t + WIN_S)
        if m.sum() >= 20 and (ts[m][-1] - ts[m][0]) >= 0.7 * WIN_S:
            f = compute_features(ts[m], L[m], why)
            if f is not None:
                feats.append(f)
                starts.append(t)
        else:
            reject(why, "person lost for part of the window")
        t += HOP_S
    return feats, starts


def process_video(path, progress_cb=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Could not open video: " + path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ts, rows = [], []
    w0 = h0 = i = 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] > MAX_WIDTH:
                s = MAX_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s)
            h0, w0 = frame.shape[:2]
            arr, _ = run_pose(frame, pose)
            if arr is not None:
                ts.append(i / fps)
                rows.append(arr)
            i += 1
            if progress_cb and i % 15 == 0:
                progress_cb(min(i / total, 1.0))
    cap.release()
    return np.array(ts), np.array(rows), {"frames": i, "found": len(ts), "fps": fps, "w": w0, "h": h0}


def file_sig(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        h.update(fh.read(2000000))
    return os.path.getsize(path), h.hexdigest()


def extract_clip(path, progress_cb=None):
    """Video file -> {'X': windows x 13 features, 'T': window start times, 'info', 'why'} (cached)."""
    st = os.stat(path)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_",
                  f"{os.path.basename(path)}_{st.st_size}_{int(st.st_mtime)}_{CACHE_VER}_{WIN_S}_{HOP_S}.pkl")
    cpath = os.path.join(CACHE_DIR, name)
    if os.path.exists(cpath):
        with open(cpath, "rb") as fh:
            return pickle.load(fh)
    ts, L, info = process_video(path, progress_cb)
    why = {}
    feats, starts = windows_from_sequence(ts, L, why)
    res = {"X": np.array(feats).reshape(-1, NF), "T": np.array(starts), "info": info, "why": why}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cpath, "wb") as fh:
        pickle.dump(res, fh)
    return res


def diag_text(res):
    i = res["info"]
    t = f"{i['w']}x{i['h']}, {i['fps']:.0f} fps, {i['frames']} frames, person found in {i['found']}."
    if res["why"]:
        t += " Rejected windows: " + "; ".join(f"{k} x{v}" for k, v in res["why"].items())
    return t


# ----------------------------------------------------------------------------- statistical check
def spread(sa, sb, ma, mb):
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
    return np.maximum.reduce([pooled, 0.10 * np.maximum(np.abs(ma), np.abs(mb)), MIN_SCALE])


def compare_clips(XA, XB, tol):
    ma, sa, mb, sb = XA.mean(0), XA.std(0), XB.mean(0), XB.std(0)
    s = spread(sa, sb, ma, mb)
    z = np.abs(mb - ma) / s
    feat_ok = z <= tol
    dist = np.sqrt(np.mean(((XB - ma) / s) ** 2, axis=1))
    win_ok = dist <= 2 * tol
    fmr, wmr = float(feat_ok.mean()), float(win_ok.mean())
    return {"ma": ma, "mb": mb, "z": z, "feat_ok": feat_ok, "dist": dist, "win_ok": win_ok,
            "fmr": fmr, "wmr": wmr, "sim": (fmr + wmr) / 2, "match": fmr >= FEAT_NEEDED and wmr >= WIN_NEEDED}


# ----------------------------------------------------------------------------- machine learning (pair verification)
def pair_vec(Xi, Xj, gs):
    """Features of a PAIR of windows: how much each gait feature differs, and the typical value."""
    return np.hstack([np.abs(Xi - Xj) / gs, (Xi + Xj) / 2])


def sample_pairs(clips, rng, n_per_class):
    """Genuine pairs = windows of the same walking group (never overlapping windows of one video).
    Impostor pairs = windows of different groups."""
    by_group = {}
    for k, c in enumerate(clips):
        by_group.setdefault(c["group"], []).append(k)
    names = list(by_group)
    Xi, Xj, y = [], [], []
    for label in (1, 0):
        got = tries = 0
        while got < n_per_class and tries < 40 * n_per_class:
            tries += 1
            a = int(rng.integers(len(clips)))
            ca = clips[a]
            i = int(rng.integers(len(ca["X"])))
            if label == 1:
                members = by_group[ca["group"]]
                b = members[int(rng.integers(len(members)))]
            else:
                others = [g for g in names if g != ca["group"]]
                if not others:
                    break
                g2 = others[int(rng.integers(len(others)))]
                b = by_group[g2][int(rng.integers(len(by_group[g2])))]
            cb = clips[b]
            j = int(rng.integers(len(cb["X"])))
            if label == 1 and a == b and abs(ca["T"][i] - cb["T"][j]) < WIN_S:
                continue
            Xi.append(ca["X"][i])
            Xj.append(cb["X"][j])
            y.append(label)
            got += 1
    return np.array(Xi), np.array(Xj), np.array(y)


def fit_verifier(clips, n_per_class=2500):
    allX = np.vstack([c["X"] for c in clips])
    gs = np.maximum(allX.std(axis=0), MIN_SCALE)
    Xi, Xj, y = sample_pairs(clips, np.random.default_rng(0), n_per_class)
    if len(set(y.tolist())) < 2:
        raise ValueError("Not enough different pairs to learn from. Add more videos or longer videos.")
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced",
                                   random_state=0, n_jobs=1)
    model.fit(pair_vec(Xi, Xj, gs), y)
    return model, gs


def ai_scores(model, gs, XA, XB):
    """For every window of B: average chance (0-1) that it shows the same walking as video A."""
    if len(XA) > 60:
        XA = XA[np.random.default_rng(0).choice(len(XA), 60, replace=False)]
    nA, nB = len(XA), len(XB)
    ia, ib = np.repeat(np.arange(nA), nB), np.tile(np.arange(nB), nA)
    p = model.predict_proba(pair_vec(XA[ia], XB[ib], gs))[:, list(model.classes_).index(1)]
    return p.reshape(nA, nB).mean(axis=0)


def metrics(rows):
    """rows = [(is_genuine_pair, accepted_as_match)] -> dict of error measures."""
    tp = sum(1 for g, a in rows if g and a)
    fn = sum(1 for g, a in rows if g and not a)
    fp = sum(1 for g, a in rows if not g and a)
    tn = sum(1 for g, a in rows if not g and not a)
    tmr = tp / (tp + fn) if tp + fn else 0.0
    fmr = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "tmr": tmr, "fmr": fmr, "bal": (tmr + 1 - fmr) / 2,
            "acc": (tp + tn) / max(1, len(rows)), "prec": prec, "rec": tmr,
            "f1": 2 * prec * tmr / (prec + tmr) if prec + tmr else 0.0}


def evaluate(clips, log):
    """Honest test: every pair of videos is judged by a model that has seen NEITHER video."""
    n = len(clips)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    if len(pairs) > 40:
        sel = np.random.default_rng(1).choice(len(pairs), 40, replace=False)
        pairs = [pairs[k] for k in sorted(sel)]
    ai_rows, st_rows = [], []
    for k, (a, b) in enumerate(pairs):
        train = [c for t, c in enumerate(clips) if t not in (a, b)]
        if len({c["group"] for c in train}) < 2:
            continue
        try:
            model, gs = fit_verifier(train, n_per_class=1500)
        except ValueError:
            continue
        p = float(ai_scores(model, gs, clips[a]["X"], clips[b]["X"]).mean())
        genuine = clips[a]["group"] == clips[b]["group"]
        ai_rows.append((genuine, p >= 0.5))
        st_rows.append((genuine, compare_clips(clips[a]["X"], clips[b]["X"], 1.0)["match"]))
        if (k + 1) % 5 == 0:
            log(f"   tested {k + 1} of {len(pairs)} video pairs ...")
    n_gen = sum(1 for g, _ in ai_rows if g)
    n_imp = len(ai_rows) - n_gen
    if n_gen == 0 or n_imp == 0:
        return None
    return {"n_gen": n_gen, "n_imp": n_imp, "ai": metrics(ai_rows), "stat": metrics(st_rows)}


def load_library(log):
    clips = []
    for g in sorted(os.listdir(LIB_DIR)):
        gdir = os.path.join(LIB_DIR, g)
        if not os.path.isdir(gdir):
            continue
        for fn in sorted(os.listdir(gdir)):
            if fn.lower().endswith(VIDEO_EXT):
                log(f"Reading {g}/{fn} ...")
                r = extract_clip(os.path.join(gdir, fn))
                n = len(r["X"])
                log(f"   {n} clean windows." + ("" if n >= 2 else " NOT USED. " + diag_text(r)))
                if n >= 2:
                    clips.append({"group": g, "name": fn, "X": r["X"], "T": r["T"]})
    return clips


def build_report(b):
    ev = b["eval"]
    r = ["GAITGUARD AI REPORT", "=" * 52,
         f"Library: {len(b['groups'])} walking groups, {b['n_clips']} videos, {b['n_windows']} clean windows"]
    for g, (nv, nw) in b["groups"].items():
        r.append(f"   {g}: {nv} video(s), {nw} windows")
    r += ["", "Model: Random Forest on PAIRS of windows (fixed random seed, so the same library always gives the same model).",
          "It learns which gait differences mean 'same walking' and which mean 'different walking'.", ""]
    if ev is None:
        r += ["VALIDATION: not possible yet. To test the AI honestly you need at least 2 walking groups and",
              "at least 2 videos in one group (so that 'same walking' pairs exist in different videos).", ""]
    else:
        r += [f"VALIDATION on video pairs the model never saw ({ev['n_gen']} same-walking pairs, {ev['n_imp']} different-walking pairs):",
              f"{'':26}{'Statistical':>12}{'AI':>10}"]
        for label, key in (("True match rate", "tmr"), ("False match rate", "fmr"), ("Balanced accuracy", "bal"),
                           ("Accuracy", "acc"), ("Precision", "prec"), ("F1-score", "f1")):
            r.append(f"{label:<26}{ev['stat'][key] * 100:11.1f}%{ev['ai'][key] * 100:9.1f}%")
        for name, m in (("Statistical", ev["stat"]), ("AI", ev["ai"])):
            r.append(f"{name} confusion: same pairs accepted {m['tp']}, rejected {m['fn']} | "
                     f"different pairs accepted {m['fp']}, rejected {m['tn']}")
        r.append("")
    r.append("FINAL DECISION USES: " + ("AI" if b["use_ai"] else "STATISTICAL CHECK") + " - " + b["why"])
    r += ["", "Which features the AI relies on most:"]
    for i in np.argsort(b["w"])[::-1]:
        r.append(f"   {FNAMES[i]:<28}{b['w'][i] * 100:5.1f}%")
    r += ["", "Notes: results from a few videos by few people are only a first indication. Use the same camera",
          "position for all videos, and test with videos that were not used for training."]
    return "\n".join(r)


def save_plot(b):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(b["w"])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh([FNAMES[i] for i in order], b["w"][order] * 100)
        ax.set_xlabel("Importance (%)")
        ax.set_title("Which gait features the AI relies on")
        fig.tight_layout()
        fig.savefig(IMP_PNG, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def train_verifier(log):
    clips = load_library(log)
    groups = {}
    for c in clips:
        v, w = groups.get(c["group"], (0, 0))
        groups[c["group"]] = (v + 1, w + len(c["X"]))
    if len(groups) < 2:
        raise ValueError("The AI needs at least TWO different walking groups with usable videos, for example "
                         "'patient_style' and 'normal_style'. Use 'Add videos to AI library' first.")
    log("Testing on pairs of videos the model has never seen ...")
    ev = evaluate(clips, log)
    log("Training the final model on all videos ...")
    model, gs = fit_verifier(clips, n_per_class=2500)
    imp = model.feature_importances_
    w = imp[:NF] + imp[NF:]
    w = w / w.sum()
    if ev is None:
        use_ai = False
        why = "there were not enough videos to test the AI on unseen pairs, so it is shown for information only (add a second video of the same walking and two groups)."
    elif ev["ai"]["bal"] >= ev["stat"]["bal"]:
        use_ai = True
        why = f"on unseen video pairs the AI reached {ev['ai']['bal'] * 100:.0f}% balanced accuracy against {ev['stat']['bal'] * 100:.0f}% for the statistical check."
    else:
        use_ai = False
        why = f"on unseen video pairs the AI reached only {ev['ai']['bal'] * 100:.0f}% balanced accuracy against {ev['stat']['bal'] * 100:.0f}% for the statistical check, so the statistical check decides."
    bundle = {"model": model, "gs": gs, "w": w, "eval": ev, "use_ai": use_ai, "why": why, "groups": groups,
              "n_clips": len(clips), "n_windows": int(sum(len(c["X"]) for c in clips))}
    bundle["report"] = build_report(bundle)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(bundle["report"])
    save_plot(bundle)
    return bundle


def load_bundle():
    try:
        with open(MODEL_PATH, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


# ----------------------------------------------------------------------------- explanation texts
def short(name):
    return re.sub(r"\s*\(.*?\)", "", name)


def narrate(r, nA, nB, tol, same, ai):
    order = np.argsort(r["z"])
    close = [short(FNAMES[i]) for i in order[:3] if r["feat_ok"][i]]
    far = [i for i in order[::-1][:3] if not r["feat_ok"][i]]
    n_ok, n_win = int(r["feat_ok"].sum()), int(r["win_ok"].sum())
    L = ["WHAT THE SYSTEM FOUND",
         f"1. Video A (reference) gave {nA} clean walking windows and Video B (test) gave {nB}.",
         f"2. Statistical check: {n_ok} of {NF} gait features are close in both videos ({r['fmr'] * 100:.0f}%, needed "
         f"{FEAT_NEEDED * 100:.0f}%) and {n_win} of {nB} windows of B look like A ({r['wmr'] * 100:.0f}%, needed "
         f"{WIN_NEEDED * 100:.0f}%) -> {'MATCH' if r['match'] else 'NO MATCH'}."]
    if close:
        L.append("3. Most alike: " + ", ".join(close) + ".")
    if far:
        L.append("4. Most different: " + ", ".join(
            f"{short(FNAMES[i])} (A {r['ma'][i]:.2f} vs B {r['mb'][i]:.2f})" for i in far) + ".")
    if ai is not None:
        L.append(f"5. AI check: the trained model compares pairs of windows and rates the chance that both videos show "
                 f"the same walking at {ai['p'] * 100:.0f}% on average (needed {ai['thr'] * 100:.0f}%) -> "
                 f"{'MATCH' if ai['match'] else 'NO MATCH'}.")
        L.append("   " + ("The AI decides the final answer because " if ai["used"] else
                          "The AI is shown for information only because ") + ai["why"])
        final = ai["match"] if ai["used"] else r["match"]
    else:
        L.append("5. No AI model is trained yet, so the statistical check decides. (Add videos to the AI library and train for more accuracy.)")
        final = r["match"]
    L.append("RESULT: MATCH - the two videos show the same walking pattern." if final else
             "RESULT: NO MATCH - the walking patterns differ.")
    if same:
        L.append("Note: Video A and Video B are the same file, so a match is expected. This only proves the program works.")
    return "\n".join(L)


HOW_IT_WORKS = [
    ("h", "WHAT THIS PROGRAM DOES"),
    ("p", "It checks whether the walking pattern in two videos (or in a live camera) is the SAME. In the final product, Video A is the dementia patient's walking video and the live camera watches the ward door. When the walking in the camera matches the patient, the door locks and the buzzer rings."),
    ("h", "HOW IT WORKS - 5 STEPS"),
    ("p", "1. VIDEO IN: each video is read frame by frame, about 30 pictures every second."),
    ("p", "2. BODY POINTS: an AI model called MediaPipe BlazePose finds 33 points on the body (shoulders, hips, knees, ankles, toes) in every picture. This is called pose estimation."),
    ("p", "3. WALKING WINDOWS: the walking is cut into 4-second pieces called windows. Windows where the person is turning, standing, cut off at the edge of the picture, or wrongly tracked are thrown away, so only clean walking is used."),
    ("p", "4. 13 GAIT FEATURES: for every window the program measures 13 numbers that describe the walking, such as speed, step length, knee bend and foot lift. Every distance is divided by the person's torso length, so it does not matter how far the person stands from the camera."),
    ("p", "5. COMPARE: two checks are made. The STATISTICAL check compares the average of every feature in Video A with Video B; a feature matches when the difference is smaller than its normal variation (Tolerance slider). The AI check (below) gives a second opinion."),
    ("h", "THE MACHINE-LEARNING PART"),
    ("p", "Library: with 'Add videos to AI library' you give every walking style or person a group name (for example patient_style, normal_style, teammate). Videos with the same group name show the same walking."),
    ("p", "Training: a Random Forest learns from PAIRS of windows. A pair from the same group means 'same walking', a pair from different groups means 'different walking'. In this way it learns which differences matter (for example foot lift) and which are only natural variation."),
    ("p", "Honest testing: every pair of videos is judged by a model that has never seen either video. The AI report shows the true match rate, false match rate, balanced accuracy, precision and F1-score of the AI and of the statistical check side by side."),
    ("p", "Safety rule: the AI decides the final answer ONLY if it does at least as well as the statistical check on these unseen video pairs. Otherwise the statistical check decides and the AI is shown for information. So adding the AI can never silently make the result worse."),
    ("p", "Why it does not flip between runs: one fixed model type with a fixed random seed is used, so the same library always gives the same trained model. (An older version chose between two model types, which could change from run to run.)"),
    ("h", "HOW TO READ THE SCREEN"),
    ("p", "Comparison tab: one row per feature with the value in Video A and Video B. 'Diff' is the difference in units of normal variation: 0 = identical, 1 = about one normal step of variation. Green = MATCH, red = DIFFERENT. 'AI weight' shows how much the AI relies on that feature. 'What it means' explains it in plain words."),
    ("p", "Windows tab: every 4-second window of Video B with its score. AI result: chance of the same walking in %. Statistical result: distance from Video A (lower = closer)."),
    ("p", "Similarity bar: the AI chance of the same walking when the AI decides, otherwise the average of features matched and windows matched. The statistical check says MATCH only if at least 70% of features AND 60% of windows match."),
    ("p", "Tolerance slider: strictness of the statistical check. AI threshold: how sure the AI must be (50% by default)."),
    ("h", "LIVE MODE"),
    ("p", "The camera is compared with Video A once per second. If 3 of the last 5 checks match, the status turns red and the alarm sounds. Voting stops one wrong frame from causing a false alarm."),
    ("h", "DELETE ALL SAVED DATA"),
    ("p", "This button deletes everything the program stored: the cache of extracted features, the AI library copies, the trained model and the reports. Your original video files are never touched. You must type DELETE to confirm."),
    ("h", "BE HONEST ABOUT THE LIMITS"),
    ("p", "- It needs a side view with the whole body visible, one person at a time, and the same camera position for all videos."),
    ("p", "- It has only been tested with healthy volunteers acting, not with real dementia patients."),
    ("p", "- A MATCH means the two walks are SIMILAR. The AI learns only from the videos in your library, so a small library gives only a first indication."),
    ("p", "- The door lock and buzzer are part of the final product design; this program shows the detection part."),
    ("h", "QUESTIONS A JUDGE MAY ASK"),
    ("p", "Why side view? Because step length, knee bend and foot lift are visible from the side."),
    ("p", "Why divide by torso length? So that a person closer to or farther from the camera gives the same numbers."),
    ("p", "Is the AI really more accurate? The AI report answers this with numbers on video pairs the AI has never seen, next to the statistical check. If the AI does worse, the program says so and does not use it."),
    ("p", "Is the result a medical accuracy figure? No. It says how similar two walks are. Accuracy on real patients needs a clinical study."),
]


def beep():
    try:
        import winsound
        winsound.Beep(1200, 400)
    except ImportError:
        print("\a", end="", flush=True)


# ----------------------------------------------------------------------------- GUI
class App:
    def __init__(self, root):
        self.root = root
        root.title("GaitGuard v4 - Gait Match with AI (are two walking videos the same?)")
        self.q = queue.Queue()
        self.res = {"A": None, "B": None}
        self.path = {"A": None, "B": None}
        self.bundle = load_bundle()
        self.mode = "idle"
        self.cap = self.pose = None
        self.buf = deque()
        self.t0 = self.last_eval = self.last_beep = 0.0
        self.last_mismatch_notification = 0.0
        self.votes = deque(maxlen=VOTES_N)
        self.mcu = MCULink(SERIAL_PORT, SERIAL_BAUD)
        self.photo = None
        self.row_n = 0
        self.tol = tk.DoubleVar(value=1.0)
        self.ai_thr = tk.DoubleVar(value=0.5)
        self.cam = tk.StringVar(value="0")
        self.last_say = ""
        self.build_ui()
        self.refresh_lib()
        self.set_ai_text(self.bundle["report"] if self.bundle else "No AI model yet.\n\n1. Click 'Add videos to AI library' and give each walking style/person a group name.\n2. Click 'Train AI'.")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.poll()
        self.tick()

    # ---- layout
    def build_ui(self):
        left = tk.Frame(self.root)
        left.grid(row=0, column=0, padx=8, pady=8, sticky="n")
        self.video = tk.Label(left, bg="black", width=80, height=16)
        self.video.pack()
        self.verdict = tk.Label(left, text="Choose Video A (reference) to start", font=("Arial", 18, "bold"),
                                width=36, height=2, bg="#dddddd", wraplength=640)
        self.verdict.pack(fill="x", pady=6)
        row = tk.Frame(left)
        row.pack(fill="x")
        tk.Label(row, text="Similarity:", font=("Arial", 11)).pack(side="left")
        self.bar = ttk.Progressbar(row, length=300, maximum=100)
        self.bar.pack(side="left", padx=6)
        self.bar_lbl = tk.Label(row, text="-", font=("Arial", 10, "bold"))
        self.bar_lbl.pack(side="left")
        self.files = tk.Label(left, text="", font=("Arial", 10), justify="left", anchor="w")
        self.files.pack(fill="x", pady=2)
        self.libinfo = tk.Label(left, text="", font=("Arial", 10), justify="left", anchor="w", fg="#1f3a68")
        self.libinfo.pack(fill="x")
        self.say = tk.Text(left, width=80, height=12, wrap="word", font=("Arial", 11), bg="#fbfbf0")
        self.say.pack()
        self.update_files()
        self.set_say("Choose Video A (the reference walking video), then Video B (the video to test).\n"
                     "The result and a plain-English explanation will appear here.")

        right = tk.Frame(self.root)
        right.grid(row=0, column=1, padx=8, pady=8, sticky="n")
        bf = tk.Frame(right)
        bf.pack()
        buttons = [("1. Choose Video A (reference)", lambda: self.load("A")),
                   ("2. Choose Video B (test)", lambda: self.load("B")),
                   ("3. Compare A and B", self.recompare), ("4. Live monitoring (camera vs A)", self.start_live),
                   ("5. Add videos to AI library", self.add_library), ("6. Train AI", self.train),
                   ("Stop", self.stop), ("Copy explanation", self.copy_say)]
        for i, (t, c) in enumerate(buttons):
            tk.Button(bf, text=t, command=c, width=30).grid(row=i // 2, column=i % 2, padx=3, pady=3)
        tk.Button(bf, text="DELETE ALL SAVED DATA", command=self.delete_all, width=62, bg="#f4b6b6",
                  fg="#7a0000", font=("Arial", 9, "bold")).grid(row=4, column=0, columnspan=2, pady=5)

        cf = tk.Frame(right)
        cf.pack(pady=2)
        tk.Label(cf, text="Tolerance").pack(side="left")
        tk.Scale(cf, from_=0.5, to=2.5, resolution=0.1, orient="horizontal", variable=self.tol, length=110,
                 command=lambda v: self.recompare(quiet=True)).pack(side="left")
        tk.Label(cf, text=" AI threshold").pack(side="left")
        tk.Scale(cf, from_=0.3, to=0.9, resolution=0.05, orient="horizontal", variable=self.ai_thr, length=110,
                 command=lambda v: self.recompare(quiet=True)).pack(side="left")
        tk.Label(cf, text=" Camera #").pack(side="left")
        ttk.Spinbox(cf, from_=0, to=5, width=3, textvariable=self.cam).pack(side="left")

        self.nb = ttk.Notebook(right)
        self.nb.pack(pady=4)
        f1, f2, f3, f4, f5 = (tk.Frame(self.nb) for _ in range(5))
        self.nb.add(f1, text="Comparison")
        self.nb.add(f2, text="Windows")
        self.nb.add(f3, text="AI results")
        self.nb.add(f4, text="How it works")
        self.nb.add(f5, text="Log")

        cols = [("f", 150, "Feature"), ("a", 58, "Video A"), ("b", 58, "Video B"), ("z", 42, "Diff"),
                ("r", 72, "Result"), ("w", 62, "AI weight"), ("m", 260, "What it means")]
        self.ft = ttk.Treeview(f1, columns=[c[0] for c in cols], show="headings", height=14)
        for c, w, t in cols:
            self.ft.heading(c, text=t)
            self.ft.column(c, width=w, anchor="w" if c in ("f", "m") else "center")
        self.ft.tag_configure("ok", background="#d6f5d6")
        self.ft.tag_configure("bad", background="#ffd6d6")
        self.ft.pack()
        tk.Label(f1, text="Diff = difference in units of normal variation (0 = identical).  Green = match, red = different.",
                 font=("Arial", 9)).pack(anchor="w")

        self.wt = ttk.Treeview(f2, columns=("n", "t", "d", "r"), show="headings", height=14)
        for c, w, t in [("n", 45, "#"), ("t", 150, "Time (s)"), ("d", 220, "Score"), ("r", 160, "Result")]:
            self.wt.heading(c, text=t)
            self.wt.column(c, width=w, anchor="center")
        self.wt.tag_configure("ok", background="#d6f5d6")
        self.wt.tag_configure("bad", background="#ffd6d6")
        self.wt.pack()

        self.ai_box = tk.Text(f3, width=98, height=22, wrap="none", font=("Consolas", 9))
        self.ai_box.pack()

        hw = tk.Text(f4, width=95, height=22, wrap="word", font=("Arial", 10))
        sb = ttk.Scrollbar(f4, command=hw.yview)
        hw.config(yscrollcommand=sb.set)
        hw.tag_configure("h", font=("Arial", 11, "bold"), foreground="#1f3a68", spacing1=8)
        for kind, text in HOW_IT_WORKS:
            hw.insert("end", text + "\n", kind)
        hw.config(state="disabled")
        hw.pack(side="left")
        sb.pack(side="left", fill="y")

        self.log_box = tk.Text(f5, width=98, height=22, font=("Consolas", 9), wrap="word")
        self.log_box.pack()

    # ---- helpers
    def log(self, msg):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    def set_verdict(self, text, color):
        self.verdict.config(text=text, bg=color)

    def set_bar(self, frac, text):
        self.bar["value"] = frac * 100
        self.bar_lbl.config(text=text)

    def set_say(self, text):
        self.last_say = text
        self.say.delete("1.0", "end")
        self.say.insert("end", text)

    def set_ai_text(self, text):
        self.ai_box.delete("1.0", "end")
        self.ai_box.insert("end", text)

    def copy_say(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_say)
        self.log("Explanation copied to the clipboard.")

    def show(self, frame):
        h, w = frame.shape[:2]
        s = 640 / w
        frame = cv2.resize(frame, (640, int(h * s)))
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video.config(image=self.photo, width=640, height=int(h * s))

    def update_files(self):
        def line(k, name):
            p, r = self.path[k], self.res[k]
            return f"Video {k} ({name}): " + (f"{os.path.basename(p)} - {len(r['X'])} clean windows" if r else "not chosen")
        self.files.config(text=line("A", "reference") + "\n" + line("B", "test"))

    def refresh_lib(self):
        groups = {}
        for g in sorted(os.listdir(LIB_DIR)):
            gdir = os.path.join(LIB_DIR, g)
            if os.path.isdir(gdir):
                groups[g] = len([f for f in os.listdir(gdir) if f.lower().endswith(VIDEO_EXT)])
        lib = ", ".join(f"{g} ({n})" for g, n in groups.items()) or "empty"
        if self.bundle:
            mt = "AI trained - " + ("decides the answer" if self.bundle["use_ai"] else "shown for information only")
        else:
            mt = "AI not trained"
        self.libinfo.config(text=f"AI library: {lib}\n{mt}")

    def fill_rows(self, a=None, b=None, z=None, ok=None):
        self.ft.delete(*self.ft.get_children())
        w = self.bundle["w"] if self.bundle else None
        for i, (_, name, meaning) in enumerate(FEATURES):
            va = f"{a[i]:.2f}" if a is not None else "-"
            vb = f"{b[i]:.2f}" if b is not None else "-"
            vz = f"{z[i]:.1f}" if z is not None else "-"
            res_txt = "-" if ok is None else ("MATCH" if ok[i] else "DIFFERENT")
            tag = "" if ok is None else ("ok" if ok[i] else "bad")
            vw = f"{w[i] * 100:.0f}%" if w is not None else "-"
            self.ft.insert("", "end", values=(name, va, vb, vz, res_txt, vw, meaning), tags=(tag,))

    # ---- loading videos (background thread)
    def load(self, which):
        path = filedialog.askopenfilename(title=f"Choose Video {which}",
                                          filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All", "*.*")])
        if not path:
            return
        self.stop()
        self.set_verdict(f"PROCESSING VIDEO {which}...", "#ffe08a")
        self.log(f"Reading Video {which}: {os.path.basename(path)}")

        def job():
            try:
                res = extract_clip(path, progress_cb=lambda f: self.q.put(("progress", which, f)))
                self.q.put(("extracted", which, res, path))
            except Exception as e:
                traceback.print_exc()
                self.q.put(("error", str(e)))
        threading.Thread(target=job, daemon=True).start()

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    self.set_verdict(f"PROCESSING VIDEO {msg[1]}... {int(msg[2] * 100)}%", "#ffe08a")
                elif kind == "error":
                    self.set_verdict("Something went wrong", "#dddddd")
                    messagebox.showerror("Error", msg[1])
                elif kind == "extracted":
                    self.on_extracted(msg[1], msg[2], msg[3])
                elif kind == "log":
                    self.log(msg[1])
                elif kind == "trained":
                    self.bundle = msg[1]
                    self.set_ai_text(self.bundle["report"])
                    self.refresh_lib()
                    self.set_verdict("AI TRAINED", "#b9f6ca")
                    self.nb.select(2)
                    ev = self.bundle["eval"]
                    if ev:
                        txt = (f"Tested on {ev['n_gen']} same-walking and {ev['n_imp']} different-walking video pairs.\n"
                               f"AI balanced accuracy: {ev['ai']['bal'] * 100:.0f}%\n"
                               f"Statistical check: {ev['stat']['bal'] * 100:.0f}%\n\n")
                    else:
                        txt = "Not enough videos to test the AI on unseen pairs.\n\n"
                    messagebox.showinfo("Training finished", txt + "The AI " + ("decides" if self.bundle["use_ai"] else "is shown for information only") +
                                        ": " + self.bundle["why"])
                    self.recompare(quiet=True)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def on_extracted(self, which, res, path):
        self.res[which], self.path[which] = res, path
        self.update_files()
        self.log(f"Video {which}: {len(res['X'])} clean windows. " + diag_text(res))
        if len(res["X"]) == 0:
            self.set_verdict(f"Video {which}: no usable walking found", "#ffe08a")
            self.set_say(f"No clean walking windows were found in Video {which}.\n" + diag_text(res) +
                         "\nRecord again: side view, whole body in the picture, walking steadily across the frame "
                         "for at least 10 seconds, good light.")
            return
        A, B = self.res["A"], self.res["B"]
        if A is not None and B is not None and len(A["X"]) and len(B["X"]):
            self.recompare()
        else:
            X = res["X"].mean(0)
            self.fill_rows(a=X if which == "A" else None, b=X if which == "B" else None)
            self.set_verdict("Now choose Video B (the video to test)" if which == "A" else "Now choose Video A (reference)", "#dddddd")
            self.set_bar(0, "-")
            self.nb.select(0)

    # ---- comparison
    def recompare(self, quiet=False):
        A, B = self.res["A"], self.res["B"]
        if A is None or B is None or len(A["X"]) == 0 or len(B["X"]) == 0:
            if not quiet:
                messagebox.showinfo("Compare", "Choose both Video A and Video B first.")
            return
        tol = self.tol.get()
        r = compare_clips(A["X"], B["X"], tol)
        same = file_sig(self.path["A"]) == file_sig(self.path["B"])
        ai = None
        if self.bundle:
            sc = ai_scores(self.bundle["model"], self.bundle["gs"], A["X"], B["X"])
            thr = self.ai_thr.get()
            ai = {"p": float(sc.mean()), "match": float(sc.mean()) >= thr, "used": self.bundle["use_ai"],
                  "thr": thr, "why": self.bundle["why"], "scores": sc}
        self.fill_rows(r["ma"], r["mb"], r["z"], r["feat_ok"])
        self.wt.delete(*self.wt.get_children())
        use_ai = ai is not None and ai["used"]
        self.wt.heading("d", text="AI: chance of same walking" if use_ai else "Distance from A (lower = closer)")
        for i in range(len(B["X"])):
            t0 = B["T"][i]
            if use_ai:
                ok = bool(ai["scores"][i] >= ai["thr"])
                score = f"{ai['scores'][i] * 100:.0f}%"
            else:
                ok = bool(r["win_ok"][i])
                score = f"{r['dist'][i]:.2f}"
            self.wt.insert("", "end", values=(i + 1, f"{t0:.1f} - {t0 + WIN_S:.1f}", score, "MATCH" if ok else "different"),
                           tags=("ok" if ok else "bad",))
        if use_ai:
            final, sim = ai["match"], ai["p"]
            self.set_bar(sim, f"{sim * 100:.0f}% same-walking chance (AI)")
        else:
            final, sim = r["match"], r["sim"]
            self.set_bar(sim, f"{sim * 100:.0f}%  (features {r['fmr'] * 100:.0f}% | windows {r['wmr'] * 100:.0f}%)")
        self.set_verdict("MATCH - same walking pattern" if final else "NO MATCH - different walking pattern",
                         "#b9f6ca" if final else "#ffd9a8")
        self.set_say(narrate(r, len(A["X"]), len(B["X"]), tol, same, ai))
        if final and not quiet:
            self.mcu.pulse(4.0)
        if not quiet:
            self.nb.select(0)

    # ---- AI library and training
    def add_library(self):
        paths = filedialog.askopenfilenames(title="Select side-view walking videos of ONE walking style/person",
                                            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All", "*.*")])
        if not paths:
            return
        group = simpledialog.askstring("Group name",
                                       "Group name for these videos (same person + same walking style = same group).\n"
                                       "Examples: patient_style, normal_style, teammate")
        if not group:
            return
        group = re.sub(r"[^A-Za-z0-9_-]", "", group.strip().replace(" ", "_")) or "group"
        gdir = os.path.join(LIB_DIR, group)
        os.makedirs(gdir, exist_ok=True)
        for p in paths:
            shutil.copy(p, os.path.join(gdir, re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(p))))
            self.log(f"Added to library: {group}/{os.path.basename(p)}")
        self.refresh_lib()

    def train(self):
        self.stop()
        self.set_verdict("TRAINING AI...", "#ffe08a")
        self.nb.select(4)

        def job():
            try:
                b = train_verifier(lambda m: self.q.put(("log", m)))
                self.q.put(("trained", b))
            except Exception as e:
                traceback.print_exc()
                self.q.put(("error", str(e)))
        threading.Thread(target=job, daemon=True).start()

    # ---- delete everything that was stored
    def delete_all(self):
        if not messagebox.askyesno("Delete ALL saved data",
                                   "This permanently deletes everything the program saved:\n\n"
                                   " - the cache of extracted features\n - the AI library (copies of videos)\n"
                                   " - the trained AI model and the reports\n\n"
                                   "Your original video files are NOT touched.\n\nContinue?", icon="warning"):
            return
        if simpledialog.askstring("Confirm", "Type DELETE (capital letters) to confirm:") != "DELETE":
            messagebox.showinfo("Cancelled", "Nothing was deleted.")
            return
        self.stop()
        for d in (CACHE_DIR, LIB_DIR, MODEL_DIR):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        for f in (REPORT_PATH, IMP_PNG):
            try:
                os.remove(f)
            except OSError:
                pass
        self.res, self.path, self.bundle = {"A": None, "B": None}, {"A": None, "B": None}, None
        self.ft.delete(*self.ft.get_children())
        self.wt.delete(*self.wt.get_children())
        self.log_box.delete("1.0", "end")
        self.set_say("All saved data was deleted.\nChoose Video A (the reference walking video) to start again.")
        self.set_ai_text("No AI model yet.\n\n1. Click 'Add videos to AI library' and give each walking style/person a group name.\n2. Click 'Train AI'.")
        self.set_verdict("All saved data deleted", "#dddddd")
        self.set_bar(0, "-")
        self.update_files()
        self.refresh_lib()
        messagebox.showinfo("Done", "All saved data was deleted. Your original video files were not touched.")

    # ---- live monitoring (camera compared with Video A)
    def start_live(self):
        A = self.res["A"]
        if A is None or len(A["X"]) == 0:
            messagebox.showinfo("Live monitoring", "Choose Video A (the reference) first.")
            return
        self.stop()
        idx = int(self.cam.get() or 0)
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Camera", f"Could not open camera #{idx}. Try another number "
                                 "(DroidCam is often 1 or 2) and close apps that use the camera.")
            return
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.XA = A["X"]
        self.ma, self.sa = A["X"].mean(0), A["X"].std(0)
        self.buf.clear()
        self.votes.clear()
        self.last_mismatch_notification = 0.0
        self.row_n = 0
        self.wt.delete(*self.wt.get_children())
        self.live_ai = bool(self.bundle and self.bundle["use_ai"])
        self.wt.heading("d", text="AI: chance of same walking" if self.live_ai else "Distance from A (lower = closer)")
        self.t0, self.last_eval = time.time(), 0.0
        self.mode = "live"
        self.nb.select(0)
        self.set_say("LIVE MONITORING\nEvery second the last 4 seconds of walking are compared with Video A"
                     + (" (AI decides)." if self.live_ai else " (statistical check decides).")
                     + " If 3 of the last 5 checks match, the alarm sounds.")
        self.log("Live monitoring started.")

    def stop(self):
        self.mode = "idle"
        self.mcu.send(False)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.pose is not None:
            self.pose.close()
            self.pose = None

    def on_close(self):
        self.stop()
        self.mcu.close()
        self.root.destroy()

    def tick(self):
        if self.mode == "live" and self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                if frame.shape[1] > MAX_WIDTH:
                    s = MAX_WIDTH / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=s, fy=s)
                arr, res = run_pose(frame, self.pose)
                now = time.time() - self.t0
                if res.pose_landmarks:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if arr is not None:
                    self.buf.append((now, arr))
                while self.buf and now - self.buf[0][0] > WIN_S:
                    self.buf.popleft()
                if now - self.last_eval >= 1.0:
                    self.last_eval = now
                    self.evaluate_live(now)
                self.show(frame)
        self.root.after(15, self.tick)

    def evaluate_live(self, now):
        decision, text = None, ""
        if len(self.buf) >= 20 and (self.buf[-1][0] - self.buf[0][0]) >= 0.7 * WIN_S:
            ts = np.array([b[0] for b in self.buf])
            L = np.array([b[1] for b in self.buf])
            f = compute_features(ts, L)
            if f is not None:
                tol = self.tol.get()
                s = spread(self.sa, self.sa, self.ma, f)
                z = np.abs(f - self.ma) / s
                if self.live_ai:
                    p = float(ai_scores(self.bundle["model"], self.bundle["gs"], self.XA, f.reshape(1, -1)).mean())
                    decision, text = p >= self.ai_thr.get(), f"{p * 100:.0f}%"
                else:
                    dval = float(np.sqrt(np.mean(((f - self.ma) / s) ** 2)))
                    decision, text = dval <= 2 * tol, f"{dval:.2f}"
                self.fill_rows(self.ma, f, z, z <= tol)
                self.row_n += 1
                self.wt.insert("", 0, values=(self.row_n, f"{now:.0f}", text, "MATCH" if decision else "different"),
                               tags=("ok" if decision else "bad",))
                if len(self.wt.get_children()) > 100:
                    self.wt.delete(self.wt.get_children()[-1])
        self.votes.append(decision)
        hits = sum(1 for v in self.votes if v is True)
        self.set_bar(hits / VOTES_N, f"{hits}/{VOTES_N} recent checks match")
        self.mcu.send(hits >= VOTES_NEED)      # buzzer ON while the pattern matches, OFF otherwise
        if hits >= VOTES_NEED:
            self.set_verdict("ALARM! MATCHING WALKING PATTERN", "#ff4d4d")
            if time.time() - self.last_beep > 1.2:
                self.last_beep = time.time()
                threading.Thread(target=beep, daemon=True).start()
        elif decision is None:
            self.set_verdict("Monitoring - nobody walking", "#e8e8e8")
        else:
            self.set_verdict(f"Monitoring - walking does not match ({text})", "#b9f6ca")
            now_wall = time.time()
            if now_wall - self.last_mismatch_notification >= NTFY_COOLDOWN_S:
                self.last_mismatch_notification = now_wall
                threading.Thread(target=send_mobile_notification,
                                 args=(f"Different walking pattern detected. GaitGuard score: {text}.",),
                                 daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
