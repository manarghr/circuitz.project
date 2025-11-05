# ---- Real-time Emotion (CPU) using a single .pt file ----
import os, json, collections, cv2, torch
import torch.nn as nn
from torchvision import models, transforms
import mediapipe as mp

MODEL_PATH = r"C:\Users\sara-\Downloads\emotion_model_best.pt"  # <- your file
IMG_SIZE   = 224
LABELS     = ["happy","neutral","sad"]  # update if you changed order
CAM_INDEX  = 0
SMOOTH_N   = 7
MIN_DET_CONF = 0.6
device = "cpu"

# 1) Load model (TorchScript -> fallback to state_dict)
model = None
try:
    model = torch.jit.load(MODEL_PATH, map_location=device).eval()
    print(f"[model] Loaded TorchScript: {MODEL_PATH}")
except Exception as e:
    print("[model] Not TorchScript, trying state_dict…")
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 3)
    state = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
    model = m.eval()
    print(f"[model] Loaded state_dict into MobileNetV3: {MODEL_PATH}")

LABEL_MAP = {i: lbl for i, lbl in enumerate(LABELS)}

# 2) Preprocessing (must match training)
tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # ✅ match training normalization
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# 3) Face detector (MediaPipe)
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=MIN_DET_CONF
)

@torch.no_grad()
def classify_crop(bgr):
    x = tfm(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)  # CPU tensor
    p = torch.softmax(model(x), 1)[0]
    k = int(p.argmax())
    return k, float(p[k])

# 4) Webcam loop
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit(f"Cannot open webcam index {CAM_INDEX}")

recent = collections.deque(maxlen=max(1, SMOOTH_N))
print("Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    pred_id, conf = 1, 0.0  # default neutral
    if res.detections:
        d = res.detections[0]
        bb = d.location_data.relative_bounding_box
        x, y, bw, bh = int(bb.xmin*w), int(bb.ymin*h), int(bb.width*w), int(bb.height*h)
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        crop = frame[y:y2, x:x2]
        if crop.size > 0:
            pred_id, conf = classify_crop(crop)

        cv2.rectangle(frame, (x, y), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f"{LABEL_MAP[pred_id]} {conf:.2f}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # temporal smoothing
    recent.append(pred_id)
    smooth_id = max(set(recent), key=recent.count)
    cv2.putText(frame, f"SMOOTH: {LABEL_MAP[smooth_id]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Real-time Emotion (CPU)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
