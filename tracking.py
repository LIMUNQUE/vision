import os
import cv2
import time
import random
import threading
import queue
from ultralytics import YOLO
from PIL import Image
import piexif

# ————— Parámetros ajustables —————
video_path     = "trafficCam.mp4"
output_dir     = "cacao_crops"
line_position  = 320
offset         = 10
min_area       = 2000
VALID_LABELS   = ['healthy', 'car']
SAVE_LABEL     = 'car'
# —————————————————————————————————

os.makedirs(output_dir, exist_ok=True)

# Cola para guardado asincrónico
save_queue = queue.Queue()

def get_random_location():
    time.sleep(2)
    return random.uniform(-90, 90), random.uniform(-180, 180)

def worker():
    while True:
        item = save_queue.get()
        if item is None:
            break
        crop_orig, obj_id, label = item
        lat, lon = get_random_location()
        crop_rgb = cv2.cvtColor(crop_orig, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop_rgb)
        desc = f"lat:{lat:.6f},lon:{lon:.6f}"
        exif_dict = {"0th": {piexif.ImageIFD.ImageDescription: desc}, "Exif": {}, "1st": {}, "GPS": {}, "Interop": {}}
        exif_bytes = piexif.dump(exif_dict)
        filename = f"{label}_id{obj_id}.jpg"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            img.save(filepath, exif=exif_bytes)
        else:
            print(f"[SKIP] {filepath} ya existe.")
        save_queue.task_done()

t = threading.Thread(target=worker, daemon=True)
t.start()

model = YOLO("yolo12n.pt")
names = model.model.names

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] No se pudo abrir el video: {video_path}")
    exit()

cv2.namedWindow("Detección", cv2.WINDOW_NORMAL)

counted_ids = set()
counts = {label: 0 for label in VALID_LABELS}

try:
    while True:
        ret, orig_frame = cap.read()
        if not ret or orig_frame is None:
            break

        orig_h, orig_w = orig_frame.shape[:2]
        frame = cv2.resize(orig_frame, (640, 640))
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        det = results[0]

        vis = frame.copy()
        cv2.line(vis, (line_position, 0), (line_position, 640), (0, 255, 255), 2)

        if det.boxes.id is not None:
            ids = det.boxes.id.cpu().numpy()
            cls_idxs = det.boxes.cls.cpu().numpy()
            coords = det.boxes.xyxy.cpu().numpy()

            for (x1, y1, x2, y2), obj_id, cls_idx in zip(coords, ids, cls_idxs):
                label = names[int(cls_idx)]
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                w, h = x2 - x1, y2 - y1

                if w * h < min_area or label not in VALID_LABELS:
                    continue

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{label}-{obj_id}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                center_x = x1 + w // 2
                if obj_id not in counted_ids and (line_position - offset) < center_x < (line_position + offset):
                    counted_ids.add(obj_id)
                    counts[label] += 1

                    if label == SAVE_LABEL:
                        x1o = int(x1 / 640 * orig_w)
                        x2o = int(x2 / 640 * orig_w)
                        y1o = int(y1 / 640 * orig_h)
                        y2o = int(y2 / 640 * orig_h)
                        crop_orig = orig_frame[y1o:y2o, x1o:x2o]
                        if crop_orig.size > 0:
                            save_queue.put((crop_orig, obj_id, label))

        y0 = 50
        for lbl in VALID_LABELS:
            cv2.putText(vis, f"{lbl}: {counts[lbl]}", (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            y0 += 40

        cv2.imshow("Detección", vis)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        time.sleep(0.01)  # prevenir uso excesivo de CPU en Ubuntu

except KeyboardInterrupt:
    print("Interrumpido manualmente.")

finally:
    save_queue.put(None)
    t.join()
    cap.release()
    cv2.destroyAllWindows()