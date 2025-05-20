import os
import cv2
import time
import threading
import queue

# ROS2 / MAVROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

# YOLO / PIL / EXIF
from ultralytics import YOLO
from PIL import Image
import piexif

# ————— Parámetros ajustables —————
video_path     = "trafficCam.mp4"
output_dir     = "cacao_crops"
line_position  = 320
offset         = 10
min_area       = 2000            # px² en el frame 640×640
VALID_LABELS   = ['healthy', 'car']
SAVE_LABEL     = 'car'
# —————————————————————————————————

os.makedirs(output_dir, exist_ok=True)

# ——————————— Nodo ROS2 para GPS ———————————
class GPSLogger(Node):
    def __init__(self):
        super().__init__('gps_logger')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.subscription = self.create_subscription(
            NavSatFix,
            'mavros/global_position/global',
            self.listener_callback,
            qos_profile)
        self.lock = threading.Lock()
        self.latest_lat = None
        self.latest_lon = None

    def listener_callback(self, msg):
        with self.lock:
            self.latest_lat = msg.latitude
            self.latest_lon = msg.longitude

    def get_latest(self):
        with self.lock:
            return self.latest_lat, self.latest_lon

# ————— Cola y worker de guardado —————
save_queue = queue.Queue()

def worker(gps_node: GPSLogger):
    """Hilo que procesa la cola: obtiene ubicación real y guarda el crop con EXIF."""
    while True:
        item = save_queue.get()
        if item is None:
            break  # señal de terminación

        crop_orig, obj_id, label = item

        # Obtener lat/lon actuales
        lat, lon = gps_node.get_latest()
        if lat is None or lon is None:
            print("[WARN] Coordenadas GPS no disponibles, uso (0,0).")
            lat, lon = 0.0, 0.0

        # Preparar EXIF
        crop_rgb = cv2.cvtColor(crop_orig, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop_rgb)
        exif_dict = {"0th": {}, "Exif": {}, "1st": {}, "GPS": {}, "Interop": {}}
        desc = f"lat:{lat:.6f},lon:{lon:.6f}"
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = desc
        exif_bytes = piexif.dump(exif_dict)

        # Guardar archivo
        fn = f"{label}_id{obj_id}.jpg"
        fp = os.path.join(output_dir, fn)
        if not os.path.exists(fp):
            img.save(fp, exif=exif_bytes)
        else:
            print(f"[SKIP] {fp} ya existe.")

        save_queue.task_done()

# ————— Main de detección + ROS2 spin —————
def main():
    # Iniciar ROS2
    rclpy.init()
    gps_node = GPSLogger()

    # Lanzar spin de ROS2 en hilo aparte
    ros_thread = threading.Thread(target=rclpy.spin, args=(gps_node,), daemon=True)
    ros_thread.start()

    # Lanzar el worker de guardado
    t_worker = threading.Thread(target=worker, args=(gps_node,), daemon=True)
    t_worker.start()

    # Carga del modelo YOLO
    model = YOLO("yolo12n.pt")
    names = model.model.names

    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow("Detección y Conteo de Cacaos", cv2.WINDOW_NORMAL)
    counted_ids = set()
    counts = {label: 0 for label in VALID_LABELS}

    try:
        while cap.isOpened():
            ret, orig_frame = cap.read()
            if not ret:
                break

            orig_h, orig_w = orig_frame.shape[:2]
            frame = cv2.resize(orig_frame, (640, 640))
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                show=False  # desactiva la ventana interna de YOLO
            )
            det = results[0]

            vis = frame.copy()
            cv2.line(vis, (line_position, 0), (line_position, frame.shape[0]), (0, 255, 255), 2)

            if det.boxes.id is not None:
                ids      = det.boxes.id.cpu().numpy()
                cls_idxs = det.boxes.cls.cpu().numpy()
                coords   = det.boxes.xyxy.cpu().numpy()

                for (x1, y1, x2, y2), obj_id, cls_idx in zip(coords, ids, cls_idxs):
                    label = names[int(cls_idx)]
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    w, h = x2 - x1, y2 - y1

                    # Filtrar por área y etiqueta
                    if w * h < min_area or label not in VALID_LABELS:
                        continue

                    # Dibujar caja e ID
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"{label}-{obj_id}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Conteo al cruzar línea
                    center_x = x1 + w // 2
                    if obj_id not in counted_ids and (line_position - offset) < center_x < (line_position + offset):
                        counted_ids.add(obj_id)
                        counts[label] += 1

                        if label == SAVE_LABEL:
                            # Reescalar a coordenadas originales
                            x1o = int(x1 / 640 * orig_w)
                            x2o = int(x2 / 640 * orig_w)
                            y1o = int(y1 / 640 * orig_h)
                            y2o = int(y2 / 640 * orig_h)
                            crop_orig = orig_frame[y1o:y2o, x1o:x2o]

                            if crop_orig.size > 0:
                                save_queue.put((crop_orig, obj_id, label))

            # Mostrar contadores
            y0 = 50
            for lbl in VALID_LABELS:
                text = f"{lbl}: {counts[lbl]}"
                cv2.putText(vis, text, (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                y0 += 40

            cv2.imshow("Detección y Conteo de Cacaos", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Parar worker y ROS2
        save_queue.put(None)
        t_worker.join()
        rclpy.shutdown()
        ros_thread.join()

if __name__ == '__main__':
    main()
