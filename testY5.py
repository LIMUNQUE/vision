import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("best.onnx")
byte_tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
       for _, _, confidence, class_id, tracker_id
      in detections]
    return annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)

sv.process_video(source_path="prueba.mp4", target_path=f"result.mp4", callback=callback)