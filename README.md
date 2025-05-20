# Video Tracking & Conteo de Objetos

Este proyecto permite realizar tracking y conteo automático de objetos (por enjemplo, carros o frutos) en videos usando modelos YOLO y ByteTrack. El sistema:

- Procesa video cuadro por cuadro, detectando y traqueando objetos de interés.
- Solo cuenta objetos que cruzan una línea virtual definida en el frame.
- Filtra y descarta objetos cuya bounding box es demasiado pequeña.
- Permite guardar recortes (crops) de los objetos detectados con metadatos EXIF (incluyendo ubicación simulada).
- El procesamiento de guardado se realiza en background para mayor eficiencia.

## Estructura principal
- `tracking.py`: Testeo del tracking.
- `main.py`: Script principal de conteo y tracking con mavros.
- `cacao_crops/`: Carpeta donde se guardan los recortes de objetos.
- Modelos YOLO (`.pt`): Se usan para la detección.

## Requisitos
- Python 3.8+
- OpenCV
- Ultralytics (YOLO)
- Pillow
- piexif

## Testing
1. Coloca tu video en la ruta indicada en el script (`video_path`).
2. Ajusta los parámetros según tus necesidades.
3. Ejecuta el script principal:
   ```bash
   python tracking.py
   ```
## Mavros
El script consulta a mavros la posición del rover a tiempo real y la envía al script de conteo para que pueda guardar los recortes con la ubicación correcta.
Para esto debes tener ya activos la simulación en PX4, qgroundcontrol y la comunicación con mavros.
```bash
   python main.py
   ```