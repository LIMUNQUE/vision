# Video Tracking & Conteo de Objetos

Este proyecto permite realizar tracking y conteo automático de objetos (por ejemplo, carros o frutos) en videos usando modelos YOLO y ByteTrack. El sistema:

- Procesa video cuadro por cuadro, detectando y traqueando objetos de interés.
- Solo cuenta objetos que cruzan una línea virtual definida en el frame.
- Filtra y descarta objetos cuya bounding box es demasiado pequeña.
- Permite guardar recortes (crops) de los objetos detectados con metadatos EXIF (incluyendo ubicación simulada).
- El procesamiento de guardado se realiza en background para mayor eficiencia.

## Estructura principal
- `tracking.py`: Script principal de conteo y tracking.
- `cacao_crops/`: Carpeta donde se guardan los recortes de objetos.
- Modelos YOLO (`.pt`): Se usan para la detección.

## Requisitos
- Python 3.8+
- OpenCV
- Ultralytics (YOLO)
- Pillow
- piexif

## Ejecución
1. Coloca tu video en la ruta indicada en el script (`video_path`).
2. Ajusta los parámetros según tus necesidades.
3. Ejecuta el script principal:
   ```bash
   python tracking.py
   ```

## Notas
- El sistema está preparado para filtrar por etiquetas y tamaño mínimo de objeto.
- No subas archivos de modelo (`.pt`) ni imágenes de salida al repositorio.
