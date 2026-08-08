# 🚗 Vehicle Detection

A Streamlit web app that detects and classifies vehicles in images and videos using **two YOLO models working together**:

- **YOLOv8n** pretrained on COCO — detects general vehicle classes (car, bus, motorcycle, bicycle, etc.)
- **A custom-trained YOLO model** — detects and classifies vehicles with domain-specific labels

The two models' predictions are merged with an IoU-based filter: custom-model detections are always kept, and COCO detections are added only when they don't overlap with a custom detection. This combines the accuracy of the custom model with the broader coverage of the COCO-trained model.

## ✨ Features

- 🖼️ Supports both **image** and **video** uploads (`jpg`, `jpeg`, `png`, `mp4`, `avi`, `mov`, `mkv`)
- 🔀 Merges results from two YOLO models using Intersection-over-Union (IoU) deduplication
- 🏷️ Remaps raw class names to custom labels (e.g. `bus` → `heavy_truck`, `motorcycle`/`bicycle` → `two_wheeled_vehicle`)
- 🎚️ Adjustable confidence threshold via a slider
- 📊 Detection results table with class and confidence for each detected object
- ⬇️ Download button for processed videos
- ⚡ Model caching with `st.cache_resource` for fast repeated inference

## 🧠 How It Works

1. Both models run inference on the uploaded frame/image.
2. Boxes from the custom model are added to the final detection list first.
3. Boxes from the COCO model are added only if their IoU with every custom-model box is below `0.5` (i.e. they don't already describe an object the custom model found).
4. Final detections are drawn on the image/video and summarized in a results table.

## 📁 Project Structure

```
vehicle-detection/
├── app.py                          # Streamlit application
├── data.yaml                       # Dataset/class configuration for the custom model
├── requirements.txt                # Python dependencies
├── yolov8n.pt                      # Pretrained YOLOv8n weights (COCO)
├── runs/detect/vehicle_detector/   # Custom model training output (weights, etc.)
├── data/                           # Dataset files
└── .streamlit/                     # Streamlit configuration
```

## 🔧 Requirements

- Python 3.9+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Streamlit
- OpenCV
- NumPy / Pandas

## 🚀 Installation

```bash
git clone https://github.com/aliqlmix/vehicle-detection.git
cd vehicle-detection
pip install -r requirements.txt
```

> **Note:** `requirements.txt` doesn't currently list `pandas`, which is used for the results table — install it separately if needed:
> ```bash
> pip install pandas
> ```

## ⚙️ Configuration

Before running the app, open `app.py` and set the path to your custom-trained model weights:

```python
MODEL1_PATH = 'yolov8n.pt'
MODEL2_PATH = 'runs/detect/vehicle_detector/weights/best.pt'  # update to your own path
```

You can also adjust:
- `COCO_CLASSES` — which COCO class IDs to keep from the YOLOv8n model
- `DEFAULT_CONF` — default confidence threshold
- `LABEL_MAP` — how raw class names are renamed in the output

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then, in the browser:
1. Upload an image or video.
2. Adjust the confidence threshold slider if needed.
3. View the annotated output and the detection details table.
4. For videos, download the processed file once inference is complete.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👤 Author

**Ali Gholami** — [@aliqlmix](https://github.com/aliqlmix)
