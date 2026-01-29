# YOLOv8 Object Detection Project

A comprehensive implementation of YOLOv8 object detection with training, inference, evaluation, and deployment capabilities.

## 🚀 Features

- **Complete Training Pipeline**: End-to-end training with custom datasets
- **Advanced Data Preprocessing**: Support for multiple formats (COCO, Pascal VOC, YOLO)
- **Real-time Inference**: FastAPI-based API for real-time object detection
- **Model Evaluation**: Comprehensive metrics and performance benchmarking
- **Interactive Notebook**: Jupyter notebook for experimentation and visualization
- **Export Support**: ONNX, TensorRT, and other formats for deployment
- **Docker Ready**: Containerized deployment with docker-compose

## 📁 Project Structure

```
project17_yolov8_object_detection/
├── src/
│   ├── yolov8_detector.py      # Main YOLOv8 pipeline
│   ├── data_preprocessing.py   # Data preparation utilities
│   ├── api.py                  # FastAPI inference service
│   └── evaluation.py           # Model evaluation suite
├── data/                       # Dataset directory
├── models/                     # Trained models
├── runs/                       # Training runs and results
├── notebooks/
│   └── yolov8_interactive.ipynb # Interactive experimentation
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── demo.py                     # Quick demo script
```

## 🛠️ Installation

1. **Clone and setup environment:**
```bash
cd project17_yolov8_object_detection
pip install -r requirements.txt
```

2. **GPU Support (Optional):**
   - Install CUDA-compatible PyTorch if using GPU
   - The code automatically detects and uses GPU if available

## 📊 Quick Start

### 1. Prepare Your Dataset

Organize your data in YOLO format:
```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Create a `data.yaml` file:
```yaml
train: data/train/images
val: data/val/images
test: data/test/images
nc: 3
names: ['class1', 'class2', 'class3']
```

### 2. Train Model

```python
from src.yolov8_detector import YOLOv8Detector

detector = YOLOv8Detector(model_size='yolov8n')
results = detector.train('data/data.yaml', epochs=50, batch_size=16)
```

### 3. Run Inference

```python
# Load trained model
detector.load_model('runs/train/yolov8n_custom/weights/best.pt')

# Detect objects
results = detector.predict('path/to/image.jpg')
detector.visualize_results(results)
```

### 4. Start API Server

```bash
python src/api.py
```

The API will be available at `http://localhost:8000`

## 🔧 API Usage

### Load Model
```bash
POST /load_model
{
  "model_path": "path/to/model.pt",
  "device": "auto"
}
```

### Detect Objects
```bash
POST /detect
{
  "image": "base64_encoded_image",
  "conf_threshold": 0.25,
  "iou_threshold": 0.6
}
```

### Batch Detection
```bash
POST /detect_batch
{
  "images": ["base64_image1", "base64_image2"],
  "conf_threshold": 0.25
}
```

## 📈 Evaluation

Run comprehensive evaluation:

```python
from src.evaluation import run_comprehensive_evaluation

report = run_comprehensive_evaluation(
    model_path='path/to/model.pt',
    data_yaml_path='data/data.yaml',
    test_images_dir='data/test/images'
)
```

## 🎯 Model Zoo

| Model | Size | Parameters | mAP |
|-------|------|------------|-----|
| YOLOv8n | Nano | 3.2M | 37.3 |
| YOLOv8s | Small | 11.2M | 44.9 |
| YOLOv8m | Medium | 25.9M | 50.2 |
| YOLOv8l | Large | 43.7M | 52.9 |
| YOLOv8x | XLarge | 68.2M | 53.9 |

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up --build
```

### ONNX Export

```python
detector.export_model(format='onnx', opset=11, simplify=True)
```

### FastAPI Production

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📊 Performance Benchmarks

| Model | GPU (RTX 3080) | CPU (i7-11700K) |
|-------|----------------|-----------------|
| YOLOv8n | 150 FPS | 25 FPS |
| YOLOv8s | 110 FPS | 18 FPS |
| YOLOv8m | 75 FPS | 12 FPS |
| YOLOv8l | 50 FPS | 8 FPS |
| YOLOv8x | 35 FPS | 6 FPS |

## 🎨 Interactive Notebook

Launch the interactive notebook:

```bash
jupyter notebook notebooks/yolov8_interactive.ipynb
```

Features:
- Step-by-step training walkthrough
- Real-time visualization
- Hyperparameter tuning
- Model comparison
- Performance analysis

## 🧪 Testing

Run unit tests:

```bash
python -m pytest tests/
```

## 📚 Documentation

### Data Formats Supported

1. **YOLO Format**: `[class_id x_center y_center width height]`
2. **COCO Format**: Standard COCO JSON annotations
3. **Pascal VOC**: XML annotation format

### Configuration Options

```python
config = {
    'model_size': 'yolov8n',      # Model variant
    'img_size': 640,              # Input image size
    'batch_size': 16,             # Training batch size
    'epochs': 100,                # Training epochs
    'conf_threshold': 0.25,       # Detection confidence
    'iou_threshold': 0.6,         # NMS IoU threshold
    'device': 'auto'              # cpu/cuda/auto
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the interactive notebook examples
- Review the API documentation

---

**Happy Detecting! 🚀**