# Quick Start Guide - Face Recognition eKYC

## ⚡ 5-Minute Setup

### Step 1: Install (2 minutes)

```bash
# Clone repository
git clone <repository-url>
cd project22_face_ekyc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Test with Demo (2 minutes)

```bash
# Run demo script
python demo.py
```

Demo includes:
- Face detection
- Face matching
- Liveness detection
- Complete verification workflow

### Step 3: Start API (1 minute)

```bash
# Start API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Test API

### Using cURL

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Face Detection:**
```bash
# Prepare image
import base64
import cv2

image = cv2.imread("test.jpg")
_, buffer = cv2.imencode('.jpg', image)
image_b64 = base64.b64encode(buffer).decode('utf-8')

# Test
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$image_b64\", \"max_faces\": 1}"
```

### Using Python

```python
import requests
import base64
import cv2

# Read and encode image
image = cv2.imread("test.jpg")
_, buffer = cv2.imencode('.jpg', image)
image_b64 = base64.b64encode(buffer).decode('utf-8')

# Call API
response = requests.post(
    "http://localhost:8000/detect",
    json={
        "image": image_b64,
        "max_faces": 1,
        "return_landmarks": True
    }
)

print(response.json())
```

### Using Swagger UI

1. Open http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Fill in parameters
5. Click "Execute"

## 📝 Common Use Cases

### Use Case 1: Verify Identity (eKYC)

```python
from matching import FaceMatcher
from liveness import LivenessDetector
import cv2

# Initialize
matcher = FaceMatcher(ctx_id=0)
liveness_detector = LivenessDetector()

# Load images
selfie = cv2.imread("customer_selfie.jpg")
cccd = cv2.imread("customer_cccd.jpg")

# Verify
result = matcher.verify_identity(
    selfie, cccd,
    liveness_check=True,
    liveness_detector=liveness_detector
)

if result['verified']:
    print(f"✅ Identity verified! Confidence: {result['confidence']:.2%}")
else:
    print(f"❌ Verification failed: {result['errors']}")
```

### Use Case 2: Face Matching Only

```python
from matching import FaceMatcher
import cv2

matcher = FaceMatcher(similarity_threshold=0.6)

image1 = cv2.imread("person1.jpg")
image2 = cv2.imread("person2.jpg")

is_match, similarity, details = matcher.match_faces(image1, image2)

print(f"Match: {is_match}")
print(f"Similarity: {similarity:.4f}")
```

### Use Case 3: Liveness Check

```python
from liveness import TextureAnalyzer
from detection import FaceDetector
import cv2

# Detect face
detector = FaceDetector()
image = cv2.imread("selfie.jpg")
face = detector.get_largest_face(image)

if face:
    # Extract face region
    bbox = face['bbox']
    face_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    # Check liveness
    analyzer = TextureAnalyzer()
    is_live, details = analyzer.analyze_texture(face_crop)
    
    print(f"Live: {is_live}")
    print(f"Blur score: {details['blur_score']:.2f}")
    print(f"Color diversity: {details['color_diversity']:.2f}")
```

## 🐳 Docker Quick Start

```bash
# Build
docker build -t face-ekyc:latest .

# Run
docker run -p 8000:8000 face-ekyc:latest

# Or use docker-compose
docker-compose up -d
```

## 🔧 Troubleshooting

### Issue: "InsightFace model not found"

**Solution:**
InsightFace will download models automatically on first run. Make sure you have internet connection.

### Issue: "CUDA out of memory"

**Solution:**
Set CPU mode in config:
```yaml
performance:
  use_gpu: false
```

Or in code:
```python
detector = FaceDetector(ctx_id=-1)  # -1 = CPU mode
```

### Issue: "No face detected"

**Solution:**
- Check image quality (resolution, lighting)
- Adjust detection threshold in config:
```yaml
models:
  insightface:
    det_thresh: 0.3  # Lower = more sensitive
```

### Issue: "API returns 503"

**Solution:**
Models not loaded. Check logs:
```bash
# Check API logs
tail -f logs/ekyc.log

# Restart API
uvicorn api.main:app --reload
```

## 📊 Expected Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Face Detection | ~50ms | Single face, GPU |
| Face Embedding | ~30ms | 512-dim vector |
| Face Matching | ~100ms | Including detection |
| Liveness Check | ~200ms | Texture analysis only |
| Complete Verification | ~300ms | Match + Liveness |

## 🎯 Next Steps

1. **Customize thresholds** in `config/config.yaml`
2. **Add authentication** to API
3. **Setup monitoring** (Prometheus/Grafana)
4. **Deploy to production** (Docker/K8s)
5. **Integrate with your app**

## 💡 Tips

- Use GPU for better performance (10x faster)
- Adjust thresholds based on your security requirements
- Enable liveness detection for critical operations
- Cache face embeddings to speed up repeated matching
- Use batch processing for multiple faces

## 📚 More Information

- Full documentation: [README.md](README.md)
- API reference: http://localhost:8000/docs
- Configuration guide: [config/config.yaml](config/config.yaml)

## 🆘 Need Help?

- Check [README.md](README.md) for detailed documentation
- Open an issue on GitHub
- Contact: your-email@example.com

---

**Happy Coding! 🚀**
