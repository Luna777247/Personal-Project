# System Architecture

## 🏗️ Overview

The AI Ops Dashboard is a comprehensive MLOps system built on modern infrastructure. It implements the complete machine learning lifecycle from training to production deployment.

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AI Ops Dashboard                             │
│                      MLOps Mini System                               │
└─────────────────────────────────────────────────────────────────────┘

┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   Developer   │      │   Data Team   │      │   ML Engineer │
└───────┬───────┘      └───────┬───────┘      └───────┬───────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Streamlit Dashboard │
                    │   (Port 8501)       │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼────────┐   ┌────────▼─────────┐   ┌───────▼────────┐
│ MLflow Server  │   │   FastAPI API    │   │ Triton Server  │
│  (Port 5000)   │   │   (Port 8080)    │   │  (Ports 8000/  │
│                │   │                  │   │   8001/8002)   │
│ - Experiments  │   │ - REST API       │   │                │
│ - Runs         │   │ - Prometheus     │   │ - gRPC API     │
│ - Registry     │   │   metrics        │   │ - HTTP API     │
│ - Artifacts    │   │ - Health checks  │   │ - Metrics      │
└────────┬───────┘   └──────────────────┘   └────────┬───────┘
         │                                            │
         │                                            │
┌────────▼───────┐                          ┌────────▼───────┐
│   PostgreSQL   │                          │  Model Repo    │
│  (Port 5432)   │                          │  (Volume)      │
│                │                          │                │
│ - Runs         │                          │ - ONNX models  │
│ - Experiments  │                          │ - Configs      │
│ - Metrics      │                          │ - Versions     │
└────────────────┘                          └────────────────┘
         │
┌────────▼───────┐
│     MinIO      │
│  (Ports 9000/  │
│     9001)      │
│                │
│ - Model files  │
│ - Artifacts    │
│ - Datasets     │
└────────────────┘

┌─────────────────────────────────────────────────────────┐
│              Monitoring & Observability                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐         ┌──────────────┐            │
│  │  Prometheus  │────────▶│   Grafana    │            │
│  │ (Port 9090)  │         │ (Port 3000)  │            │
│  │              │         │              │            │
│  │ - Scraping   │         │ - Dashboards │            │
│  │ - Metrics    │         │ - Alerts     │            │
│  │ - Rules      │         │ - Panels     │            │
│  └──────────────┘         └──────────────┘            │
│         ▲                                              │
│         │                                              │
│         └─────────────┬────────────┬──────────────────┤
│                       │            │                   │
│                  MLflow API   Triton API               │
│                   Metrics      Metrics                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  CI/CD Pipeline                          │
│                  (GitHub Actions)                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Code Quality → Unit Tests → Model Training →           │
│  Docker Build → Deploy Staging → Deploy Production →    │
│  Monitor Deployment                                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### 1. Training Flow

```
Developer
   │
   └─▶ Write Training Code
       │
       └─▶ ModelTrainer (src/training/)
           │
           ├─▶ Log Params/Metrics ──▶ MLflow Tracking
           │                            │
           │                            └─▶ PostgreSQL
           │
           └─▶ Save Model Artifacts ──▶ MinIO S3
```

### 2. Registry Flow

```
Trained Model
   │
   └─▶ ModelRegistryManager (src/registry/)
       │
       ├─▶ Register Model ──▶ MLflow Registry
       │                       │
       │                       └─▶ PostgreSQL
       │
       ├─▶ Stage Transition (None → Staging → Production)
       │
       └─▶ Version Management
```

### 3. Deployment Flow

```
Production Model
   │
   └─▶ TritonModelExporter (src/deployment/)
       │
       ├─▶ Load from Registry ──▶ MLflow
       │
       ├─▶ Convert to ONNX ──▶ skl2onnx
       │
       ├─▶ Generate config.pbtxt
       │
       └─▶ Export to Repository ──▶ models/
                                     │
                                     └─▶ Triton loads model
```

### 4. Inference Flow

```
Client Request
   │
   ├─▶ REST API (FastAPI) ──▶ Port 8080
   │   │
   │   └─▶ TritonGRPCClient ──▶ Triton gRPC (Port 8001)
   │                             │
   │                             └─▶ Model Inference
   │                                 │
   │                                 └─▶ Return Prediction
   │
   └─▶ Direct gRPC ──▶ Port 8001
       │
       └─▶ Model Inference
           │
           └─▶ Return Prediction
```

### 5. Monitoring Flow

```
Services (MLflow, Triton, API)
   │
   └─▶ Expose Metrics ──▶ Prometheus (Port 9090)
       │                   │
       │                   ├─▶ Scrape every 15s
       │                   │
       │                   └─▶ Store Time Series
       │
       └─▶ Grafana Queries ──▶ Visualize Dashboards
                                │
                                └─▶ Alerts (Email, Slack)
```

## 🧩 Component Details

### MLflow Server

**Purpose**: Experiment tracking and model registry

**Key Features**:
- Experiment tracking with runs
- Parameter and metric logging
- Artifact storage (models, plots, data)
- Model registry with versioning
- Stage transitions (Staging → Production)
- Model comparison

**Technology**:
- Backend: PostgreSQL
- Artifact Store: MinIO (S3-compatible)
- UI: Web interface on port 5000

### Triton Inference Server

**Purpose**: High-performance model serving

**Key Features**:
- Multi-framework support (ONNX, TensorFlow, PyTorch)
- Dynamic batching for throughput
- GPU acceleration
- gRPC and HTTP APIs
- Concurrent model execution
- Model versioning

**Technology**:
- Platform: NVIDIA Triton
- APIs: gRPC (port 8001), HTTP (port 8000)
- Metrics: port 8002

### FastAPI Service

**Purpose**: REST API wrapper for Triton

**Key Features**:
- RESTful interface
- Pydantic validation
- OpenAPI documentation
- Prometheus metrics
- Health checks
- CORS support

**Technology**:
- Framework: FastAPI + Uvicorn
- Metrics: Prometheus client
- Port: 8080

### Streamlit Dashboard

**Purpose**: Interactive UI for MLOps

**Key Features**:
- Experiment browser
- Model registry viewer
- Deployment management
- Inference testing
- Metrics visualization

**Technology**:
- Framework: Streamlit
- Port: 8501

### PostgreSQL

**Purpose**: MLflow backend storage

**Stores**:
- Experiments metadata
- Run information
- Parameters
- Metrics
- Tags
- Model registry metadata

### MinIO

**Purpose**: S3-compatible artifact storage

**Stores**:
- Model files
- Training artifacts
- Confusion matrices
- Plots and charts
- Datasets

### Prometheus

**Purpose**: Metrics collection

**Monitors**:
- API request counts
- Inference latency
- Active requests
- Model ready status
- Triton statistics

### Grafana

**Purpose**: Metrics visualization

**Features**:
- Pre-built dashboards
- Real-time charts
- Alert rules
- Email/Slack notifications

## 🔒 Security Considerations

### Network Security
- Internal Docker network for service communication
- Exposed ports only for user-facing services
- Optional SSL/TLS for Triton gRPC

### Authentication
- PostgreSQL password authentication
- MinIO access key and secret key
- Grafana admin credentials
- Optional OAuth for MLflow

### Data Security
- Volume mounts for persistent data
- Backup strategies for PostgreSQL and MinIO
- Model versioning for rollback

## 📈 Scalability

### Horizontal Scaling
- Triton: Multiple instances behind load balancer
- FastAPI: Multiple workers with gunicorn
- MLflow: Read replicas for PostgreSQL

### Vertical Scaling
- Triton: GPU acceleration for inference
- PostgreSQL: Increase memory/CPU
- MinIO: Distributed mode for storage

### Kubernetes Deployment
- Helm charts for service deployment
- HPA for auto-scaling
- Persistent volumes for storage
- Ingress for routing

## 🔄 CI/CD Integration

### GitHub Actions Pipeline

```
Code Push
   │
   ├─▶ Code Quality (black, flake8, isort)
   │
   ├─▶ Unit Tests (pytest with coverage)
   │
   ├─▶ Model Training (MLflow service)
   │   │
   │   └─▶ Metric Validation (min thresholds)
   │
   ├─▶ Docker Build (3 images)
   │
   ├─▶ Deploy Staging
   │   │
   │   └─▶ Smoke Tests
   │
   ├─▶ Deploy Production (manual approval)
   │
   └─▶ Monitor Deployment (health checks)
```

## 🎯 Design Patterns

### Microservices Architecture
- Each component is independently deployable
- Service discovery via Docker DNS
- Health checks for reliability

### Repository Pattern
- Model repository in Triton
- Artifact repository in MinIO
- Metadata repository in PostgreSQL

### Observer Pattern
- Prometheus scrapes metrics
- Grafana observes Prometheus
- Alerts notify on thresholds

### Factory Pattern
- ModelTrainer creates different trainers
- TritonExporter creates different exporters

## 📚 References

- **MLflow**: https://mlflow.org
- **Triton**: https://github.com/triton-inference-server
- **FastAPI**: https://fastapi.tiangolo.com
- **Prometheus**: https://prometheus.io
- **Grafana**: https://grafana.com
