# Deployment Guide

## 🚀 Deployment Options

This guide covers deploying the AI Ops Dashboard to different environments:

1. **Local Development** - Docker Compose
2. **Cloud VM** - Single server deployment
3. **Kubernetes** - Production cluster
4. **AWS ECS** - Container orchestration

## 🏠 Local Development

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 20GB+ disk space

### Quick Start

```bash
# Clone repository
git clone <your-repo-url>
cd project23_mlops_dashboard

# Configure environment
cp .env.example .env
# Edit .env as needed

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### Accessing Services

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow | http://localhost:5000 | None |
| Dashboard | http://localhost:8501 | None |
| API | http://localhost:8080 | None |
| API Docs | http://localhost:8080/docs | None |
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | None |
| MinIO | http://localhost:9001 | minioadmin/minioadmin |

### Port Configuration

Edit `.env` to change ports:

```bash
# MLflow
MLFLOW_PORT=5000

# Triton
TRITON_HTTP_PORT=8000
TRITON_GRPC_PORT=8001
TRITON_METRICS_PORT=8002

# API
API_PORT=8080
API_METRICS_PORT=9091

# Dashboard
DASHBOARD_PORT=8501

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Storage
POSTGRES_PORT=5432
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001
```

## ☁️ Cloud VM Deployment

### AWS EC2 / Azure VM / GCP Compute Engine

#### Instance Requirements

**Minimum Specs:**
- 4 vCPUs
- 16GB RAM
- 100GB SSD
- Ubuntu 22.04 LTS

**Recommended Specs:**
- 8 vCPUs
- 32GB RAM
- 200GB SSD
- GPU instance for Triton (optional)

#### Installation Steps

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 3. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. Clone repository
git clone <your-repo-url>
cd project23_mlops_dashboard

# 5. Configure environment
cp .env.example .env
nano .env  # Edit as needed

# 6. Start services
docker-compose up -d

# 7. Enable auto-start on reboot
sudo systemctl enable docker
```

#### Security Configuration

```bash
# Configure firewall (UFW)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 5000/tcp    # MLflow
sudo ufw allow 8080/tcp    # API
sudo ufw allow 8501/tcp    # Dashboard
sudo ufw allow 3000/tcp    # Grafana (optional)
sudo ufw enable

# For production, use reverse proxy (Nginx)
sudo apt install nginx -y

# Configure Nginx
sudo nano /etc/nginx/sites-available/mlops
```

**Nginx Configuration:**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # MLflow
    location /mlflow/ {
        proxy_pass http://localhost:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API
    location /api/ {
        proxy_pass http://localhost:8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Dashboard
    location / {
        proxy_pass http://localhost:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/mlops /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Setup SSL (Let's Encrypt)
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

#### Backup Strategy

```bash
# Backup PostgreSQL
docker exec -t postgres pg_dumpall -c -U mlflow > backup_$(date +%Y%m%d).sql

# Backup MinIO data
docker exec minio mc mirror /data /backup/minio_$(date +%Y%m%d)

# Backup model repository
tar -czf models_$(date +%Y%m%d).tar.gz models/

# Automate backups (cron)
crontab -e

# Add line:
0 2 * * * /path/to/backup_script.sh
```

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3+

### Kubernetes Manifests

#### 1. Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlops
```

#### 2. PostgreSQL

```yaml
# k8s/postgres.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: mlops
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: mlops
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: mlflow
        - name: POSTGRES_USER
          value: mlflow
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: mlops
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

#### 3. MLflow

```yaml
# k8s/mlflow.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: mlops
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: your-registry/mlops-mlflow:latest
        env:
        - name: MLFLOW_BACKEND_STORE_URI
          value: postgresql://mlflow:password@postgres:5432/mlflow
        - name: MLFLOW_DEFAULT_ARTIFACT_ROOT
          value: s3://mlflow-artifacts/
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: secret-key
        - name: MLFLOW_S3_ENDPOINT_URL
          value: http://minio:9000
        ports:
        - containerPort: 5000
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: mlops
spec:
  selector:
    app: mlflow
  type: LoadBalancer
  ports:
  - port: 5000
    targetPort: 5000
```

#### 4. Triton

```yaml
# k8s/triton.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton
  namespace: mlops
spec:
  replicas: 3
  selector:
    matchLabels:
      app: triton
  template:
    metadata:
      labels:
        app: triton
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:23.12-py3
        args:
        - tritonserver
        - --model-repository=/models
        - --strict-model-config=false
        - --log-verbose=1
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: grpc
        - containerPort: 8002
          name: metrics
        volumeMounts:
        - name: model-repository
          mountPath: /models
        resources:
          limits:
            nvidia.com/gpu: 1  # Optional GPU
        livenessProbe:
          httpGet:
            path: /v2/health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v2/health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: model-repository
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: triton
  namespace: mlops
spec:
  selector:
    app: triton
  type: LoadBalancer
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  - port: 8001
    targetPort: 8001
    name: grpc
  - port: 8002
    targetPort: 8002
    name: metrics
```

#### 5. FastAPI

```yaml
# k8s/api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: mlops
spec:
  replicas: 4
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: your-registry/mlops-api:latest
        env:
        - name: TRITON_GRPC_URL
          value: triton:8001
        ports:
        - containerPort: 8080
        - containerPort: 9091
          name: metrics
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 3
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
---
apiVersion: v1
kind: Service
metadata:
  name: api
  namespace: mlops
spec:
  selector:
    app: api
  type: LoadBalancer
  ports:
  - port: 8080
    targetPort: 8080
    name: api
  - port: 9091
    targetPort: 9091
    name: metrics
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: mlops
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Deployment Steps

```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Create secrets
kubectl create secret generic postgres-secret \
  --from-literal=password=your-password \
  -n mlops

kubectl create secret generic minio-secret \
  --from-literal=access-key=minioadmin \
  --from-literal=secret-key=minioadmin \
  -n mlops

# 3. Deploy services
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/minio.yaml
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/triton.yaml
kubectl apply -f k8s/api.yaml

# 4. Check status
kubectl get pods -n mlops
kubectl get svc -n mlops

# 5. View logs
kubectl logs -f deployment/mlflow -n mlops

# 6. Access services
kubectl port-forward svc/mlflow 5000:5000 -n mlops
kubectl port-forward svc/api 8080:8080 -n mlops
```

### Helm Chart (Alternative)

```bash
# Create Helm chart
helm create mlops-dashboard

# Edit values.yaml
# ...

# Install
helm install mlops-dashboard ./mlops-dashboard -n mlops

# Upgrade
helm upgrade mlops-dashboard ./mlops-dashboard -n mlops

# Uninstall
helm uninstall mlops-dashboard -n mlops
```

## 📊 Monitoring Setup

### Prometheus in Kubernetes

```yaml
# k8s/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: mlops
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'mlflow'
      static_configs:
      - targets: ['mlflow:5000']
    - job_name: 'triton'
      static_configs:
      - targets: ['triton:8002']
    - job_name: 'api'
      static_configs:
      - targets: ['api:9091']
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: mlops
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
```

## 🔐 Security Best Practices

1. **Use Secrets**: Never commit credentials
2. **Network Policies**: Restrict pod communication
3. **RBAC**: Limit service account permissions
4. **Image Scanning**: Scan Docker images for vulnerabilities
5. **SSL/TLS**: Use HTTPS for external endpoints
6. **Authentication**: Add OAuth for MLflow UI
7. **Monitoring**: Set up alerts for security events

## 📚 Next Steps

- Configure auto-scaling
- Set up CI/CD pipelines
- Implement backup strategies
- Add custom dashboards
- Configure alerts
