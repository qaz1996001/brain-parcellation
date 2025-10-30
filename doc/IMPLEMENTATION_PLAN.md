# 实施计划与技术规范

## 目录
- [Docker容器化方案](#docker容器化方案)
- [Kubernetes部署配置](#kubernetes部署配置)
- [CI/CD Pipeline设计](#cicd-pipeline设计)
- [监控告警系统](#监控告警系统)
- [安全实施细节](#安全实施细节)
- [性能优化实施](#性能优化实施)

---

## Docker容器化方案

### 1. 多阶段Dockerfile设计

```dockerfile
# syntax=docker/dockerfile:1.4

##############################################
# Stage 1: Builder - 构建应用依赖
##############################################
FROM python:3.11-slim as builder

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装uv包管理器
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 创建虚拟环境并安装依赖
RUN uv sync --frozen --no-dev

##############################################
# Stage 2: AI Models - 下载和优化AI模型
##############################################
FROM builder as models

WORKDIR /models

# 下载AI模型
RUN mkdir -p synthseg wmh cmb && \
    curl -L https://example.com/synthseg_2.0.h5 -o synthseg/model.h5 && \
    curl -L https://example.com/wmh_detector.h5 -o wmh/model.h5

# 可选: 模型优化为ONNX格式
# RUN python -m tf2onnx.convert --saved-model /models/synthseg --output /models/synthseg/model.onnx

##############################################
# Stage 3: Runtime - 最终运行时镜像
##############################################
FROM python:3.11-slim

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从builder复制虚拟环境
COPY --from=builder /build/.venv /app/.venv

# 从models复制AI模型
COPY --from=models /models /app/models

# 复制应用代码
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser code_ai/ /app/code_ai/

# 设置环境变量
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH="/app/models"

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "backend.app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. docker-compose.yml (开发环境)

```yaml
version: '3.9'

services:
  # 后端API服务
  backend:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
    container_name: brain-parcellation-backend
    environment:
      - APP_DEBUG=true
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/medical_ai
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY:-dev-secret-key-change-in-production}
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app/backend:ro
      - ./code_ai:/app/code_ai:ro
      - ./uploads:/app/uploads
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    networks:
      - app-network
    restart: unless-stopped

  # PostgreSQL数据库
  postgres:
    image: postgres:16-alpine
    container_name: brain-parcellation-db
    environment:
      - POSTGRES_DB=medical_ai
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - app-network
    restart: unless-stopped

  # Redis缓存
  redis:
    image: redis:7-alpine
    container_name: brain-parcellation-redis
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD:-redis123}
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - app-network
    restart: unless-stopped

  # Celery异步任务工作器
  celery-worker:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: brain-parcellation-worker
    command: celery -A backend.app.celery worker --loglevel=info --concurrency=4
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/medical_ai
      - REDIS_URL=redis://redis:6379/2
    volumes:
      - ./backend:/app/backend:ro
      - ./code_ai:/app/code_ai:ro
      - ./uploads:/app/uploads
    depends_on:
      - postgres
      - redis
    networks:
      - app-network
    restart: unless-stopped

  # Celery监控 (Flower)
  celery-flower:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: brain-parcellation-flower
    command: celery -A backend.app.celery flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - REDIS_URL=redis://redis:6379/2
    depends_on:
      - redis
      - celery-worker
    networks:
      - app-network
    restart: unless-stopped

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    container_name: brain-parcellation-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
    networks:
      - app-network
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:

networks:
  app-network:
    driver: bridge
```

---

## Kubernetes部署配置

### 1. Deployment配置

```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-api
  namespace: medical-ai
  labels:
    app: backend-api
    version: v2.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: backend-api
  template:
    metadata:
      labels:
        app: backend-api
        version: v2.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: backend-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000

      # Init容器 - 数据库迁移
      initContainers:
      - name: db-migration
        image: registry.example.com/brain-parcellation:v2.0.0
        command: ['uv', 'run', 'alembic', 'upgrade', 'head']
        envFrom:
        - configMapRef:
            name: backend-config
        - secretRef:
            name: backend-secrets

      containers:
      - name: backend
        image: registry.example.com/brain-parcellation:v2.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP

        # 环境变量
        envFrom:
        - configMapRef:
            name: backend-config
        - secretRef:
            name: backend-secrets
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace

        # 资源限制
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"

        # 健康检查
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2

        # 启动探针
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 30

        # 挂载卷
        volumeMounts:
        - name: uploads
          mountPath: /app/uploads
        - name: tmp
          mountPath: /tmp

      volumes:
      - name: uploads
        persistentVolumeClaim:
          claimName: uploads-pvc
      - name: tmp
        emptyDir: {}
---
# k8s/backend-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-api
  namespace: medical-ai
  labels:
    app: backend-api
spec:
  type: ClusterIP
  selector:
    app: backend-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  - port: 9090
    targetPort: 9090
    protocol: TCP
    name: metrics
---
# k8s/backend-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-api-hpa
  namespace: medical-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
      selectPolicy: Min
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: backend-ingress
  namespace: medical-ai
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.medical-ai.example.com
    secretName: api-tls
  rules:
  - host: api.medical-ai.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api
            port:
              number: 80
```

### 2. ConfigMap和Secret

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: backend-config
  namespace: medical-ai
data:
  APP_NAME: "Medical Imaging AI API"
  APP_VERSION: "2.0.0"
  APP_DEBUG: "false"
  APP_ENABLE_DOCS: "true"

  # 数据库配置
  DATABASE_POOL_SIZE: "20"
  DATABASE_MAX_OVERFLOW: "30"
  DATABASE_POOL_RECYCLE: "3600"

  # Redis配置
  REDIS_CACHE_DB: "1"
  REDIS_TASK_DB: "2"

  # 处理配置
  MAX_CONCURRENT_TASKS: "5"
  PROCESSING_TIMEOUT: "3600"
  DEFAULT_DEPTH_NUMBER: "5"

  # 监控配置
  ENABLE_METRICS: "true"
  METRICS_PORT: "9090"
---
# k8s/secret.yaml (需要base64编码)
apiVersion: v1
kind: Secret
metadata:
  name: backend-secrets
  namespace: medical-ai
type: Opaque
data:
  DATABASE_URL: <base64-encoded-value>
  REDIS_URL: <base64-encoded-value>
  SECRET_KEY: <base64-encoded-value>

  # 外部服务
  ORTHANC_URL: <base64-encoded-value>
  ORTHANC_USERNAME: <base64-encoded-value>
  ORTHANC_PASSWORD: <base64-encoded-value>
```

---

## CI/CD Pipeline设计

### 1. GitHub Actions Workflow

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  release:
    types: [created]

env:
  REGISTRY: registry.example.com
  IMAGE_NAME: brain-parcellation

jobs:
  # ========================================
  # Stage 1: 代码质量检查
  # ========================================
  lint-and-format:
    name: Lint and Format Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install UV
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: |
          uv sync --dev

      - name: Run Ruff linter
        run: |
          uv run ruff check . --output-format=github

      - name: Run Ruff formatter
        run: |
          uv run ruff format --check .

      - name: Run MyPy type checker
        run: |
          uv run mypy backend/ code_ai/

  # ========================================
  # Stage 2: 安全扫描
  # ========================================
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Bandit security linter
        run: |
          uv run bandit -r backend/ code_ai/ -f json -o bandit-report.json

      - name: Run pip-audit
        run: |
          uv run pip-audit

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

  # ========================================
  # Stage 3: 单元测试和集成测试
  # ========================================
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install UV
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: |
          uv sync --dev

      - name: Run unit tests
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379/0
        run: |
          uv run pytest tests/unit --cov=backend --cov=code_ai --cov-report=xml --cov-report=html

      - name: Run integration tests
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379/0
        run: |
          uv run pytest tests/integration --cov-append --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
          fail_ci_if_error: true

  # ========================================
  # Stage 4: 构建Docker镜像
  # ========================================
  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [lint-and-format, security-scan, test]
    if: github.event_name != 'pull_request'

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ secrets.REGISTRY_USERNAME }}
          password: ${{ secrets.REGISTRY_PASSWORD }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache
          cache-to: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache,mode=max

      - name: Scan image with Trivy
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.meta.outputs.version }}
          format: 'sarif'
          output: 'trivy-image-results.sarif'

  # ========================================
  # Stage 5: 部署到Kubernetes
  # ========================================
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build]
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://staging-api.medical-ai.example.com

    steps:
      - uses: actions/checkout@v4

      - name: Setup kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'v1.29.0'

      - name: Setup Helm
        uses: azure/setup-helm@v3
        with:
          version: 'v3.13.0'

      - name: Deploy with Helm
        run: |
          helm upgrade --install brain-parcellation ./helm/brain-parcellation \
            --namespace medical-ai-staging \
            --create-namespace \
            --set image.tag=${{ github.sha }} \
            --set environment=staging \
            --values ./helm/values-staging.yaml \
            --wait --timeout 5m

      - name: Run smoke tests
        run: |
          kubectl run smoke-test \
            --namespace medical-ai-staging \
            --image=curlimages/curl \
            --rm --restart=Never \
            --command -- curl -f https://staging-api.medical-ai.example.com/health

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build]
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
      url: https://api.medical-ai.example.com

    steps:
      - uses: actions/checkout@v4

      - name: Setup kubectl
        uses: azure/setup-kubectl@v3

      - name: Setup Helm
        uses: azure/setup-helm@v3

      - name: Deploy with Helm (Canary)
        run: |
          # 金丝雀部署 - 10%流量
          helm upgrade --install brain-parcellation ./helm/brain-parcellation \
            --namespace medical-ai \
            --set image.tag=${{ github.sha }} \
            --set environment=production \
            --set canary.enabled=true \
            --set canary.weight=10 \
            --values ./helm/values-production.yaml \
            --wait --timeout 5m

      - name: Monitor canary metrics
        run: |
          # 监控5分钟
          sleep 300
          # 检查错误率和延迟
          ./scripts/check-canary-metrics.sh

      - name: Promote canary to full deployment
        if: success()
        run: |
          helm upgrade --install brain-parcellation ./helm/brain-parcellation \
            --namespace medical-ai \
            --set image.tag=${{ github.sha }} \
            --set environment=production \
            --set canary.enabled=false \
            --values ./helm/values-production.yaml \
            --wait --timeout 10m

      - name: Rollback on failure
        if: failure()
        run: |
          helm rollback brain-parcellation --namespace medical-ai
```

---

## 监控告警系统

### 1. Prometheus配置

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'medical-ai-prod'
    environment: 'production'

# 告警规则文件
rule_files:
  - 'rules/*.yml'

# 抓取配置
scrape_configs:
  # Kubernetes服务发现
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

  # PostgreSQL监控
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Redis监控
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Node监控
  - job_name: 'node'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)

# AlertManager配置
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### 2. 告警规则

```yaml
# monitoring/prometheus/rules/backend-alerts.yml
groups:
  - name: backend_alerts
    interval: 30s
    rules:
      # API响应时间告警
      - alert: HighAPILatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 1
        for: 5m
        labels:
          severity: warning
          component: backend-api
        annotations:
          summary: "API响应时间过高"
          description: "P95延迟超过1秒,当前值: {{ $value }}秒"

      # 错误率告警
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          /
          sum(rate(http_requests_total[5m]))
          > 0.05
        for: 5m
        labels:
          severity: critical
          component: backend-api
        annotations:
          summary: "API错误率过高"
          description: "5xx错误率超过5%,当前: {{ $value | humanizePercentage }}"

      # CPU使用率告警
      - alert: HighCPUUsage
        expr: |
          (sum(rate(container_cpu_usage_seconds_total{pod=~"backend-.*"}[5m])) by (pod)
          /
          sum(container_spec_cpu_quota{pod=~"backend-.*"}/container_spec_cpu_period{pod=~"backend-.*"}) by (pod))
          > 0.8
        for: 10m
        labels:
          severity: warning
          component: backend-api
        annotations:
          summary: "CPU使用率过高"
          description: "Pod {{ $labels.pod }} CPU使用率超过80%"

      # 内存使用告警
      - alert: HighMemoryUsage
        expr: |
          (container_memory_usage_bytes{pod=~"backend-.*"}
          /
          container_spec_memory_limit_bytes{pod=~"backend-.*"})
          > 0.9
        for: 5m
        labels:
          severity: critical
          component: backend-api
        annotations:
          summary: "内存使用率过高"
          description: "Pod {{ $labels.pod }} 内存使用超过90%"

      # Pod重启告警
      - alert: PodRestartingFrequently
        expr: |
          rate(kube_pod_container_status_restarts_total{pod=~"backend-.*"}[15m]) > 0
        for: 5m
        labels:
          severity: warning
          component: backend-api
        annotations:
          summary: "Pod频繁重启"
          description: "Pod {{ $labels.pod }} 在15分钟内重启了{{ $value }}次"

  - name: database_alerts
    interval: 30s
    rules:
      # 数据库连接数告警
      - alert: HighDatabaseConnections
        expr: |
          pg_stat_database_numbackends{datname="medical_ai"}
          /
          pg_settings_max_connections
          > 0.8
        for: 5m
        labels:
          severity: warning
          component: database
        annotations:
          summary: "数据库连接数过高"
          description: "PostgreSQL连接数超过限制的80%"

      # 数据库慢查询告警
      - alert: SlowDatabaseQueries
        expr: |
          rate(pg_stat_statements_mean_time_seconds{datname="medical_ai"}[5m]) > 1
        for: 10m
        labels:
          severity: warning
          component: database
        annotations:
          summary: "数据库慢查询"
          description: "平均查询时间超过1秒"

  - name: redis_alerts
    interval: 30s
    rules:
      # Redis内存使用告警
      - alert: RedisHighMemoryUsage
        expr: |
          redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
          component: redis
        annotations:
          summary: "Redis内存使用率过高"
          description: "Redis内存使用超过90%"

      # Redis连接数告警
      - alert: RedisHighConnectionCount
        expr: |
          redis_connected_clients > 100
        for: 5m
        labels:
          severity: warning
          component: redis
        annotations:
          summary: "Redis连接数过高"
          description: "Redis连接数超过100"
```

### 3. Grafana仪表板

```json
{
  "dashboard": {
    "title": "Medical AI System Overview",
    "panels": [
      {
        "title": "API Request Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (status)"
          }
        ],
        "type": "graph"
      },
      {
        "title": "API Response Time (P50, P95, P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P99"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))"
          }
        ],
        "type": "singlestat"
      },
      {
        "title": "Active Pods",
        "targets": [
          {
            "expr": "count(kube_pod_info{namespace=\"medical-ai\"})"
          }
        ],
        "type": "singlestat"
      },
      {
        "title": "Database Query Performance",
        "targets": [
          {
            "expr": "rate(pg_stat_statements_calls[5m])",
            "legendFormat": "Queries/sec"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

## 安全实施细节

### 1. Vault密钥管理

```hcl
# terraform/vault.tf
resource "vault_mount" "medical_ai" {
  path        = "medical-ai"
  type        = "kv-v2"
  description = "Medical AI secrets"
}

resource "vault_kv_secret_v2" "database" {
  mount = vault_mount.medical_ai.path
  name  = "database"

  data_json = jsonencode({
    username = var.db_username
    password = var.db_password
    host     = var.db_host
    port     = var.db_port
    database = "medical_ai"
  })
}

# Kubernetes集成
resource "vault_auth_backend" "kubernetes" {
  type = "kubernetes"
}

resource "vault_kubernetes_auth_backend_config" "kubernetes" {
  backend         = vault_auth_backend.kubernetes.path
  kubernetes_host = var.kubernetes_host
}

resource "vault_kubernetes_auth_backend_role" "backend" {
  backend                          = vault_auth_backend.kubernetes.path
  role_name                        = "backend-api"
  bound_service_account_names      = ["backend-sa"]
  bound_service_account_namespaces = ["medical-ai"]
  token_ttl                        = 3600
  token_policies                   = ["backend-policy"]
}
```

### 2. 网络策略

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-api-network-policy
  namespace: medical-ai
spec:
  podSelector:
    matchLabels:
      app: backend-api
  policyTypes:
    - Ingress
    - Egress

  ingress:
    # 只允许Nginx Ingress访问
    - from:
      - namespaceSelector:
          matchLabels:
            name: ingress-nginx
      ports:
      - protocol: TCP
        port: 8000

  egress:
    # 允许访问PostgreSQL
    - to:
      - podSelector:
          matchLabels:
            app: postgres
      ports:
      - protocol: TCP
        port: 5432

    # 允许访问Redis
    - to:
      - podSelector:
          matchLabels:
            app: redis
      ports:
      - protocol: TCP
        port: 6379

    # 允许DNS查询
    - to:
      - namespaceSelector:
          matchLabels:
            name: kube-system
      ports:
      - protocol: UDP
        port: 53
```

---

## 性能优化实施

### 1. 数据库优化脚本

```sql
-- scripts/db-optimization.sql

-- 创建关键索引
CREATE INDEX CONCURRENTLY idx_studies_status_created
ON studies(processing_status, created_at DESC)
WHERE processing_status IN ('pending', 'processing', 'completed');

CREATE INDEX CONCURRENTLY idx_series_study_type
ON series(study_id, series_type)
INCLUDE (series_uid, file_count);

CREATE INDEX CONCURRENTLY idx_results_study_metric
ON processing_results(study_id, metric_type, created_at DESC);

-- 分区表设计
CREATE TABLE studies_partitioned (
    LIKE studies INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- 创建月度分区
CREATE TABLE studies_2024_01 PARTITION OF studies_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- 物化视图
CREATE MATERIALIZED VIEW mv_study_statistics AS
SELECT
    DATE_TRUNC('day', created_at) AS date,
    processing_status,
    COUNT(*) AS study_count,
    AVG(processing_time) AS avg_processing_time
FROM studies
GROUP BY DATE_TRUNC('day', created_at), processing_status
WITH DATA;

-- 定期刷新物化视图
CREATE INDEX ON mv_study_statistics(date, processing_status);
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_study_statistics;

-- 查询优化分析
ANALYZE studies;
ANALYZE series;
ANALYZE processing_results;
```

### 2. Redis缓存策略

```python
# backend/app/cache_strategies.py
from functools import wraps
from typing import Any, Callable, Optional
import hashlib
import orjson
from backend.app.cache import cache_get, cache_set

def smart_cache(
    prefix: str,
    ttl: int = 300,
    key_func: Optional[Callable] = None
):
    """智能缓存装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存key
            if key_func:
                cache_key_parts = key_func(*args, **kwargs)
            else:
                cache_key_parts = f"{args}:{kwargs}"

            cache_key_hash = hashlib.sha256(
                cache_key_parts.encode()
            ).hexdigest()[:16]

            cache_key = f"{prefix}:{cache_key_hash}"

            # 尝试从缓存获取
            cached_value = await cache_get(cache_key)
            if cached_value is not None:
                return cached_value

            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            await cache_set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator

# 使用示例
@smart_cache(prefix="study", ttl=600, key_func=lambda study_id: str(study_id))
async def get_study_details(study_id: str):
    return await db.query(Study).filter(Study.id == study_id).first()
```

---

**文档版本**: v1.0
**最后更新**: 2025-10-15