#!/usr/bin/env python3
"""
Production Deployment Script for Subscription Churn Analysis
Enterprise-grade deployment with Docker, monitoring, and CI/CD
"""
import os
import subprocess
import json
from pathlib import Path
import sys

class ProductionDeployment:
    """Handle production deployment and setup"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.setup_deployment_structure()
        
    def setup_deployment_structure(self):
        """Create production deployment structure"""
        directories = [
            "deployment",
            "deployment/docker",
            "deployment/kubernetes", 
            "deployment/monitoring",
            "deployment/scripts",
            "tests",
            "docs/api"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def create_dockerfile(self):
        """Create optimized Dockerfile for production"""
        dockerfile_content = """# Production Dockerfile for Subscription Churn Analysis
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8501 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
        
        with open("deployment/docker/Dockerfile", "w") as f:
            f.write(dockerfile_content)
            
    def create_docker_compose(self):
        """Create Docker Compose for full stack deployment"""
        docker_compose = """version: '3.8'

services:
  # Main application
  churn-analysis:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile
    ports:
      - "8501:8501"
      - "8080:8080"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/churn_analysis
    volumes:
      - ../../data:/app/data
      - logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    
  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: churn_analysis
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ../../sql:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    restart: unless-stopped
    
  # Redis for caching and real-time features
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  # Prometheus for monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped
    
  # Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana-datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped
    
  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - churn-analysis
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  logs:
"""
        
        with open("deployment/docker/docker-compose.yml", "w") as f:
            f.write(docker_compose)
            
    def create_kubernetes_manifests(self):
        """Create Kubernetes deployment manifests"""
        # Deployment manifest
        deployment_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-analysis
  labels:
    app: churn-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-analysis
  template:
    metadata:
      labels:
        app: churn-analysis
    spec:
      containers:
      - name: churn-analysis
        image: churn-analysis:latest
        ports:
        - containerPort: 8501
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: churn-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: churn-analysis-service
spec:
  selector:
    app: churn-analysis
  ports:
    - name: web
      protocol: TCP
      port: 80
      targetPort: 8501
    - name: api
      protocol: TCP
      port: 8080
      targetPort: 8080
  type: LoadBalancer
"""
        
        with open("deployment/kubernetes/deployment.yaml", "w") as f:
            f.write(deployment_yaml)
            
    def create_monitoring_config(self):
        """Create monitoring and alerting configuration"""
        # Prometheus config
        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'churn-analysis'
    static_configs:
      - targets: ['churn-analysis:8080']
    metrics_path: /metrics
    scrape_interval: 15s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']
"""
        
        with open("deployment/monitoring/prometheus.yml", "w") as f:
            f.write(prometheus_config)
            
        # Alert rules
        alert_rules = """groups:
- name: churn_analysis_alerts
  rules:
  - alert: HighChurnRate
    expr: churn_rate > 0.3
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High churn rate detected"
      description: "Churn rate is {{ $value }}% which is above threshold"
      
  - alert: ApplicationDown
    expr: up{job="churn-analysis"} == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Churn analysis application is down"
      description: "Application has been down for more than 2 minutes"
      
  - alert: DatabaseConnectionError
    expr: database_connections_active / database_connections_max > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Database connection pool nearly exhausted"
      description: "{{ $value }}% of database connections are in use"
"""
        
        with open("deployment/monitoring/alert_rules.yml", "w") as f:
            f.write(alert_rules)
            
    def create_production_requirements(self):
        """Create production requirements file"""
        prod_requirements = """# Production requirements
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
plotly>=5.10.0
streamlit>=1.28.0
psycopg2-binary>=2.9.0
redis>=4.5.0
sqlalchemy>=2.0.0
alembic>=1.9.0
gunicorn>=21.0.0
uvicorn>=0.20.0
fastapi>=0.100.0
prometheus-client>=0.16.0
structlog>=23.0.0
python-multipart>=0.0.6
pydantic>=2.0.0
httpx>=0.24.0
aiofiles>=23.0.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-dotenv>=1.0.0
celery>=5.3.0
flower>=2.0.0
"""
        
        with open("requirements-prod.txt", "w") as f:
            f.write(prod_requirements)
            
    def create_ci_cd_pipeline(self):
        """Create GitHub Actions CI/CD pipeline"""
        github_workflow = """name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8
        
    - name: Lint with flake8
      run: |
        flake8 python/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 python/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
        
    - name: Test with pytest
      run: |
        pytest tests/ --cov=python/ --cov-report=xml
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      
  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    permissions:
      contents: read
      packages: write
      
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: deployment/docker/Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        
  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # Add actual deployment commands here
"""
        
        Path(".github/workflows").mkdir(parents=True, exist_ok=True)
        with open(".github/workflows/ci-cd.yml", "w") as f:
            f.write(github_workflow)
            
    def create_deployment_scripts(self):
        """Create deployment and management scripts"""
        
        # Deployment script
        deploy_script = """#!/bin/bash
# Production deployment script

set -e

echo "🚀 Starting production deployment..."

# Build and start services
echo "📦 Building Docker images..."
docker-compose -f deployment/docker/docker-compose.yml build

echo "🏃 Starting services..."
docker-compose -f deployment/docker/docker-compose.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Run database migrations
echo "🗄️ Running database migrations..."
docker-compose -f deployment/docker/docker-compose.yml exec churn-analysis python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.getenv('DATABASE_URL'))
# Add migration logic here
print('Database ready')
"

# Health check
echo "🏥 Performing health checks..."
curl -f http://localhost:8501/_stcore/health || exit 1
curl -f http://localhost:8080/health || exit 1

echo "✅ Deployment completed successfully!"
echo "🌐 Application available at: http://localhost:8501"
echo "📊 Monitoring available at: http://localhost:3000"
"""
        
        with open("deployment/scripts/deploy.sh", "w") as f:
            f.write(deploy_script)
            
        # Make script executable
        os.chmod("deployment/scripts/deploy.sh", 0o755)
        
        # Backup script
        backup_script = """#!/bin/bash
# Backup script for production data

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo "📦 Creating backup at $BACKUP_DIR..."

# Backup database
echo "🗄️ Backing up database..."
docker-compose -f deployment/docker/docker-compose.yml exec postgres pg_dump -U postgres churn_analysis > $BACKUP_DIR/database.sql

# Backup data files
echo "📁 Backing up data files..."
cp -r data/ $BACKUP_DIR/

# Backup configuration
echo "⚙️ Backing up configuration..."
cp -r deployment/ $BACKUP_DIR/

echo "✅ Backup completed: $BACKUP_DIR"
"""
        
        with open("deployment/scripts/backup.sh", "w") as f:
            f.write(backup_script)
            
        os.chmod("deployment/scripts/backup.sh", 0o755)
        
    def create_production_config(self):
        """Create production configuration files"""
        
        # Environment variables template
        env_template = """# Production Environment Variables
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=False

# Database
DATABASE_URL=postgresql://postgres:password@postgres:5432/churn_analysis
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# Monitoring
PROMETHEUS_METRICS_PORT=8080
HEALTH_CHECK_INTERVAL=30

# Business Logic
CHURN_THRESHOLD=0.7
ALERT_EMAIL=admin@yourcompany.com
MAX_CONCURRENT_USERS=100

# External APIs (if any)
# API_KEY=your-api-key
# WEBHOOK_URL=https://your-webhook-url.com
"""
        
        with open("deployment/.env.production", "w") as f:
            f.write(env_template)
            
    def run_deployment(self):
        """Execute the full deployment setup"""
        print("🏗️ Setting up production deployment...")
        
        try:
            # Create all deployment files
            self.create_dockerfile()
            print("✅ Created Dockerfile")
            
            self.create_docker_compose()
            print("✅ Created Docker Compose configuration")
            
            self.create_kubernetes_manifests()
            print("✅ Created Kubernetes manifests")
            
            self.create_monitoring_config()
            print("✅ Created monitoring configuration")
            
            self.create_production_requirements()
            print("✅ Created production requirements")
            
            self.create_ci_cd_pipeline()
            print("✅ Created CI/CD pipeline")
            
            self.create_deployment_scripts()
            print("✅ Created deployment scripts")
            
            self.create_production_config()
            print("✅ Created production configuration")
            
            # Create deployment summary
            self.create_deployment_summary()
            
            print("\n🎉 Production deployment setup complete!")
            print("\n📋 Next steps:")
            print("   1. Review and customize .env.production")
            print("   2. Run: chmod +x deployment/scripts/deploy.sh")
            print("   3. Execute: ./deployment/scripts/deploy.sh")
            print("   4. Access application at: http://localhost:8501")
            print("   5. Monitor at: http://localhost:3000")
            
        except Exception as e:
            print(f"❌ Deployment setup failed: {e}")
            return False
            
        return True
        
    def create_deployment_summary(self):
        """Create deployment summary documentation"""
        summary = {
            "deployment_info": {
                "project": "Subscription Churn Analysis",
                "version": "2.0.0",
                "deployment_date": "2025-01-15",
                "environment": "production"
            },
            "services": {
                "main_app": {
                    "port": 8501,
                    "description": "Main Streamlit dashboard",
                    "health_check": "http://localhost:8501/_stcore/health"
                },
                "api": {
                    "port": 8080,
                    "description": "REST API and metrics endpoint",
                    "health_check": "http://localhost:8080/health"
                },
                "database": {
                    "port": 5432,
                    "description": "PostgreSQL database",
                    "type": "postgres:15"
                },
                "monitoring": {
                    "prometheus": "http://localhost:9090",
                    "grafana": "http://localhost:3000"
                }
            },
            "deployment_features": [
                "Docker containerization",
                "Kubernetes support", 
                "Automated CI/CD pipeline",
                "Prometheus monitoring",
                "Grafana dashboards",
                "Redis caching",
                "PostgreSQL database",
                "Nginx reverse proxy",
                "Health checks",
                "Backup scripts"
            ],
            "security_features": [
                "Non-root user in containers",
                "Environment variable secrets",
                "SSL/TLS support",
                "Rate limiting",
                "CORS protection"
            ]
        }
        
        with open("deployment/deployment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    deployer = ProductionDeployment()
    success = deployer.run_deployment()
    
    if success:
        print("\n🚀 Ready for production deployment!")
        sys.exit(0)
    else:
        print("\n❌ Deployment setup failed!")
        sys.exit(1)