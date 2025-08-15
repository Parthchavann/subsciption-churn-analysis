#!/bin/bash
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
