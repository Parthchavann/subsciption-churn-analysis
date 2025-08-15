#!/bin/bash
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
