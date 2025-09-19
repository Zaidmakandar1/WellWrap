# WellWrap Deployment Guide

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/wellwrap.git
cd wellwrap

# Install dependencies
make install

# Initialize database
make init-db

# Run the application
make run
```

### Using Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t wellwrap .
docker run -p 5000:5000 wellwrap
```

## 🏗️ Production Deployment

### Prerequisites
- Python 3.8+
- PostgreSQL (recommended for production)
- Redis (optional, for caching)
- Nginx (for reverse proxy)
- SSL certificate

### Environment Variables
Create a `.env` file:
```env
# Application
SECRET_KEY=your-super-secret-key-here
FLASK_ENV=production
DEBUG=False

# Database
DATABASE_URL=postgresql://user:password@localhost/wellwrap

# File Upload
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216

# Security
SESSION_COOKIE_SECURE=True
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax

# ML Features
ENABLE_OCR=True
ENABLE_ADVANCED_ML=True

# Monitoring
SENTRY_DSN=your-sentry-dsn-here
```

### Database Setup
```bash
# PostgreSQL setup
sudo -u postgres createdb wellwrap
sudo -u postgres createuser wellwrap
sudo -u postgres psql -c "ALTER USER wellwrap PASSWORD 'your-password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE wellwrap TO wellwrap;"

# Initialize WellWrap database
python init_database.py --init
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;

    client_max_body_size 16M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /path/to/wellwrap/frontend/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### Systemd Service
Create `/etc/systemd/system/wellwrap.service`:
```ini
[Unit]
Description=WellWrap Healthcare Application
After=network.target

[Service]
Type=exec
User=wellwrap
Group=wellwrap
WorkingDirectory=/opt/wellwrap
Environment=PATH=/opt/wellwrap/venv/bin
ExecStart=/opt/wellwrap/venv/bin/gunicorn --bind 127.0.0.1:5000 --workers 4 --timeout 120 run_app:app
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable wellwrap
sudo systemctl start wellwrap
sudo systemctl status wellwrap
```

## ☁️ Cloud Deployment

### Heroku
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-wellwrap-app

# Set environment variables
heroku config:set SECRET_KEY=your-secret-key
heroku config:set FLASK_ENV=production

# Add PostgreSQL
heroku addons:create heroku-postgresql:hobby-dev

# Deploy
git push heroku main

# Initialize database
heroku run python init_database.py --init
```

### AWS EC2
```bash
# Launch EC2 instance (Ubuntu 20.04 LTS)
# SSH into instance

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv nginx postgresql postgresql-contrib

# Clone and setup
git clone https://github.com/yourusername/wellwrap.git
cd wellwrap
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup database and run
make init-db
make run
```

### Google Cloud Platform
```bash
# Install gcloud CLI
gcloud init

# Create App Engine app
gcloud app create

# Deploy
gcloud app deploy

# View logs
gcloud app logs tail -s default
```

### Docker Swarm
```yaml
# docker-stack.yml
version: '3.8'

services:
  wellwrap:
    image: wellwrap/wellwrap:latest
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres/wellwrap
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
    depends_on:
      - postgres

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=wellwrap
      - POSTGRES_USER=wellwrap
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure

volumes:
  postgres_data:
```

Deploy:
```bash
docker stack deploy -c docker-stack.yml wellwrap
```

### Kubernetes
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wellwrap
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wellwrap
  template:
    metadata:
      labels:
        app: wellwrap
    spec:
      containers:
      - name: wellwrap
        image: wellwrap/wellwrap:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: wellwrap-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: wellwrap-secrets
              key: secret-key
```

## 📊 Monitoring & Logging

### Health Checks
```bash
# Application health
curl -f http://localhost:5000/test

# Database health
python -c "from run_app import db; print('DB OK' if db.engine.execute('SELECT 1').scalar() == 1 else 'DB Error')"
```

### Logging Configuration
```python
# In production, add to run_app.py
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/wellwrap.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
```

### Monitoring with Prometheus
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'wellwrap'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
```

## 🔒 Security Considerations

### SSL/TLS
- Use Let's Encrypt for free SSL certificates
- Configure HSTS headers
- Use secure cookie settings

### Database Security
- Use strong passwords
- Enable SSL connections
- Regular backups
- Limit database user permissions

### Application Security
- Keep dependencies updated
- Use environment variables for secrets
- Implement rate limiting
- Regular security audits

### HIPAA Compliance
- Encrypt data at rest and in transit
- Implement audit logging
- User access controls
- Regular security assessments

## 🔄 Backup & Recovery

### Database Backup
```bash
# PostgreSQL backup
pg_dump -h localhost -U wellwrap wellwrap > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore
psql -h localhost -U wellwrap wellwrap < backup_20231219_120000.sql
```

### File Backup
```bash
# Backup uploads
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz uploads/

# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env instance/
```

## 📈 Scaling

### Horizontal Scaling
- Use load balancer (Nginx, HAProxy)
- Multiple application instances
- Shared database and file storage

### Vertical Scaling
- Increase server resources
- Optimize database queries
- Use caching (Redis, Memcached)

### Performance Optimization
- Enable gzip compression
- Use CDN for static files
- Database indexing
- Connection pooling

---

## 🆘 Troubleshooting

### Common Issues
1. **Database connection errors**: Check DATABASE_URL and credentials
2. **File upload failures**: Verify upload directory permissions
3. **ML processing errors**: Ensure OCR dependencies are installed
4. **Memory issues**: Increase server memory or optimize code

### Debug Mode
```bash
# Enable debug mode (development only)
export FLASK_ENV=development
export DEBUG=True
python run_app.py
```

### Logs Location
- Application logs: `logs/wellwrap.log`
- System logs: `/var/log/syslog`
- Nginx logs: `/var/log/nginx/`

For more help, check the [GitHub Issues](https://github.com/yourusername/wellwrap/issues) or contact support.