# Deployment Instructions

This project is currently designed for **local development** and **demo environments**. Below are instructions for both local deployment and guidance for future production deployment.

---

## 1. Local Deployment (Development)

### Prerequisites

- Docker Desktop installed and running
- Python 3.11+ with pip
- Node.js 18+ with npm

### Step-by-Step

```bash
# 1. Start the database
docker compose up -d

# 2. Start the backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Start the frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Verification

| Check                          | URL/Command                          | Expected                      |
|--------------------------------|--------------------------------------|-------------------------------|
| Database running               | `docker compose ps`                  | Both services `running`       |
| Backend health                 | `http://localhost:8000/health`        | `{"status":"ok"}`            |
| Frontend dashboard             | `http://localhost:5173`               | Dashboard with live data      |
| pgAdmin                        | `http://localhost:5050`               | Login page                    |

---

## 2. Production Build (Frontend)

```bash
cd frontend
npm run build
```

Output is written to `frontend/dist/`. This folder contains static files that can be served by any web server (Nginx, Apache, Caddy, etc.).

---

## 3. Production Deployment Guidance

### 3.1 Backend

For production, use Uvicorn with multiple workers behind a reverse proxy:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

> **Important**: The `--reload` flag should NOT be used in production.

Consider:
- Using **Gunicorn** with Uvicorn workers: `gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker`
- Running behind **Nginx** as a reverse proxy
- Using environment variables for all configuration (no hardcoded URLs)
- Enabling HTTPS via SSL certificates

### 3.2 Database

For production PostgreSQL:
- Use managed services (AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL)
- Enable connection pooling (PgBouncer)
- Set up automated backups
- Use strong passwords and restrict network access

### 3.3 Frontend

Serve the `frontend/dist/` static files via:
- **Nginx** (recommended)
- Caddy
- Cloud CDN (AWS CloudFront, Vercel, Netlify)

Example Nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Serve frontend
    location / {
        root /path/to/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to backend
    location /network/ {
        proxy_pass http://127.0.0.1:8000;
    }
    location /sim/ {
        proxy_pass http://127.0.0.1:8000;
    }
    location /metrics/ {
        proxy_pass http://127.0.0.1:8000;
    }
    location /health {
        proxy_pass http://127.0.0.1:8000;
    }

    # WebSocket proxy
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3.4 Docker Compose (Full Stack)

For a fully containerized deployment, extend `docker-compose.yml` to include backend and frontend services:

```yaml
services:
  db:
    # ... existing PostgreSQL config

  backend:
    build: ./backend
    ports: ["8000:8000"]
    depends_on:
      db:
        condition: service_healthy
    env_file: .env
    command: uvicorn main:app --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    ports: ["80:80"]
    depends_on: [backend]
```

> **Note**: This requires adding Dockerfiles to `backend/` and `frontend/` directories (not yet created).

---

## 4. Environment Variables for Production

| Variable          | Dev Value                        | Production Notes                       |
|-------------------|----------------------------------|----------------------------------------|
| `DATABASE_URL`    | `postgresql+asyncpg://...@localhost:5433/...` | Point to production DB  |
| `POSTGRES_USER`   | `routinguser`                    | Use a strong, unique username          |
| `POSTGRES_PASSWORD`| `routingpass`                   | Use a strong, randomly generated password |
| CORS origins      | `localhost:5173`                 | Set to your production domain          |

---

## 5. Current Limitations for Production

- **No Dockerfile** for backend or frontend (only Docker Compose for DB)
- **No health check endpoint** for load balancers (only basic `/health`)
- **No rate limiting** on API endpoints
- **No authentication/authorization** — all endpoints are public
- **WebSocket connections are not authenticated**
- **CORS is restricted to localhost** — must be updated for production domains
- **Database migrations not versioned** — tables are created via `CREATE IF NOT EXISTS`
