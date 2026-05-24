# Pack Web Startup Scripts

## Script Files

| File | Purpose |
|------|---------|
| `start-all.bat` | Start both backend and frontend (RECOMMENDED) |
| `start-backend.bat` | Start backend only |
| `start-frontend.bat` | Start frontend only |
| `stop-all.bat` | Stop all services |

## Usage

### 1. One-Click Start (Recommended)

Double-click `start-all.bat` to start both backend and frontend in separate terminal windows.

### 2. Start Individually

- Double-click `start-backend.bat` - Backend only (FastAPI)
- Double-click `start-frontend.bat` - Frontend only (Vite)

### 3. Stop Services

- Double-click `stop-all.bat` - Stop all backend and frontend processes

## Service URLs

| Service | URL |
|---------|-----|
| Frontend UI | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

## Prerequisites

1. **Anaconda/Miniconda installed**
2. **Conda environment 'pack' created**
   ```bash
   conda create -n pack python=3.9
   conda activate pack
   ```
3. **Python dependencies installed**
   ```bash
   pip install -r requirements.txt
   ```
4. **Node.js and npm installed**
5. **Frontend dependencies installed**
   ```bash
   cd web/frontend
   npm install
   ```
