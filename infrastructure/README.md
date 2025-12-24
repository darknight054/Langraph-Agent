# DeepSeek-OCR Local Infrastructure

Local Docker setup for DeepSeek-OCR with vLLM backend.

## Prerequisites

- Docker Desktop for Windows (WSL2 backend)
- NVIDIA GPU drivers (latest)
- NVIDIA Container Toolkit
- Git (with LFS support)
- ~50GB disk space (35GB model + 10GB Docker)

## Quick Start

### 1. Verify GPU Support

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. Clone DeepSeek-OCR Source Code

```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
```

### 3. Download Model Weights (~35GB)

```bash
# Create models directory
mkdir -p models

# Option 1: Using Hugging Face CLI (recommended)
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir models/deepseek-ai/DeepSeek-OCR

# Option 2: Using Git LFS
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR models/deepseek-ai/DeepSeek-OCR
```

### 4. Build and Run Docker Container

```bash
# Build the image (first time takes 10-20 minutes)
docker-compose build

# Start the service
docker-compose up -d
```

### 5. Verify

```bash
# Wait 1-2 minutes for model to load, then check health
curl http://localhost:8000/health
```

## Commands

| Action | Command |
|--------|---------|
| Start | `docker-compose up -d` |
| Stop | `docker-compose down` |
| Logs | `docker-compose logs -f` |
| Restart | `docker-compose restart` |
| Rebuild | `docker-compose build --no-cache` |

## API Endpoints

- `GET /health` - Health check
- `POST /ocr/image` - Process single image
- `POST /ocr/pdf` - Process PDF document
- `POST /ocr/batch` - Process multiple files

## Configuration

Environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | 0 | GPU device ID |
| `MODEL_PATH` | /app/models/deepseek-ai/DeepSeek-OCR | Model weights path |
| `MAX_CONCURRENCY` | 3 | Max concurrent requests |
| `GPU_MEMORY_UTILIZATION` | 0.80 | GPU memory usage (80% for 12GB GPUs) |

## Troubleshooting

**GPU not detected:**
- Ensure Docker Desktop uses WSL2 backend
- Install NVIDIA Container Toolkit in WSL2

**Out of memory:**
- Reduce `GPU_MEMORY_UTILIZATION` in docker-compose.yml
- Reduce `MAX_CONCURRENCY` to 1-2

**Build fails:**
- Ensure Docker Desktop is running
- Check available disk space (need 10GB+)
- Try `docker-compose build --no-cache`

## Directory Structure

```
infrastructure/
├── README.md                         # This file
├── docker-compose.yml                # Docker Compose configuration
├── Dockerfile                        # Container build instructions
├── start_server.py                   # FastAPI server entry point
├── custom_config.py                  # OCR configuration (overrides)
├── custom_deepseek_ocr.py            # OCR model implementation (overrides)
├── custom_image_process.py           # Image processing utilities (overrides)
├── custom_run_dpsk_ocr_pdf.py        # PDF OCR script (overrides)
├── custom_run_dpsk_ocr_image.py      # Image OCR script (overrides)
├── custom_run_dpsk_ocr_eval_batch.py # Batch evaluation script (overrides)
├── DeepSeek-OCR/                     # Cloned from github.com/deepseek-ai/DeepSeek-OCR
├── models/                           # Model weights (download via HuggingFace)
└── outputs/                          # OCR output directory
```
