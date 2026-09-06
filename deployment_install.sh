#!/bin/bash
# deployment_install.sh - Script to ensure all dependencies are installed

echo "🚀 Installing Python dependencies for deployment..."

# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify critical imports
echo "🔍 Verifying critical imports..."
python -c "import structlog; print('✅ structlog OK')"
python -c "import fastapi; print('✅ fastapi OK')" 
python -c "import uvicorn; print('✅ uvicorn OK')"
python -c "import openai; print('✅ openai OK')"
python -c "import qdrant_client; print('✅ qdrant_client OK')"
python -c "import redis; print('✅ redis OK')"
python -c "import PyPDF2; print('✅ PyPDF2 OK')"
python -c "import pdfplumber; print('✅ pdfplumber OK')"
python -c "import psutil; print('✅ psutil OK')"
python -c "import numpy; print('✅ numpy OK')"

echo "✅ All dependencies verified successfully!"
