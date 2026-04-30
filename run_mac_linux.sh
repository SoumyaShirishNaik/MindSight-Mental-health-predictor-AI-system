#!/bin/bash
set -e

echo "================================================"
echo "  MindSight - Mental Health AI System"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Install from https://python.org"
    exit 1
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Quick setup
if [ ! -f "models/bert_mental_health/config.json" ]; then
    echo "Setting up demo model..."
    python scripts/quick_setup.py
fi

# Start
echo ""
echo "Starting server at http://localhost:5000"
echo "Press Ctrl+C to stop."
echo ""
python app.py
