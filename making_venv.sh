#!/bin/bash

# Set the name of the virtual environment
VENV_NAME=venv

# Set the path to the Python executable
PYTHON_EXECUTABLE=python3

# Check if virtual environment exists, if not, create it
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    $PYTHON_EXECUTABLE -m venv $VENV_NAME
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Virtual environment and dependencies set up successfully."
echo "To deactivate the virtual environment, run: deactivate"