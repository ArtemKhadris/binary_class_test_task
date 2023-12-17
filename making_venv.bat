@echo off
setlocal enabledelayedexpansion

REM Set the name of the virtual environment
set VENV_NAME=venv

REM Set the path to the Python executable
set PYTHON_EXECUTABLE=python

REM Check if virtual environment exists, if not, create it
if not exist %VENV_NAME% (
    echo Creating virtual environment...
    %PYTHON_EXECUTABLE% -m venv %VENV_NAME%
)

REM Activate the virtual environment
echo Activating virtual environment...
call %VENV_NAME%\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Virtual environment and dependencies set up successfully.
echo To deactivate the virtual environment, run: deactivate