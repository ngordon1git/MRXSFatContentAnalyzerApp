#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Clone the repository
git clone https://github.com/ngordon1git/MRXSFatContentAnalyzerApp.git
cd MRXSFatContentAnalyzerApp

# Create a virtual environment
python -m venv MRXSFatAnalyzer

# Activate the virtual environment
MRXSFatAnalyzer/Scripts/activate

# Install prerequisites
pip install --upgrade pip
pip install -r requirements.txt

# Notify user of completion
echo "Setup is complete. The virtual environment is activated. Run your scripts as needed."
