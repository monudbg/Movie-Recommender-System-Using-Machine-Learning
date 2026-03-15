#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing requirements..."
pip install -r requirements.txt

echo "Generating Machine Learning Models..."
# This script will run on Render and output models to artifacts/ directory 
python generate_models.py

echo "Running Django collectstatic and migrate..."
cd web_app
python manage.py collectstatic --no-input
python manage.py migrate
