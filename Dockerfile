# Use Python 3.10 as recommended in the README
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONPATH /app
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE movie_recommender.settings

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user matching Hugging Face Spaces requirements
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy project files
COPY --chown=user . /app/

# Install the local source package (src) in editable mode
RUN pip install --no-cache-dir -e .

# Collect static files for Django
WORKDIR /app/web_app
RUN python manage.py collectstatic --noinput

# Expose the default port for Hugging Face Spaces
EXPOSE 7860

# Command to run the Django application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "movie_recommender.wsgi:application"]
