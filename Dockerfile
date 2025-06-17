# Dockerfile
# This Dockerfile builds the container image for Luna's FastAPI application.

# Use a specific Python base image. Python 3.10 is widely compatible.
# Use a slim-buster image for a smaller footprint.
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir ensures a smaller image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your application code into the container
# This includes main.py, chatbot_core.py, admin_api.py, database_manager.py, etc.
COPY . .

# Expose the port your FastAPI application listens on
# This should match the container port in docker-compose.yml's ports mapping
EXPOSE 8000

# Command to run your FastAPI application when the container starts
# `uvicorn main:app --host 0.0.0.0 --port 8000` starts the FastAPI app.
# `--workers 1` is a simple start; for production, you might use Gunicorn with Uvicorn workers.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]