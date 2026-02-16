FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (LightGBM needs libgomp1)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock* /app/

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --no-root

# Copy project
COPY . /app

# Expose the port (Render uses this to verify startup)
EXPOSE 8000

# Run the application (Respect $PORT if set by Render)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
