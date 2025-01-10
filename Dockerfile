# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 7000

# Set environment variables
ENV MODEL_PATH=/app/random_forest_model.pkl

# Command to run the FastAPI application
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "7000"]