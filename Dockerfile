# Start with the official Airflow image (Stable version)
FROM apache/airflow:2.7.1

# Switch to root to install system dependencies (if needed)
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user to install Python packages
USER airflow

# Copy the requirements file
COPY requirements.txt /requirements.txt

# Install the Python libraries
RUN pip install --no-cache-dir -r /requirements.txt