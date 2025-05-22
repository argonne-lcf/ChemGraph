FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gfortran \
    liblapack-dev \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with tblite, rdkit and other dependencies that are hard to install with pip
RUN conda install -c conda-forge -c rdkit -c pytorch \
    python=3.11 \
    "pytorch<2.6" \
    cpuonly \
    tblite=0.4.0 \
    rdkit \
    -y

# Install Python dependencies using modified pyproject.toml (excluding problematic packages)
RUN grep -v "tblite\|rdkit\|torch<2.6" pyproject.toml > temp_pyproject.toml && \
    mv temp_pyproject.toml pyproject.toml

# Install packages using pip
RUN pip install --no-cache-dir .

# Install JupyterLab
RUN pip install --no-cache-dir jupyterlab

# Expose JupyterLab port
EXPOSE 8888

# Command to run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--LabApp.token=''"]