FROM continuumio/miniconda3:latest

WORKDIR /app

COPY . /app

# System packages required by scientific Python stack and headless browser deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    pkg-config \
    curl \
    # Dependencies for headless Chrome (pyppeteer)
    libx11-xcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 \
    libxrandr2 libxrender1 libxss1 libxtst6 libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Use conda for packages that are typically more reliable from conda-forge
RUN conda install -y -c conda-forge \
    python=3.11 \
    rdkit \
    nwchem \
    tblite \
    && conda clean -afy

# Install ChemGraph and UI runtime
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir jupyterlab

# Allow git commands in bind-mounted repo paths inside the container.
RUN git config --system --add safe.directory /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

EXPOSE 8888 8501 9003

# Default container mode: JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--LabApp.token="]
