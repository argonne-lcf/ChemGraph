FROM continuumio/miniconda3:latest

WORKDIR /app

# System packages required by the scientific Python stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    git \
    cmake \
    pkg-config \
    curl \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Use conda for packages that are typically more reliable from conda-forge
RUN conda install -y -c conda-forge \
    python=3.11 \
    rdkit \
    nwchem \
    && conda clean -afy

# The PyPI CUDA dependency set includes unsupported wheels on Linux ARM.
# Use PyTorch's CPU distribution on ARM; x86 retains its normal PyPI selection.
RUN if [ "$(uname -m)" = "aarch64" ]; then \
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch; \
    fi

COPY . /app

# Resolve ChemGraph, Polar, JupyterLab and TBLite together. Build TBLite from
# source with conservative flags to avoid ABI/symbol issues on ARM.
RUN CFLAGS="-O2 -fno-tree-vectorize" \
    FFLAGS="-O2 -fno-tree-vectorize" \
    python -m pip install --no-cache-dir --no-binary=tblite \
    . jupyterlab -r requirements/mace-polar.txt "tblite==0.4.0"

# Validate calculator runtimes at build time after package install.
RUN which nwchem && python -c "from tblite.ase import TBLite; import graph_longrange" && \
    python -m pip check

# Allow git commands in bind-mounted repo paths inside the container.
RUN git config --system --add safe.directory /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

EXPOSE 8888 8501 9003

# Default container mode: JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--LabApp.token="]
