# Instalation instructions for ChemGraph on Aurora

```bash
git clone https://github.com/argonne-lcf/ChemGraph
cd ChemGraph
git fetch origin
git checkout mcp_dev

# Create and activate a Python virtual environment
module load frameworks
python3 -m venv /path/to/venv --system-site-packages
source activate /path/to/venv

# Install ChemGraph in editable mode
pip install -e . 
```

Note: The installation may take up to 10 minutes due to several backend simulation packages being installed.