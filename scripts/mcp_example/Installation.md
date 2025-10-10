# Instalation instructions for ChemGraph on Aurora

```bash
git clone https://github.com/argonne-lcf/ChemGraph
cd ChemGraph
git fetch origin
git checkout mcp_dev

module load frameworks
python3 -m venv /path/to/venv --system-site-packages

source activate /path/to/venv
pip install -e . 
```

Note: We had a conda-based installation and are moving toward using frameworks module. Some dependencies conflict are there (numpy), and are working with Khalid to fix this. This may affect certain simulation software, but examples are good.
