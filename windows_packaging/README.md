# Windows Standalone Package for ChemGraph

This directory contains all files needed to create a standalone Windows package of ChemGraph that can be easily installed by end users without requiring Python or dependency management.

## Directory Structure

```
windows_packaging/
├── README.md                 # This file (main documentation)
├── scripts/                   # Build scripts
│   ├── build_windows.bat      # Builds the standalone executable
│   ├── build_installer.bat    # Creates the Windows installer
│   └── launcher.bat           # Alternative launcher script
├── config/                    # Configuration files
│   ├── chemgraph.spec         # PyInstaller specification
│   ├── build_installer.nsi    # NSIS installer script
│   └── requirements-build.txt  # Build dependencies
├── docs/                      # Documentation
│   ├── QUICKSTART.md          # Quick reference guide
│   ├── PACKAGING_SUMMARY.md   # Technical overview
│   └── BEST_PRACTICES.md      # Best practices documentation
├── dist/                      # Build output (created during build)
└── build/                     # PyInstaller build files (created during build)
```

## Quick Start

### Step 1: Build the Executable

```batch
cd windows_packaging
scripts\build_windows.bat
```

This creates `dist\chemgraph\chemgraph.exe`

### Step 2: Create the Installer

```batch
scripts\build_installer.bat
```

This creates `ChemGraph-Setup.exe` in the `windows_packaging` directory.

## Prerequisites

### For Building

- **Python 3.10+** - Must be installed and in PATH
- **PyInstaller 5.13.0+** - Installed automatically by build script
- **NSIS** - Required for installer creation
  - Download: https://nsis.sourceforge.io/Download
  - Ensure `makensis.exe` is in PATH

### For End Users

- **Windows 10 or later** (64-bit)
- **No Python required** - Everything is bundled
- **Internet connection** - For API calls to LLM services

## Detailed Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Quick reference for building
- **[PACKAGING_SUMMARY.md](docs/PACKAGING_SUMMARY.md)** - Technical details and architecture
- **[BEST_PRACTICES.md](docs/BEST_PRACTICES.md)** - Best practices and implementation details

## Usage After Installation

Users can run ChemGraph from any command prompt:

```batch
chemgraph -q "What is the SMILES string for water?"
chemgraph --help
chemgraph --list-models
```

## API Key Configuration

Users need to set API keys as environment variables:

```batch
setx OPENAI_API_KEY "your_key_here"
setx ANTHROPIC_API_KEY "your_key_here"
```

## Troubleshooting

See the detailed documentation in the `docs/` directory for:
- Build issues and solutions
- Runtime problems
- Customization options
- Testing procedures

## Support

For issues related to:
- **Building the package**: Check `docs/QUICKSTART.md` and build logs
- **Using the package**: Refer to main ChemGraph documentation
- **ChemGraph functionality**: See main `README.md` in project root
