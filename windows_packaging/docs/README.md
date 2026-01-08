# Windows Standalone Package for ChemGraph

This directory contains scripts and configuration files for creating a standalone Windows package of ChemGraph that can be easily installed by end users without requiring Python or dependency management.

## Overview

The Windows package includes:
- **Standalone Executable**: A single-file executable containing ChemGraph CLI and all dependencies
- **Windows Installer**: An NSIS-based installer for easy installation
- **Start Menu Integration**: Shortcuts and proper Windows integration
- **PATH Configuration**: Automatic addition to system PATH

## Prerequisites

### For Building the Package

1. **Python 3.10 or higher** - Must be installed and in PATH
2. **PyInstaller 5.13.0 or higher** - Will be installed automatically
3. **NSIS (Nullsoft Scriptable Install System)** - Required for creating the installer
   - Download from: https://nsis.sourceforge.io/Download
   - Make sure `makensis.exe` is in your PATH after installation
4. **EnVar NSIS Plugin** (Optional but recommended)
   - Download from: https://nsis.sourceforge.io/EnVar_plug-in
   - Place `EnVar.dll` in `%NSISDIR%\Plugins\x86-unicode\`

### For End Users

- **Windows 10 or later** (64-bit)
- **No Python installation required** - Everything is bundled
- **Internet connection** - Required for API calls to LLM services

## Building the Package

### Step 1: Build the Standalone Executable

Run the build script to create a standalone executable:

```batch
cd windows_packaging
scripts\build_windows.bat
```

This will:
1. Check Python installation
2. Install/upgrade PyInstaller
3. Install ChemGraph and all dependencies
4. Build the standalone executable using PyInstaller
5. Create output in `windows_packaging\dist\chemgraph\`

**Expected Output:**
- `windows_packaging\dist\chemgraph\chemgraph.exe` - Main executable
- Supporting DLLs and libraries in the same directory

**Testing the Executable:**
```batch
windows_packaging\dist\chemgraph\chemgraph.exe --help
```

### Step 2: Create the Windows Installer

After successfully building the executable, create the installer:

```batch
cd windows_packaging
scripts\build_installer.bat
```

This will:
1. Check NSIS installation
2. Verify the PyInstaller build exists
3. Create the installer using NSIS
4. Generate `ChemGraph-Setup.exe` in the `windows_packaging` directory

**Expected Output:**
- `windows_packaging\ChemGraph-Setup.exe` - Windows installer

## Installation for End Users

1. **Download** `ChemGraph-Setup.exe`
2. **Run** the installer (may require administrator privileges)
3. **Follow** the installation wizard:
   - Accept the license
   - Choose installation directory (default: `C:\Program Files\ChemGraph`)
   - Select components (Core, Documentation, Examples)
   - Choose Start Menu folder
4. **Complete** the installation

After installation, users can:
- Run `chemgraph` from any command prompt (added to PATH)
- Access shortcuts from Start Menu
- Use the desktop shortcut

## Usage

### Command Line Interface

After installation, users can use ChemGraph from any command prompt:

```batch
# Basic usage
chemgraph -q "What is the SMILES string for water?"

# With model selection
chemgraph -q "Optimize methane geometry" -m gpt-4o

# Get help
chemgraph --help

# List available models
chemgraph --list-models

# Check API keys
chemgraph --check-keys
```

### API Key Configuration

Users need to set API keys as environment variables. They can do this by:

1. **Using Command Prompt:**
   ```batch
   setx OPENAI_API_KEY "your_key_here"
   setx ANTHROPIC_API_KEY "your_key_here"
   ```

2. **Using System Properties:**
   - Right-click "This PC" → Properties
   - Advanced system settings → Environment Variables
   - Add new user/system variables

3. **Using PowerShell:**
   ```powershell
   [System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your_key_here", "User")
   ```

## Troubleshooting

### Build Issues

**PyInstaller fails with "ModuleNotFoundError":**
- Ensure all dependencies are installed: `pip install -e .`
- Check that the project is installed in development mode
- Try adding missing modules to `hiddenimports` in `chemgraph.spec`

**NSIS build fails:**
- Verify NSIS is installed and `makensis` is in PATH
- Check that `dist\chemgraph\chemgraph.exe` exists
- Review `build_installer.nsi` for any path issues

**Large executable size:**
- This is normal - scientific computing libraries (numpy, pandas, rdkit, etc.) are large
- Expected size: 500MB - 1GB
- Consider using UPX compression (already enabled in spec file)

### Runtime Issues

**"chemgraph is not recognized as a command":**
- Restart command prompt after installation
- Verify PATH includes installation directory
- Check installation directory exists

**API key not found:**
- Verify environment variables are set correctly
- Restart command prompt after setting variables
- Use `chemgraph --check-keys` to verify

**Import errors at runtime:**
- Some dependencies may need to be added to `hiddenimports` in `chemgraph.spec`
- Rebuild the executable after updating the spec file

## Customization

### Adding an Icon

1. Create or obtain a `.ico` file
2. Update `chemgraph.spec`:
   ```python
   icon='path/to/icon.ico',
   ```
3. Rebuild the executable

### Modifying Installation Options

Edit `build_installer.nsi` to:
- Change default installation directory
- Add/remove components
- Modify shortcuts
- Change installer appearance

### Including Additional Files

Add files to the installer by modifying `build_installer.nsi`:
```nsis
Section "Additional Files"
    SetOutPath "$INSTDIR\additional"
    File /r "path\to\files\*.*"
SectionEnd
```

## File Structure

```
windows_packaging/
├── README.md                 # Main documentation (start here)
├── scripts/                   # Build scripts
│   ├── build_windows.bat      # Builds the standalone executable
│   ├── build_installer.bat    # Creates the Windows installer
│   └── launcher.bat           # Alternative launcher script
├── config/                    # Configuration files
│   ├── chemgraph.spec         # PyInstaller specification
│   ├── build_installer.nsi    # NSIS installer script
│   └── requirements-build.txt # Build dependencies
├── docs/                      # Detailed documentation
│   ├── README.md              # Comprehensive guide (this file)
│   ├── QUICKSTART.md          # Quick reference
│   └── PACKAGING_SUMMARY.md   # Technical overview
├── dist/                      # PyInstaller output (created during build)
│   └── chemgraph/
│       └── chemgraph.exe
└── build/                     # PyInstaller build files (created during build)
```

## Notes

- **First Run**: The first execution may be slower as PyInstaller extracts files to a temporary directory
- **Antivirus**: Some antivirus software may flag PyInstaller executables as suspicious. This is a false positive
- **Dependencies**: All Python dependencies are bundled, but external tools (like NWChem, ORCA) are not included
- **Streamlit UI**: The Streamlit web interface is not included in the standalone package. Users can install the full package for UI access

## Support

For issues related to:
- **Building the package**: Check build logs and ensure all prerequisites are installed
- **Using the package**: Refer to the main ChemGraph documentation
- **ChemGraph functionality**: See the main README.md in the project root

## License

This packaging solution is provided under the same license as ChemGraph (Apache 2.0).

